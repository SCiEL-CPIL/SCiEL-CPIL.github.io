"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 80.0  # wall-clock limit with safety margin
    
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        if not parents and not outputs:
            return []
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order using Kahn's algorithm
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = sorted([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    queue.sort()
        
        if len(topo_order) != len(all_vertices):
            # Cycle detected or disconnected; add remaining vertices
            remaining = all_vertices - set(topo_order)
            topo_order.extend(sorted(remaining))
        
        topo_index = {v: i for i, v in enumerate(topo_order)}
        n = len(topo_order)
        
        # Compute last_use: the maximum topo_index of any child, or own index if leaf
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, n - 1)
        
        # For outputs, last_use should be n-1 (they must be blue at end)
        for o in outputs:
            last_use[o] = n - 1
        
        # Compute recomputation cost: number of ancestors (approximate by depth)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
        
        # Count how many times each vertex is needed in the future
        # This helps decide which vertices to keep
        future_needs = defaultdict(int)
        for v in all_vertices:
            if v in children:
                for c in children[v]:
                    future_needs[v] += 1
        
        # Track which vertices have been stored to slow memory
        stored = set()
        
        ops = []
        fast_mem = set()
        
        def get_eviction_candidate(current_idx):
            """
            Find the best vertex to evict from fast_mem.
            Priority:
            1. Vertices whose last_use < current_idx (dead) - evict these first
            2. Among live vertices, prefer those with:
               - Smaller future_needs (less likely to be needed again)
               - Smaller depth (cheaper to recompute)
               - Earlier last_use (needed sooner, so less critical to keep? No, later last_use means we need it further out, so we can afford to evict earlier ones)
            Actually, we want to keep vertices that are needed soon or are expensive to recompute.
            So evict vertices with:
               - Latest last_use? No, that means we need it far in the future, so we should keep it.
               - Earliest last_use? That means we need it soon, so we should keep it.
            Wait, this is confusing. Let me think again.
            
            We want to minimize total transfers. A vertex that is needed many times in the future
            should be kept. A vertex that is expensive to recompute should be kept.
            
            So the eviction cost (how bad it is to evict) is:
              - High if future_needs[v] is high (will be needed many more times)
              - High if depth[v] is high (expensive to recompute)
              - Low if last_use[v] is close to current_idx (will be needed soon, but if we evict, we'll reload/recompute soon anyway)
            
            Actually, if last_use[v] is close, we'll need it soon, so evicting it means we'll need to reload/recompute it soon, which is bad.
            If last_use[v] is far, we might not need it for a while, so evicting it now saves memory for more urgent needs.
            
            So eviction priority (lower score = better to evict):
              - Lower future_needs
              - Lower depth
              - Higher last_use (further in the future, so less urgent)
            """
            candidates = []
            for v in fast_mem:
                if v in outputs:
                    # Outputs must be blue at end. Avoid evicting unless necessary.
                    # Give them a high eviction cost (low priority to evict)
                    continue
                # Calculate eviction score: lower is better to evict
                # We want to evict vertices that are cheap to recompute and not needed soon
                score = depth[v] * 10 + (n - last_use[v]) * 1  # depth matters more
                # Actually, let's use: prefer evicting low depth, and among those, prefer evicting those with earlier last_use
                # Wait no: if last_use is earlier, we need it sooner, so we should NOT evict it.
                # If last_use is later, we need it further out, so we CAN evict it.
                # So higher last_use = safer to evict.
                candidates.append((depth[v], last_use[v], v))
            
            if not candidates:
                # All are outputs. Must evict one.
                # Pick the one with smallest depth (easiest to recompute)
                candidates = [(depth[v], last_use[v], v) for v in fast_mem]
            
            if not candidates:
                return None
            
            # Sort: prefer evicting low depth, then high last_use (further in future)
            candidates.sort(key=lambda x: (x[0], -x[1]))
            return candidates[0][2]
        
        def make_space(needed=1):
            """Free up 'needed' slots in fast_mem by evicting vertices."""
            evicted = 0
            while len(fast_mem) + needed > fast_memory_budget and evicted < needed:
                cand = get_eviction_candidate(len(topo_order) - 1)  # use max index as current
                if cand is None:
                    break
                ops.append(['delete', cand])
                fast_mem.discard(cand)
                evicted += 1
            return evicted >= needed
        
        def ensure_in_fast_mem(v, current_idx):
            """
            Ensure vertex v is in fast_mem.
            Returns True if successful.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            if not ps:
                # Input vertex
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space(1):
                        return False
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # Computed vertex: need to recompute it
                # First ensure all parents are in fast_mem
                for p in ps:
                    if p not in fast_mem:
                        if not ensure_in_fast_mem(p, current_idx):
                            return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space(1):
                        return False
                
                ops.append(['compute', v])
                fast_mem.add(v)
                stored.add(v)  # Mark as stored so we could load it later
                return True
        
        # Main loop: process vertices in topological order
        for i, v in enumerate(topo_order):
            if time.time() - start_time > DEADLINE:
                break
            
            if v not in fast_mem:
                if not ensure_in_fast_mem(v, i):
                    break
            
            # Proactively delete vertices that are no longer needed
            # A vertex u is no longer needed if last_use[u] < i and u is not an output
            to_delete = []
            for u in list(fast_mem):
                if u != v and u not in outputs and last_use[u] < i:
                    to_delete.append(u)
            
            for u in to_delete:
                ops.append(['delete', u])
                fast_mem.discard(u)
        
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                if not ensure_in_fast_mem(o, n - 1):
                    # Force evict if needed
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate(n - 1)
                        if cand is not None:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    ensure_in_fast_mem(o, n - 1)
        
        # Validate: all outputs must be in fast_mem
        for o in outputs:
            if o not in fast_mem:
                # This shouldn't happen, but if it does, return empty
                if pebble_baseline:
                    return pebble_baseline(dag, fast_memory_budget)
                return []
        
        if len(ops) <= 20000:
            return ops
        else:
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
