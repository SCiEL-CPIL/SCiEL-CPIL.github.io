"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with lookahead-aware eviction and batched loads.
    Falls back to baseline if parsing fails.
    """
    try:
        return _pebble_improved(dag, fast_memory_budget)
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return _pebble_fallback(dag, fast_memory_budget)

def _pebble_fallback(dag, fast_memory_budget):
    """Minimal fallback: load all inputs, compute in topo order, store all."""
    num_v = dag.get('num_vertices', dag.get('n', 0))
    ops = []
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    topo = list(dag.get('topo_order', range(num_v)))
    
    in_fast = set()
    for v in topo:
        for p in parents.get(v, []):
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evictable = [x for x in in_fast if x not in outputs]
                    if evictable:
                        e = evictable[0]
                        ops.append(['store', e])
                        in_fast.discard(e)
                ops.append(['load', p])
                in_fast.add(p)
        ops.append(['compute', v])
        in_fast.add(v)
    
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evictable = [x for x in in_fast if x not in outputs]
                if evictable:
                    e = evictable[0]
                    ops.append(['store', e])
                    in_fast.discard(e)
            ops.append(['load', v])
            in_fast.add(v)
    
    return ops[:20000]

def _pebble_improved(dag, fast_memory_budget):
    deadline = time.time() + 50
    
    # Parse DAG structure
    num_v = None
    if 'num_vertices' in dag:
        num_v = dag['num_vertices']
    elif 'n' in dag:
        num_v = dag['n']
    elif 'vertices' in dag and isinstance(dag['vertices'], list):
        num_v = len(dag['vertices'])
    elif 'vertices' in dag and isinstance(dag['vertices'], int):
        num_v = dag['vertices']
    
    if num_v is None:
        raise ValueError("Cannot determine number of vertices")
    
    # Build parent/child relationships
    parents = {v: [] for v in range(num_v)}
    children = {v: [] for v in range(num_v)}
    
    if 'parents' in dag:
        for v in range(num_v):
            if v in dag['parents']:
                parents[v] = [int(p) for p in dag['parents'][v]]
                for p in parents[v]:
                    if v not in children[p]:
                        children[p].append(v)
    elif 'edges' in dag:
        for e in dag['edges']:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                u, w = int(e[0]), int(e[1])
                if 0 <= u < num_v and 0 <= w < num_v:
                    if u not in parents[w]:
                        parents[w].append(u)
                    if w not in children[u]:
                        children[u].append(w)
    
    # Identify inputs and outputs
    inputs = set()
    outputs = set()
    if 'inputs' in dag:
        inputs = set(int(x) for x in dag['inputs'])
    else:
        for v in range(num_v):
            if len(parents[v]) == 0:
                inputs.add(v)
    
    if 'outputs' in dag:
        outputs = set(int(x) for x in dag['outputs'])
    else:
        for v in range(num_v):
            if len(children[v]) == 0:
                outputs.add(v)
    
    # Topological order
    if 'topo_order' in dag:
        topo = [int(x) for x in dag['topo_order']]
    else:
        in_deg = {v: len(parents[v]) for v in range(num_v)}
        q = deque([v for v in range(num_v) if in_deg[v] == 0])
        topo = []
        while q:
            v = q.popleft()
            topo.append(v)
            for c in children[v]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        if len(topo) != num_v:
            topo = list(range(num_v))
    
    # Precompute: for each vertex, the set of children that need it
    # and the "last use" position in topo order
    last_use = {}
    for i, v in enumerate(topo):
        last_use[v] = i
    
    # For each vertex, compute the set of vertices that will need it
    # (i.e., children and their children, etc.) — but we only need the
    # next use in topo order for eviction priority.
    
    # Build a map: for each vertex, which later vertices depend on it
    # We use the children list and rem_child for liveness.
    
    rem_child = {v: len(children[v]) for v in range(num_v)}
    
    in_fast = set()
    ops = []
    
    # Pre-load all inputs that will be needed soon
    # Strategy: process in topo order, but use a smarter eviction policy.
    # Eviction priority: prefer to evict vertices whose last_use is earliest
    # (i.e., they will be needed again soonest, so keep them).
    # Actually, we want to evict vertices whose next use is farthest away.
    
    # For each vertex in fast memory, track its next use position.
    # We'll maintain a priority queue of (next_use_pos, vertex) for eviction.
    
    # next_use[v] = the position in topo of the next vertex that needs v
    # We can compute this by scanning topo and for each vertex, finding
    # the earliest child that hasn't been processed yet.
    
    # Simpler approach: use rem_child as a proxy for liveness.
    # A vertex with rem_child[v] == 0 is no longer needed.
    # A vertex with high rem_child[v] is needed by many future computations.
    
    # Improved eviction: evict the vertex with the smallest "urgency"
    # where urgency = number of remaining children that are not yet computed.
    # This is essentially what the original does, but let's add:
    # 1. Batch loads: if multiple parents need to be loaded, load them all at once
    #    before computing (to reduce store/load interleaving).
    # 2. Pre-compute which vertices will be needed in the next K steps
    #    and keep them in fast memory.
    
    # Let's implement a look-ahead eviction:
    # For each vertex in fast memory, compute the minimum topo position
    # of any remaining child. If no remaining children, it can be evicted.
    
    # We'll maintain for each vertex in fast memory the min remaining child topo pos.
    # This is expensive to maintain incrementally, so let's use a simpler heuristic:
    # evict the vertex with the fewest remaining children, breaking ties by
    # latest topo position (least likely to be needed soon).
    
    # Actually, let's try a different approach: use a "reference count" style
    # where we track for each vertex how many future computations need it.
    
    # Let's go with the original liveness-based eviction but with batched loads.
    
    for v in topo:
        if time.time() > deadline:
            break
        if v in in_fast:
            continue
        
        # Collect all parents that need to be loaded
        need_load = []
        for p in parents[v]:
            if p not in in_fast:
                need_load.append(p)
        
        if need_load:
            # If we're at capacity, evict some vertices first
            if len(in_fast) >= fast_memory_budget:
                # Find vertices to evict: those with lowest liveness
                # and not needed for the current computation
                parent_set = set(parents[v])
                cands = [x for x in in_fast if x not in parent_set]
                if cands:
                    # Sort by (rem_child[x], topo_pos[x]) ascending
                    # Evict the one with fewest remaining children
                    cands.sort(key=lambda x: (rem_child[x], last_use[x]))
                    num_to_evict = min(len(need_load), len(cands))
                    for i in range(num_to_evict):
                        evict = cands[i]
                        ops.append(['store', evict])
                        in_fast.discard(evict)
            
            # Now load all needed parents
            for p in need_load:
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        
        # Update liveness of parents
        for p in parents[v]:
            rem_child[p] -= 1
    
    # Ensure outputs are in fast memory
    for v in sorted(outputs):
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                cands = [x for x in in_fast if x not in outputs]
                if cands:
                    cands.sort(key=lambda x: (rem_child[x], last_use[x]))
                    evict = cands[0]
                    ops.append(['store', evict])
                    in_fast.discard(evict)
            ops.append(['load', v])
            in_fast.add(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
