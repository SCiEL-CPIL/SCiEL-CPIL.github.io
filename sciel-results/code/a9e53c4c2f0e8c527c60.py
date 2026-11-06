"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    DEADLINE = 50.0  # Wall-clock limit with safety margin

    try:
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | inputs | outputs
        
        # Topological Sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
            
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort() # Deterministic order
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
            
        if len(topo_order) != len(all_vertices):
            # Fallback if cycle or error
            topo_order = list(all_vertices)

        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use for each vertex
        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                if topo_index[c] > max_idx:
                    max_idx = topo_index[c]
            last_use[v] = max_idx

        ops = []
        fast_mem = set()
        
        def ensure(v):
            if v in fast_mem:
                return
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex
                ops.append(['load', v])
                fast_mem.add(v)
            else:
                for p in ps:
                    ensure(p)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                
                # Post-compute cleanup: Delete dead nodes
                # A node is dead if its last_use is less than the current vertex's index
                # We don't have current index here easily, so we do it in the main loop.
                
                # If over budget, evict
                if len(fast_mem) > fast_memory_budget:
                    _evict()

        def _evict():
            # Evict node with max last_use (furthest in future)
            # Exclude outputs
            max_lu = -1
            victim = None
            for u in fast_mem:
                if u in outputs:
                    continue
                if last_use[u] > max_lu:
                    max_lu = last_use[u]
                    victim = u
            
            if victim:
                ops.append(['delete', victim])
                fast_mem.discard(victim)
            else:
                # If only outputs are in fast_mem and we are over budget, 
                # we are stuck. But outputs must be kept.
                # If budget < num_outputs, we can't satisfy the constraint?
                # Assume budget is sufficient for outputs.
                pass

        # Main loop
        for v in topo_order:
            if time.time() > DEADLINE:
                break
            
            ensure(v)
            
            # Clean up dead nodes: last_use[u] < topo_index[v]
            for u in list(fast_mem):
                if u in outputs:
                    continue
                if last_use[u] < topo_index[v]:
                    ops.append(['delete', u])
                    fast_mem.discard(u)
            
            # Check budget again after cleanup
            if len(fast_mem) > fast_memory_budget:
                _evict()

        # Ensure outputs are in fast_mem
        for o in outputs:
            if o not in fast_mem:
                ensure(o)
                # Note: ensure(o) might trigger evictions, which might evict other outputs?
                # _evict skips outputs. So outputs are safe.
                # But if we evict a non-output, and then need an output, it's fine.
                
                # If we evicted an output (which we shouldn't), we'd have a problem.
                # But _evict skips outputs.

        if len(ops) <= 20000:
            return ops
        else:
            # Fallback: Simple topological compute (no deletes, no re-computes)
            # This uses min ops, but might exceed memory budget?
            # If memory budget is small, this will fail verification.
            # But it's a valid fallback for op limit.
            ops_simple = []
            for v in topo_order:
                ps = parents.get(v, [])
                if not ps:
                    ops_simple.append(['load', v])
                else:
                    ops_simple.append(['compute', v])
            
            if len(ops_simple) <= 20000:
                return ops_simple
            
            return ops[:20000]

    except Exception:
        return []
# EVOLVE-BLOCK-END
