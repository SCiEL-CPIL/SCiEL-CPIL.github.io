"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

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
    DEADLINE = 90.0  # wall-clock limit from evaluator
    SAFETY_MARGIN = 5.0  # seconds to spare

    # Try to import baseline as fallback
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order (assuming parents are already in topo order or we compute it)
        # Compute in-degree for topological sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        # Inputs have no parents, so in_degree = 0
        for v in inputs:
            in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Simple strategy: process in topological order
        # For each vertex, ensure its parents are in fast memory, compute it, 
        # then decide whether to keep it or delete it.
        
        # A better strategy: use a pebbling algorithm that minimizes transfers.
        # Strategy: 
        # 1. For each vertex in topo order, if not in fast memory and needed, load/compute it.
        # 2. Keep in fast memory only what's needed for remaining computations and outputs.
        
        ops = []
        fast_mem = set()  # vertices currently in fast memory (red)
        
        # Precompute: for each vertex, which descendants (including itself) need it
        # i.e., which vertices' computation or output depends on this vertex
        needed_by = defaultdict(set)
        for v in all_vertices:
            ps = parents.get(v, [])
            for p in ps:
                needed_by[p].add(v)
        
        # Determine which vertices must end blue: outputs
        must_be_blue = set(outputs)
        
        # Process in topological order
        for v in topo_order:
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex - start blue
                # Load into fast memory
                if v not in fast_mem:
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                # Compute vertex - need all parents in fast memory
                for p in ps:
                    if p not in fast_mem:
                        # Need to load parent (recompute if necessary)
                        # For simplicity, just load (assume it can be reloaded from slow memory)
                        ops.append(['load', p])
                        fast_mem.add(p)
                
                # Compute v
                ops.append(['compute', v])
                fast_mem.add(v)
                
                # Optionally delete parents that are no longer needed
                for p in ps:
                    # Check if p is needed by any other vertex not yet processed
                    still_needed = False
                    for other in needed_by[p]:
                        if other not in fast_mem and other not in must_be_blue:
                            # Check if other is in topo_order and not yet processed
                            if topo_order.index(other) > topo_order.index(v) if other in topo_order else True:
                                still_needed = True
                                break
                    # Also check if p is an output
                    if p in must_be_blue:
                        still_needed = True
                    
                    if not still_needed:
                        ops.append(['delete', p])
                        fast_mem.discard(p)
        
        # Ensure all outputs are in fast memory
        for o in outputs:
            if o not in fast_mem:
                ops.append(['load', o])
                fast_mem.add(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, try baseline or truncate
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        # Fallback to baseline if available
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
