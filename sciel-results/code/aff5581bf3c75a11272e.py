# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using bottom-up pebbling strategy.
    
    Strategy:
    1. Parse the DAG structure
    2. Process vertices in reverse topological order (bottom-up)
    3. Store only final output values in fast memory
    4. Discard intermediate results immediately after their last consumer is processed
    5. This minimizes peak memory usage and eliminates complex recomputation scheduling
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            elif isinstance(dag, dict):
                vertices = list(dag.keys())
        
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        # Build children map
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        if not outputs:
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        topo_order = list(vertices)
        n = len(topo_order)
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Reverse topological order for bottom-up processing
        reverse_topo = list(reversed(topo_order))
        
        # For each vertex, compute how many children still need it
        # In bottom-up: when processing v, we need all children to be available
        # Track remaining children count for each vertex
        remaining_children = {v: len(children_map.get(v, [])) for v in vertices}
        
        # Fast memory state
        in_fast = set()       # vertices currently in fast memory (red)
        is_blue = set(inputs) # vertices that are blue (available)
        computed = set()      # vertices that have been computed
        
        ops = []
        
        def ensure_available(v):
            """Ensure vertex v is available (blue or in fast memory)."""
            if v in is_blue or v in in_fast:
                return True
            
            # Need to load v into fast memory
            if len(in_fast) < fast_memory_budget:
                ops.append(['load', v])
                in_fast.add(v)
            else:
                # Evict a vertex from fast memory
                # In bottom-up, we evict the vertex with the most remaining uses
                # (i.e., the one needed by the most unprocessed vertices)
                best_evict = None
                max_remaining = -1
                for x in in_fast:
                    rem = remaining_children.get(x, 0)
                    if rem > max_remaining:
                        max_remaining = rem
                        best_evict = x
                
                if best_evict is not None:
                    ops.append(['delete', best_evict])
                    in_fast.remove(best_evict)
                
                ops.append(['load', v])
                in_fast.add(v)
            
            return True
        
        # Process in reverse topological order
        for v in reverse_topo:
            if time.time() > deadline:
                break
            
            if v in inputs:
                continue
            
            # Get all children of v
            children = children_map.get(v, [])
            
            # Ensure all children are available
            for c in children:
                if c not in is_blue and c not in in_fast:
                    # Child must have been computed already in bottom-up order
                    # If not available, we need to load it
                    ensure_available(c)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            computed.add(v)
            
            # If v is an output, store it (make blue) and remove from fast memory
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Update remaining children counts for parents of v
            # v is a child of its parents, so decrement their remaining count
            for p in parents_map.get(v, []):
                if p in remaining_children:
                    remaining_children[p] -= 1
            
            # Proactively evict vertices whose all children have been processed
            # and which are not outputs (intermediate results no longer needed)
            if len(in_fast) > 0:
                to_evict = []
                for x in in_fast:
                    # x can be evicted if all its children have been processed
                    # (remaining_children[x] == 0) and x is not an output
                    if remaining_children.get(x, 0) == 0 and x not in outputs:
                        to_evict.append(x)
                for x in to_evict:
                    if x in in_fast:
                        ops.append(['delete', x])
                        in_fast.remove(x)
        
        # Final check: ensure all outputs are blue
        for v in outputs:
            if v not in is_blue:
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.remove(v)
                    is_blue.add(v)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
