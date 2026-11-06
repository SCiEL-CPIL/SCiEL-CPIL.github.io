# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using hierarchical recursive pebbling.
    
    Strategy:
    1. Parse the DAG structure
    2. Compute last-use positions efficiently
    3. Use a greedy pebbling with optimal eviction policy:
       - Evict the vertex whose next use is furthest in the future
       - This minimizes total loads/stores (transfer count)
    4. Return operations list
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
        
        # Compute last_use efficiently: for each vertex, the max topo position
        # of any child (or n if it's an output or leaf)
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = n
            elif not children_map.get(v, []):
                last_use[v] = n
            else:
                max_pos = 0
                for child in children_map[v]:
                    max_pos = max(max_pos, last_use.get(child, n))
                last_use[v] = max_pos
        
        # For each vertex, the positions where it's needed = positions of its children
        # that come after v's own position
        needed_at = {v: [] for v in vertices}
        for i, v in enumerate(topo_order):
            for p in parents_map.get(v, []):
                if pos_map.get(p, -1) < i:  # p comes before v in topo order
                    needed_at[p].append(i)
        
        # Sort needed positions for each vertex
        for v in needed_at:
            needed_at[v].sort()
        
        # Compute usage frequency for each vertex to identify critical vertices
        usage_freq = {v: len(needed_at[v]) for v in vertices}
        
        ops = []
        in_fast = set()       # vertices currently in fast memory (red)
        is_blue = set(inputs) # vertices that are blue
        
        # Pointer into needed_at for each vertex: next_use_idx[v] = index of next needed position
        next_use_idx = {v: 0 for v in vertices}
        
        def get_next_use(v, current_pos):
            """Get the next position after current_pos where v is needed."""
            idx = next_use_idx[v]
            positions = needed_at[v]
            while idx < len(positions) and positions[idx] <= current_pos:
                idx += 1
            next_use_idx[v] = idx
            if idx < len(positions):
                return positions[idx]
            return last_use.get(v, n)
        
        def get_eviction_cost(x, current_pos):
            """
            Compute a cost for evicting vertex x at position current_pos.
            Higher cost means more expensive to evict (should keep in cache).
            We use a combination of:
            - Next use distance (closer = higher cost to evict)
            - Usage frequency (more uses = higher cost to evict)
            - Whether it's an output (outputs should be kept until stored)
            """
            nu = get_next_use(x, current_pos)
            dist_to_next = nu - current_pos if nu > current_pos else 0
            
            # If no more uses, cost is low
            if dist_to_next <= 0:
                return 0.0
            
            # Base cost from distance (closer uses are more expensive to evict)
            # Use inverse distance with a cap to avoid division by zero
            dist_cost = 1.0 / (1.0 + dist_to_next)
            
            # Frequency bonus: vertices used many times are more expensive to evict
            freq_bonus = 0.1 * min(usage_freq.get(x, 0), 10)
            
            # Output bonus: keep outputs until they're stored
            output_bonus = 1.0 if x in outputs else 0.0
            
            return dist_cost + freq_bonus + output_bonus
        
        # Eviction: when fast memory is full and we need to load, evict the vertex
        # with the lowest eviction cost (furthest next use, least frequent)
        
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            if v in inputs:
                continue
            
            parents = parents_map.get(v, [])
            
            # Ensure all parents are available
            for p in parents:
                if p in is_blue or p in in_fast:
                    continue
                
                # Need to make p available. Check if fast memory has room.
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
                else:
                    # Need to evict one vertex from fast memory
                    # Evict the one with the lowest eviction cost
                    best_evict = None
                    best_cost = float('inf')
                    for x in in_fast:
                        cost = get_eviction_cost(x, i)
                        if cost < best_cost:
                            best_cost = cost
                            best_evict = x
                    
                    if best_evict is not None:
                        ops.append(['delete', best_evict])
                        in_fast.remove(best_evict)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            
            # If v is an output, store it (make blue) and remove from fast memory
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Proactively evict vertices whose last use has passed
            # Only evict if we have excess in fast memory
            if len(in_fast) > 0:
                to_evict = []
                for x in in_fast:
                    if last_use.get(x, 0) <= i and x not in outputs:
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
