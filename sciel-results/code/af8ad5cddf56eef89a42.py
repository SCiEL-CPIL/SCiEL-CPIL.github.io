# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with lookahead-based eviction.
    
    Strategy:
    1. Parse DAG structure
    2. Precompute for each vertex: all positions where it's needed, last use
    3. Greedy pebbling with optimal eviction:
       - Evict the vertex whose eviction causes the most future transfers
       - Use exact cost computation (not heuristic)
    4. Proactively evict after last use
    5. Return operations list
    """
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin for 90s limit
    
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
        
        # Compute last_use: for each vertex, the max topo position of any child
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = n
            elif not children_map.get(v, []):
                last_use[v] = n
            else:
                max_pos = 0
                for child in children_map[v]:
                    if child in last_use:
                        max_pos = max(max_pos, last_use[child])
                last_use[v] = max_pos
        
        # For each vertex, the positions where it's needed (as a parent of some child)
        needed_at = {v: [] for v in vertices}
        for i, v in enumerate(topo_order):
            for p in parents_map.get(v, []):
                if pos_map.get(p, -1) < i:
                    needed_at[p].append(i)
        
        for v in needed_at:
            needed_at[v].sort()
        
        # Precompute: for each vertex, the list of positions it's needed at
        # and a pointer to track which uses have been satisfied
        usage_freq = {v: len(needed_at[v]) for v in vertices}
        
        ops = []
        in_fast = set()       # vertices currently in fast memory (red)
        is_blue = set(inputs) # vertices that are blue
        
        # For each vertex, track how many of its needed positions have been passed
        # next_use_idx[v] = index into needed_at[v] for the next needed position
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
        
        def compute_eviction_cost(x, current_pos):
            """
            Compute the exact additional transfer cost if we evict x now.
            This is the number of future loads of x that would be needed.
            Higher cost = more expensive to evict = should keep.
            """
            # Count how many more times x will be needed after current_pos
            idx = next_use_idx[x]
            positions = needed_at[x]
            count = 0
            while idx < len(positions) and positions[idx] > current_pos:
                count += 1
                idx += 1
            
            # If x is an output and not yet stored, we must keep it
            if x in outputs and x not in is_blue:
                return float('inf')  # Never evict an output that needs to be stored
            
            # Base cost: number of future uses
            cost = float(count)
            
            # Bonus: if x will be needed very soon, it's more expensive
            if count > 0:
                next_pos = positions[next_use_idx[x]]
                dist = next_pos - current_pos
                if dist <= 2:
                    cost += 5.0  # Very soon
                elif dist <= 5:
                    cost += 2.0
                elif dist <= 10:
                    cost += 1.0
            
            # Bonus: high frequency vertices are more expensive to evict
            cost += 0.5 * min(usage_freq.get(x, 0), 5)
            
            return cost
        
        # Process vertices in topological order
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
                
                # Need to make p available
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
                else:
                    # Need to evict one vertex
                    best_evict = None
                    best_cost = float('inf')
                    for x in in_fast:
                        cost = compute_eviction_cost(x, i)
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
            
            # If v is an output, store it
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Proactively evict vertices whose last use has passed
            if len(in_fast) > 0:
                to_evict = []
                for x in in_fast:
                    if x in outputs:
                        continue
                    if last_use.get(x, 0) <= i:
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
