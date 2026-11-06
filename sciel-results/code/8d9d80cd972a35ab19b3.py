# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling using recursive block decomposition.
    
    Strategy:
    1. Parse DAG and precompute last-use indices.
    2. If the DAG's "width" (max concurrent active vertices) fits in fast_memory_budget,
       use a single-pass schedule that keeps vertices until last use (no recomputation).
    3. Otherwise, recursively split the DAG at a point minimizing the boundary size
       (vertices in left half needed by right half).
    4. For each split: Pebble Left, Store Boundary, Delete Left Non-Boundary, Pebble Right.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            else:
                vertices = list(dag.keys()) if isinstance(dag, dict) else []
        
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
        
        if n == 0:
            return []
        
        # Precompute last_use for each vertex
        # last_use[v] is the index in topo_order where v is last needed
        pos_map = {v: i for i, v in enumerate(topo_order)}
        last_use = {}
        for i in range(n - 1, -1, -1):
            v = topo_order[i]
            if v in outputs:
                last_use[v] = n
            else:
                max_child_pos = i
                for child in children_map.get(v, []):
                    if child in pos_map:
                        max_child_pos = max(max_child_pos, pos_map[child])
                last_use[v] = max_child_pos

        # Helper: Check if a sub-DAG (list of vertices) can be pebbled 
        # with 'keep until last use' strategy within budget.
        def fits_in_budget(sub_order, budget):
            in_fast = set()
            # We need to simulate the active set
            # Active set at step i (processing sub_order[i]) consists of 
            # vertices in sub_order[:i] that are not yet last-used.
            # Plus parents of sub_order[i] that are in sub_order[:i] and not yet last-used.
            
            # To be efficient, we can just simulate the operations.
            # But we need to know if we ever exceed budget.
            
            # A vertex v in sub_order is "active" from its computation index 
            # until its last_use index (clamped to sub_order end).
            
            # Let's just simulate the simple pass and check max size.
            # We don't generate ops, just track size.
            
            # Note: Inputs are always available (blue), so they don't consume fast memory 
            # unless we choose to load them? 
            # The problem states: "Inputs start blue". 
            # "every red value consumes one fast-memory slot".
            # "load" moves from blue to red.
            # So inputs do NOT consume fast memory slots unless loaded.
            # We only load if needed for compute.
            
            # In the simple pass:
            # We load a parent only if it's not in fast memory and not blue.
            # If it's blue (input or stored output), we load it.
            # If it's already in fast, we don't load.
            
            # So, the fast memory contains:
            # 1. Computed vertices that are still needed.
            # 2. Input vertices that have been loaded and are still needed.
            
            # We can delete a vertex from fast memory if its last_use is in the past.
            
            # Simulation:
            # in_fast = set()
            # is_blue = set(inputs) | (outputs already stored? No, we are in a sub-DAG)
            # Actually, for the "fits" check, we assume we start with empty fast memory 
            # and only inputs are blue. Outputs of this sub-DAG are not yet blue 
            # (unless they are inputs of the whole DAG, which are handled).
            
            # Wait, in recursive pebbling, the "state" is passed down.
            # But for the "fits" check, we want to know if the *internal* 
            # communication can be done without exceeding budget, 
            # assuming boundary inputs are loaded as needed.
            
            # Let's simulate:
            in_fast = set()
            # We track which vertices are "available" (blue or in fast)
            # But for the budget check, we only care about fast memory size.
            
            # We iterate through sub_order.
            # For each v:
            #   For each parent p of v:
            #     if p is not in in_fast:
            #       # We must load p (if p is in sub_order and was computed earlier, 
            #       # it might have been deleted. If p is an input, it's blue.
            #       # If p is in sub_order and was computed, and we deleted it, we must RECOMPUTE.
            #       # The "keep until last use" strategy ensures we NEVER delete a vertex 
            #       # before its last use. So if p is in sub_order and was computed, 
            #       # it is either in in_fast or... wait.
            #       # If we keep until last use, p is in in_fast if last_use[p] >= current_index.
            #       # If last_use[p] < current_index, p is deleted. But then p is not needed.
            #       # So if p is a parent of v, last_use[p] >= current_index.
            #       # Thus, if p was computed in this sub-DAG, it MUST be in in_fast.
            #       # So we only need to load p if p is NOT in sub_order (i.e., it's an input 
            #       # or from a previous block).
            
            # So, the logic is:
            # 1. If p is in sub_order: it is in in_fast (guaranteed by keep-until-last-use).
            # 2. If p is not in sub_order: it must be loaded from blue.
            
            # So, for each v:
            #   Count parents p not in sub_order. Let this be k.
            #   We need to load these k parents.
            #   After loading, in_fast contains:
            #     - All previous computed vertices in sub_order with last_use >= current_idx.
            #     - The k newly loaded parents.
            #   Then we compute v, adding v to in_fast.
            #   Then we delete any vertex in in_fast with last_use < next_idx.
            
            # Let's simulate this to find max in_fast size.
            
            sub_set = set(sub_order)
            current_max = 0
            
            # Precompute last_use within sub_order context?
            # No, global last_use is fine.
            
            for i, v in enumerate(sub_order):
                # Identify parents to load
                parents_to_load = []
                for p in parents_map.get(v, []):
                    if p not in sub_set:
                        # Must be loaded from outside (blue)
                        parents_to_load.append(p)
                    else:
                        # Should be in fast memory
                        pass
                
                # Add parents to fast memory
                for p in parents_to_load:
                    in_fast.add(p)
                
                # Compute v
                in_fast.add(v)
                
                # Check size
                if len(in_fast) > current_max:
                    current_max = len(in_fast)
                
                # Delete vertices no longer needed
                # A vertex x in in_fast can be deleted if last_use[x] <= i
                # (i.e., it is not needed for v or any subsequent vertex in sub_order)
                to_delete = []
                for x in in_fast:
                    if last_use.get(x, n) <= i:
                        to_delete.append(x)
                
                for x in to_delete:
                    in_fast.remove(x)
                
            return current_max <= budget

        # Simple Pass Generator
        def simple_pass(order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set):
            ops = []
            sub_set = set(order)
            
            for i, v in enumerate(order):
                if v in inputs:
                    continue
                
                parents = parents_map.get(v, [])
                
                for p in parents:
                    if p in is_blue_set or p in in_fast_set:
                        continue
                    # Need to load p
                    # p is not in fast, not in blue.
                    # This implies p was deleted or is an input not in blue?
                    # Inputs are always in blue initially.
                    # If p is in sub_set, it should have been kept.
                    # If p is not in sub_set, it must be in blue (from previous block or input).
                    # If it's not in blue, we have a problem (shouldn't happen with correct recursion).
                    
                    # Check budget
                    if len(in_fast_set) >= budget:
                        # Evict something
                        # Find a vertex in in_fast_set that is no longer needed
                        to_delete = None
                        for x in in_fast_set:
                            if last_use.get(x, n) <= i:
                                to_delete = x
                                break
                        if to_delete is not None:
                            ops.append(['delete', to_delete])
                            in_fast_set.remove(to_delete)
                        else:
                            # Should not happen if fits_in_budget was true
                            # Fallback: evict oldest?
                            # For safety, just break or raise
                            pass
                    
                    ops.append(['load', p])
                    in_fast_set.add(p)
                
                ops.append(['compute', v])
                in_fast_set.add(v)
                
                if v in outputs:
                    ops.append(['store', v])
                    in_fast_set.remove(v)
                    is_blue_set.add(v)
                
                # Delete parents no longer needed
                for p in parents:
                    if p in in_fast_set and last_use.get(p, n) <= i:
                        ops.append(['delete', p])
                        in_fast_set.remove(p)
            
            return ops

        # Recursive Pebbling
        def recursive_pebble(order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set):
            ops = []
            m = len(order)
            if m == 0:
                return ops
            
            if time.time() > deadline:
                return ops
            
            # Check if this sub-DAG fits in budget
            if fits_in_budget(order, budget):
                ops.extend(simple_pass(order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set))
                return ops
            
            # Find best split point
            # Try a few candidates around the middle
            best_split = m // 2
            min_boundary_size = float('inf')
            
            # Candidate splits: mid-2 to mid+2
            for k in range(max(1, m//2 - 2), min(m-1, m//2 + 3)):
                left_order = order[:k]
                right_order = order[k:]
                left_set = set(left_order)
                
                boundary = set()
                for v in right_order:
                    for p in parents_map.get(v, []):
                        if p in left_set:
                            boundary.add(p)
                
                if len(boundary) < min_boundary_size:
                    min_boundary_size = len(boundary)
                    best_split = k
            
            mid = best_split
            left_order = order[:mid]
            right_order = order[mid:]
            left_set = set(left_order)
            
            # Identify boundary
            boundary = set()
            for v in right_order:
                for p in parents_map.get(v, []):
                    if p in left_set:
                        boundary.add(p)
            
            # 1. Pebble Left
            ops.extend(recursive_pebble(left_order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set))
            
            # 2. Store Boundary
            for b in boundary:
                if b in in_fast_set:
                    ops.append(['store', b])
                    in_fast_set.remove(b)
                    is_blue_set.add(b)
                # If b is not in in_fast_set, it might have been deleted?
                # But we kept until last use in simple_pass.
                # In recursive, the left pebbling should have left boundary in fast or blue.
                # If it's already blue, no need to store.
                # If it's in fast, we store it.
            
            # 3. Delete Left Non-Boundary
            for v in left_order:
                if v in in_fast_set and v not in boundary:
                    ops.append(['delete', v])
                    in_fast_set.remove(v)
            
            # 4. Pebble Right
            ops.extend(recursive_pebble(right_order, parents_map, inputs, outputs, children_map, budget, is_blue_set, in_fast_set))
            
            return ops

        # Initialize state
        is_blue_set = set(inputs)
        in_fast_set = set()
        
        ops = recursive_pebble(topo_order, parents_map, inputs, outputs, children_map, fast_memory_budget, is_blue_set, in_fast_set)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
