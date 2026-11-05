"""Pebbling algorithm for communication-minimizing DAG evaluation."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with bounded fast memory to minimize communication.
    
    Strategy: 
    1. Parse DAG.
    2. Topological sort.
    3. Greedy pebbling:
       - For each vertex in topo order:
         - Load parents into fast memory if not present.
         - If fast memory is full, evict a vertex that is no longer needed 
           (all its children are computed). 
           If no such vertex exists, evict the one needed latest.
         - Compute vertex.
    4. Ensure all outputs are in fast memory (blue) at the end.
    """
    deadline = time.time() + 50  # 50s safety margin
    
    # --- Parse DAG ---
    vertices = []
    parents_map = {}
    children_map = defaultdict(list)
    is_output = {}
    
    try:
        if isinstance(dag, list):
            for item in dag:
                if isinstance(item, dict):
                    v_id = item.get('id', item.get('vertex', item.get('name', None)))
                    pars = item.get('parents', item.get('parent', []))
                    out = item.get('is_output', item.get('output', False))
                elif isinstance(item, (tuple, list)):
                    if len(item) >= 3:
                        v_id, pars, out = item[0], item[1], item[2]
                    elif len(item) == 2:
                        v_id, pars = item[0], item[1]
                        out = False
                    else:
                        continue
                else:
                    continue
                
                if v_id is None:
                    continue
                    
                vertices.append(v_id)
                parents_map[v_id] = list(pars) if pars else []
                is_output[v_id] = bool(out)
                
                for p in parents_map[v_id]:
                    children_map[p].append(v_id)
    except Exception:
        # Fallback: assume simple structure or return empty
        return []

    if not vertices:
        return []

    # --- Topological Sort ---
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    topo_order = []
    queue = sorted([v for v in vertices if in_degree[v] == 0])
    
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
        queue.sort()
        
    # If cycle or missing nodes, add remaining
    if len(topo_order) < len(vertices):
        remaining = set(vertices) - set(topo_order)
        topo_order.extend(sorted(remaining))

    # Identify inputs and outputs
    inputs = [v for v in topo_order if not parents_map.get(v, [])]
    outputs = [v for v in topo_order if is_output.get(v, False)]
    if not outputs:
        outputs = [v for v in topo_order if not children_map.get(v, [])]

    # --- State ---
    fast_mem = set()
    in_slow = set()
    ops = []
    op_count = 0
    MAX_OPS = 20000
    
    # Track remaining children for each vertex to determine if it can be evicted
    remaining_children = {v: len(children_map.get(v, [])) for v in vertices}
    
    # Track the earliest topo position of a child that still needs this vertex
    # We update this as we go.
    # For eviction strategy, we want to keep vertices that are needed "soon".
    # A vertex can be safely evicted (stored) if all its children are computed.
    # If we MUST evict (budget full, no safe candidate), we pick the one with the latest next need.
    
    def get_next_need(v):
        """Return the min topo index of children not yet computed, or inf."""
        min_pos = float('inf')
        for c in children_map.get(v, []):
            # Check if child is already computed
            # We don't have a 'computed' set directly, but we can infer from remaining_children?
            # No, remaining_children counts total children. We need to know which are done.
            # Let's maintain a set of computed vertices.
            pass
        return min_pos

    computed = set()
    
    def get_earliest_pending_child(v):
        min_pos = float('inf')
        for c in children_map.get(v, []):
            if c not in computed:
                pos = topo_order.index(c) # O(N) - might be slow. Precompute index map.
                if pos < min_pos:
                    min_pos = pos
        return min_pos

    topo_index = {v: i for i, v in enumerate(topo_order)}

    def get_earliest_pending_child_fast(v):
        min_pos = float('inf')
        for c in children_map.get(v, []):
            if c not in computed:
                pos = topo_index.get(c, float('inf'))
                if pos < min_pos:
                    min_pos = pos
        return min_pos

    def make_space():
        """Evict a vertex from fast_mem to in_slow if budget is exceeded or space needed."""
        if len(fast_mem) < fast_memory_budget:
            return True
        
        # Find a vertex to evict
        # Priority 1: Vertices with no pending children (safe to evict)
        # Priority 2: Vertices with pending children, pick the one with latest next need
        
        candidates = []
        for v in fast_mem:
            pending_children = [c for c in children_map.get(v, []) if c not in computed]
            if not pending_children:
                # Can be evicted safely (no one needs it anymore in fast mem)
                # Note: If it's an output, we might need it later for final check, 
                # but we can reload it.
                candidates.append((0, 0, v)) # Priority 0, Cost 0
            else:
                next_need = min(topo_index.get(c, float('inf')) for c in pending_children)
                # Higher next_need means we can wait longer -> better candidate for eviction
                candidates.append((1, -next_need, v))
                
        if not candidates:
            return False
            
        # Sort: Priority 0 first. For Priority 1, we want MAX next_need, so MIN -next_need.
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        v_to_evict = candidates[0][2]
        
        # Evict
        fast_mem.discard(v_to_evict)
        in_slow.add(v_to_evict)
        ops.append(['store', v_to_evict])
        op_count += 1
        return True

    # --- Main Loop ---
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
            
        parents = parents_map.get(v, [])
        
        # 1. Ensure parents are in fast memory
        for p in parents:
            if p not in fast_mem:
                # Need to load p
                if p in in_slow:
                    # It's in slow memory, load it
                    # Check space
                    if len(fast_mem) >= fast_memory_budget:
                        if not make_space():
                            # Should not happen if budget > 0 and we have space logic
                            break
                    in_slow.discard(p)
                    fast_mem.add(p)
                    ops.append(['load', p])
                    op_count += 1
                else:
                    # p is an input or was never stored? 
                    # If input, load from "external" (load op)
                    # If not computed, error in logic. But inputs start "blue" in slow? 
                    # Contract: "Inputs start blue". This usually means available in slow mem or free to load.
                    # Let's assume load is required.
                    if len(fast_mem) >= fast_memory_budget:
                        if not make_space():
                            break
                    fast_mem.add(p)
                    ops.append(['load', p])
                    op_count += 1

        # 2. Compute v
        if v not in fast_mem:
            if len(fast_mem) >= fast_memory_budget:
                if not make_space():
                    continue
            fast_mem.add(v)
        
        # Perform compute
        ops.append(['compute', v])
        op_count += 1
        computed.add(v)
        
        # Update remaining children for parents? 
        # We use 'computed' set for eviction logic, so we don't strictly need remaining_children 
        # except for optimization. The get_earliest_pending_child_fast uses 'computed'.

    # --- Finalize Outputs ---
    # All outputs must be blue (in fast_mem) at the end.
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
            
        if v not in fast_mem:
            if v in in_slow:
                # Load from slow
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space():
                        # If we can't make space, we fail. 
                        # But we must return something. 
                        pass
                if v in in_slow:
                    in_slow.discard(v)
                    fast_mem.add(v)
                    ops.append(['load', v])
                    op_count += 1
            else:
                # v is not in slow mem? 
                # This implies v was evicted and deleted? Or never computed?
                # If never computed, we need to recompute.
                # If evicted and deleted (we don't delete in this strategy, we store), 
                # it should be in in_slow.
                # If it's an input that wasn't loaded?
                # Let's try to load it as an input.
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space():
                        pass
                fast_mem.add(v)
                ops.append(['load', v])
                op_count += 1

    # Truncate if necessary
    if op_count > MAX_OPS:
        ops = ops[:MAX_OPS]
        
    return ops

# EVOLVE-BLOCK-END
