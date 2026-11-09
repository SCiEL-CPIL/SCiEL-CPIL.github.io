"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

_DEADLINE = 55.0  # Safety margin before 60s limit
_start_time = time.time()

def _check_deadline():
    if time.time() - _start_time > _DEADLINE:
        return True
    return False

def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return parents.get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    else:
        return []

def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []

def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs

def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
        
    parents = {v: _get_parents(dag, v) for v in all_v}
    children = defaultdict(list)
    for v in all_v:
        for p in parents[v]:
            children[p].append(v)
            
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = deque([v for v in all_v if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo) != len(all_v):
        return all_v
    return topo

def _compute_depths(dag, topo):
    """Compute depth of each vertex."""
    depths = {}
    for v in topo:
        parents = _get_parents(dag, v)
        if not parents:
            depths[v] = 0
        else:
            depths[v] = max(depths[p] for p in parents) + 1
    return depths

def _pebble_recursive(dag, vertices, fast_memory_budget, ops, state):
    """
    Recursive divide and conquer pebbling.
    vertices: list of vertices in this sub-DAG (topologically sorted)
    state: dict tracking which vertices are currently in fast memory
    """
    if _check_deadline():
        return
        
    if not vertices:
        return
        
    # Base case: if small enough, just do a simple topological pass
    if len(vertices) <= max(10, fast_memory_budget):
        _pebble_simple(dag, vertices, fast_memory_budget, ops, state)
        return
        
    # Find the split point. We want to split such that the "boundary" 
    # (vertices in top half that depend on bottom half, or vice versa) is small.
    # For layered DAGs, a mid-depth split is good.
    
    depths = _compute_depths(dag, vertices)
    if not depths:
        _pebble_simple(dag, vertices, fast_memory_budget, ops, state)
        return
        
    max_depth = max(depths.values())
    mid_depth = max_depth // 2
    
    # Partition vertices into top and bottom based on depth
    # Bottom: depths <= mid_depth
    # Top: depths > mid_depth
    bottom_vertices = [v for v in vertices if depths[v] <= mid_depth]
    top_vertices = [v for v in vertices if depths[v] > mid_depth]
    
    if not bottom_vertices or not top_vertices:
        # Cannot split, do simple
        _pebble_simple(dag, vertices, fast_memory_budget, ops, state)
        return
        
    # Identify boundary: 
    # 1. Vertices in bottom that are parents of vertices in top (need to be stored after bottom)
    # 2. Vertices in top that are parents of vertices in bottom (shouldn't happen in topo order if split by depth)
    
    top_set = set(top_vertices)
    bottom_set = set(bottom_vertices)
    
    # Boundary values are the bottom vertices that are parents of top vertices
    boundary = set()
    for v in top_vertices:
        for p in _get_parents(dag, v):
            if p in bottom_set:
                boundary.add(p)
                
    # 1. Pebble bottom half
    _pebble_recursive(dag, bottom_vertices, fast_memory_budget, ops, state)
    
    # 2. Store boundary values to slow memory (so we can free them if needed, 
    #    but actually we want to keep them in fast memory if budget allows, 
    #    otherwise store and reload)
    # In standard pebbling, we store the intermediate results.
    for v in boundary:
        if v in state:
            ops.append(['store', v])
            del state[v]
            
    # 3. Pebble top half
    # Note: The top half computation might need to load inputs from bottom.
    # Our simple pebbling in top half will handle loads if they are not in state.
    _pebble_recursive(dag, top_vertices, fast_memory_budget, ops, state)
    
    # 4. Load boundary values back? 
    # Actually, the top half computation should have loaded what it needed.
    # If the top half needed boundary values, it would have loaded them during its execution.
    # However, if the top half finished and deleted them, and we need them for final outputs?
    # The final outputs are in the top half.
    
    # This simple recursive approach might miss the "reload" phase if the top half 
    # deleted the boundary values after using them. 
    # A more robust approach for the "store/load" minimization is to ensure 
    # the boundary is kept in memory if possible, or explicitly managed.
    
    # For now, this recursive structure is a strong heuristic for layered DAGs.

def _pebble_simple(dag, vertices, fast_memory_budget, ops, state):
    """
    Simple topological pebbling with LRU-like eviction or immediate deletion.
    """
    if _check_deadline():
        return
        
    # We process vertices in the given order (assumed topological)
    # We need to ensure parents are in fast memory before computing child.
    
    # To minimize transfers, we try to keep parents in memory as long as possible.
    # But if budget is tight, we might need to evict.
    # For a "simple" fallback, we will:
    # 1. Load parent if not in state.
    # 2. Compute child.
    # 3. If child is not a parent of any *remaining* vertex in this batch, delete it immediately?
    #    Or keep it if it fits?
    
    # Let's use a "keep if possible" strategy.
    # Pre-calculate which vertices are needed in the future within this batch.
    remaining = set(vertices)
    future_parents = set()
    for v in vertices:
        for p in _get_parents(dag, v):
            future_parents.add(p)
            
    for v in vertices:
        if _check_deadline():
            break
            
        parents = _get_parents(dag, v)
        
        # Ensure parents are in fast memory
        for p in parents:
            if p not in state:
                if len(state) >= fast_memory_budget:
                    # Need to evict someone. 
                    # Heuristic: evict a vertex that is not needed in the future.
                    evicted = False
                    for k in list(state.keys()):
                        if k not in future_parents:
                            ops.append(['delete', k])
                            del state[k]
                            evicted = True
                            break
                    if not evicted:
                        # Evict arbitrary (or LRU) - for simplicity, evict first
                        # This is a worst case, but should be rare if budget is adequate
                        k = next(iter(state))
                        ops.append(['delete', k])
                        del state[k]
                        
                ops.append(['load', p])
                state[p] = True
                
        # Compute v
        if v in state:
            # Already there? Shouldn't happen in simple topo if we manage correctly, 
            # but just in case
            pass
        else:
            if len(state) >= fast_memory_budget:
                # Evict
                evicted = False
                for k in list(state.keys()):
                    if k not in future_parents:
                        ops.append(['delete', k])
                        del state[k]
                        evicted = True
                        break
                if not evicted:
                    k = next(iter(state))
                    ops.append(['delete', k])
                    del state[k]
                    
            ops.append(['compute', v])
            state[v] = True
            
        # Remove v from future parents set? 
        # Actually, future_parents contains parents of *all* vertices.
        # We should remove v from the "needed" set once processed? 
        # No, v might be a parent of a *later* vertex.
        # future_parents is static for this batch. 
        # But we can optimize: if v is not a parent of any *remaining* vertex, 
        # we could delete it to free space.
        
        # Check if v is needed by any vertex later in the list
        is_needed = False
        for u in vertices:
            if u == v:
                continue
            if v in _get_parents(dag, u):
                is_needed = True
                break
                
        if not is_needed and v not in _get_outputs(dag):
            # If v is not an output and not needed by anyone else in this batch, 
            # and we are tight on memory, we could delete it.
            # But deleting might cause reloads if the global state is shared.
            # In this simple batch, if it's not needed, we can delete to save space.
            if len(state) > fast_memory_budget // 2: # Heuristic: keep half free
                ops.append(['delete', v])
                del state[v]
            # Else keep it, it might be useful or deletion cost > storage cost? 
            # Actually, storage is free in fast memory, only slots are limited.
            # So we only delete if we need space.

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True

def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ['load', 'store'])

def pebble(dag, fast_memory_budget):
    global _start_time
    _start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
        
    topo = _topological_order(dag)
    outputs = _get_outputs(dag)
    
    # Strategy 1: Try Recursive Divide and Conquer
    # This is effective for layered structures.
    ops = []
    state = {}
    
    # We need to ensure all inputs are loaded initially? 
    # No, the pebbling algorithm handles loads.
    
    # Let's try the recursive approach on the whole DAG
    _pebble_recursive(dag, topo, fast_memory_budget, ops, state)
    
    # Ensure all outputs are in fast memory and then stored?
    # The task says "all outputs must finish blue". 
    # "Blue" usually means in fast memory (or available). 
    # The interface returns a list of operations. 
    # The verifier checks the state. 
    # If an output is not in fast memory at the end, is it invalid? 
    # "Inputs start blue... all outputs must finish blue". 
    # This implies the final state must have outputs in fast memory (blue).
    
    # My recursive algorithm might have deleted outputs or not computed them?
    # _pebble_recursive processes all vertices in topo. So all are computed.
    # But did it delete them?
    # In _pebble_simple, we delete if not needed. Outputs are not "needed" by others.
    # So outputs might be deleted.
    # We must ensure they are in fast memory at the end.
    
    for o in outputs:
        if o not in state:
            # If it was computed and deleted, we need to load it back?
            # But wait, if it was deleted, it's in slow memory.
            # If it was never computed? (Shouldn't happen)
            if len(state) >= fast_memory_budget:
                # Evict non-output
                evicted = False
                for k in list(state.keys()):
                    if k not in outputs:
                        ops.append(['delete', k])
                        del state[k]
                        evicted = True
                        break
                if not evicted:
                    # This is a problem. We can't fit all outputs.
                    # But the task implies a valid solution exists.
                    pass
            ops.append(['load', o])
            state[o] = True
            
    # Final check: are all outputs in state?
    # The operations list is the result.
    
    # Verify
    if not _validate_operations(ops, dag, fast_memory_budget):
        # Fallback to a very simple, safe schedule if validation fails
        # Simple: Load all inputs, compute all in topo, store all outputs.
        # This might exceed memory if budget is small, but it's a safe fallback structure.
        ops = []
        state = {}
        inputs = _get_inputs(dag)
        for i in inputs:
            if len(state) < fast_memory_budget:
                ops.append(['load', i])
                state[i] = True
            else:
                # Can't load all inputs, this simple fallback fails if budget < num_inputs
                # But let's try to compute on the fly
                pass
                
        for v in topo:
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in state:
                    if len(state) >= fast_memory_budget:
                         # Evict
                         k = next(iter(state))
                         ops.append(['delete', k])
                         del state[k]
                    ops.append(['load', p])
                    state[p] = True
            ops.append(['compute', v])
            state[v] = True
            
        for o in outputs:
            if o not in state:
                 # Load if not there
                 if len(state) >= fast_memory_budget:
                     k = next(iter(state))
                     ops.append(['delete', k])
                     del state[k]
                 ops.append(['load', o])
                 state[o] = True
                 
        # Re-validate
        if not _validate_operations(ops, dag, fast_memory_budget):
             # If still invalid, return empty or minimal valid sequence
             # Minimal valid: Load inputs, compute, store outputs.
             # If that fails, the problem is unsolvable with this budget.
             return []
             
    return ops

# EVOLVE-BLOCK-END
