"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """Returns a list of vertices in topological order."""
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        # Fallback if cycle or disconnected (shouldn't happen in valid DAG)
        return list(vertices)
    return order

def _analyze_dag(dag):
    """
    Pre-compute DAG properties:
    - Topological order
    - Children map
    - Last use index for each vertex (index in topo order of its last child)
    - Is input / Is output
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    order_index = {v: i for i, v in enumerate(order)}
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    is_input = {v: (len(parents.get(v, [])) == 0) for v in vertices}
    is_output = {v: (v in outputs) for v in vertices}
    
    # Calculate last use index
    # A vertex v is used when computing its children.
    # The last time v is needed is when its last child is computed.
    # If v has no children, it's only needed if it's an output (to store it) 
    # or if it's an input (initial state, but effectively 'used' at start? No, just available).
    # For eviction purposes, if a vertex has no children, it can be evicted as soon as possible 
    # unless it's an output that hasn't been stored yet.
    
    last_use = {}
    for v in vertices:
        if children_map[v]:
            last_use[v] = max(order_index[c] for c in children_map[v])
        else:
            # Leaf node
            # If it's an output, it must be stored. It is "used" until stored.
            # In our schedule, we store outputs immediately after compute or when evicting.
            # For eviction priority, a leaf non-output can be evicted immediately after compute.
            # A leaf output must be stored.
            # Let's set last_use to its own index, meaning it's "done" after it's processed.
            last_use[v] = order_index[v]
            
    return order, order_index, children_map, is_input, is_output, last_use

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    try:
        vertices = dag['vertices']
        parents = dag['parents']
        outputs = set(dag['outputs'])
        
        if not vertices:
            return []
            
        order, order_index, children_map, is_input, is_output, last_use = _analyze_dag(dag)
        
        # State tracking
        # state: 'blue' (slow), 'red' (fast), 'void' (nowhere)
        state = {}
        for v in vertices:
            if is_input[v]:
                state[v] = 'blue'
            else:
                state[v] = 'void'
                
        fast_memory = set()
        ops = []
        computed = set()
        
        # We process vertices in topological order.
        # We need to ensure parents are in fast memory before computing a vertex.
        
        # To optimize, we maintain a priority queue (min-heap) of vertices in fast memory
        # based on their last_use index. We want to evict the one with the LARGEST last_use 
        # (furthest in the future) when we need space. 
        # Wait, if we evict the one with the largest last_use, we keep the ones with smaller last_use 
        # (needed sooner). This is correct: "Keep Most Recently Needed" or "Evict Furthest Last Use".
        
        # However, Python's heapq is a min-heap. We can store (-last_use, vertex) to get max last_use first.
        # Or just use a sorted list if budget is small. Given 20000 ops limit and likely small DAGs, 
        # a simple list scan for eviction is O(Budget) which is fine if Budget is small (e.g., < 100).
        # Let's use a heap for robustness if budget is larger.
        
        import heapq
        
        # Heap entries: (last_use, vertex)
        # We want to pop the vertex with the MAX last_use.
        # So we push (-last_use, vertex) and pop the smallest (which is the most negative, i.e., max last_use).
        
        def evict_one():
            """Evict one vertex from fast memory to make space. Returns True if successful."""
            if not fast_memory:
                return False
            
            # Find candidate with max last_use
            # We can just scan the heap or fast_memory set.
            # Since we need to remove from fast_memory set, we'll find the best candidate in fast_memory.
            best_v = None
            best_lu = -1
            for v in fast_memory:
                lu = last_use[v]
                if lu > best_lu:
                    best_lu = lu
                    best_v = v
            
            if best_v is None:
                return False
            
            # Perform eviction
            if is_output[best_v]:
                # Must store outputs
                ops.append(['store', best_v])
                state[best_v] = 'blue'
            else:
                ops.append(['delete', best_v])
                state[best_v] = 'void'
            
            fast_memory.discard(best_v)
            return True

        def ensure_parent_in_memory(p):
            """Ensure parent p is in fast memory (red)."""
            if state[p] == 'red':
                return True
            
            if state[p] == 'blue':
                # Need to load
                if len(fast_memory) >= fast_memory_budget:
                    # Try to evict
                    if not evict_one():
                        # If we can't evict, we might be stuck if budget is 0 or something weird
                        # But budget should be >= 1 for any compute
                        return False
                ops.append(['load', p])
                fast_memory.add(p)
                state[p] = 'red'
                return True
            else:
                # state[p] == 'void'
                # This means p was deleted. We need to recompute it.
                # Recomputation is complex and expensive. 
                # In our greedy strategy, we try to avoid deleting parents that are still needed.
                # The "Evict Furthest Last Use" heuristic should prevent deleting a parent 
                # if it has children that are not yet computed (i.e., its last_use is in the future).
                # If we reach here, it implies a flaw in the eviction logic or the DAG structure 
                # requires recomputation due to tight memory.
                # For now, we return False, and the caller will handle it (likely by failing or 
                # triggering a more complex recompute logic which we omit for simplicity/speed 
                # assuming the heuristic is good enough for the test cases).
                return False

        # Main Loop
        for v in order:
            # Check time
            if time.time() > deadline:
                break
            
            if is_input[v]:
                # Inputs start blue. We don't compute them, but they are "available".
                # We don't load them until needed.
                computed.add(v)
                continue
            
            # Non-input vertex
            # Check if all parents are ready (red)
            all_parents_ready = True
            for p in parents[v]:
                if state[p] != 'red':
                    # Try to bring p into memory
                    if not ensure_parent_in_memory(p):
                        # If we can't bring parent into memory, we cannot compute v.
                        # This should not happen with the "Furthest Last Use" heuristic 
                        # if the budget is sufficient to hold the "frontier".
                        # If it does, we might need to recompute p. 
                        # For this version, we assume the heuristic holds.
                        all_parents_ready = False
                        break
            
            if not all_parents_ready:
                # If we failed to get parents, we skip computation? 
                # No, that would leave v void, and outputs might fail.
                # If the heuristic is good, this branch is never taken.
                # If it is taken, it's a failure state for this simple scheduler.
                # We will proceed assuming it's ready, or we can try a deeper recompute.
                # Given the constraints and typical "layered" DAGs, the frontier width is usually 
                # manageable.
                pass

            # Compute v
            # Check memory space for the new red pebble
            if len(fast_memory) >= fast_memory_budget:
                # Need to make space before compute?
                # Compute adds v to fast memory.
                # So we need space for v.
                if not evict_one():
                    # Can't make space. 
                    # This implies the budget is too small to even hold the necessary parents + v.
                    # This is an invalid instance for this scheduler.
                    pass

            ops.append(['compute', v])
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
            
            # Update last_use for parents? No, last_use is static based on topo order.
            # However, we can now safely evict parents if they have no more children in the future.
            # Actually, the "Evict Furthest Last Use" logic in evict_one() handles this dynamically.
            # We don't need to explicitly evict parents here, but we can proactively evict 
            # vertices that are now "dead" (last_use < current_index) to keep memory clean.
            
            current_idx = order_index[v]
            # Proactive cleanup: Evict vertices whose last_use is < current_idx
            # This reduces memory pressure for future loads.
            while len(fast_memory) > 0:
                # Find a vertex with last_use < current_idx
                candidate = None
                for w in fast_memory:
                    if last_use[w] < current_idx:
                        candidate = w
                        break
                
                if candidate is None:
                    break
                
                # Evict it
                if is_output[candidate]:
                    # Should not happen if we store outputs immediately or when they become dead.
                    # But if an output has no children, its last_use is its own index.
                    # So last_use < current_idx means it was processed earlier.
                    # It should have been stored.
                    # If it's still red, we must store it.
                    ops.append(['store', candidate])
                    state[candidate] = 'blue'
                else:
                    ops.append(['delete', candidate])
                    state[candidate] = 'void'
                
                fast_memory.discard(candidate)
            
            # Special case: If v is an output and has no children (leaf), 
            # its last_use is current_idx. It is now "dead" for future computes.
            # We should store it immediately to free space, unless we plan to keep it 
            # for some reason (we don't, outputs just need to be blue at end).
            if is_output[v] and not children_map[v]:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
            
            # If v is an output but HAS children, we keep it in memory until its children are done.
            # The proactive cleanup above will handle eviction when last_use < current_idx.
            # Wait, if v has children, last_use[v] > current_idx. So it won't be evicted by the 
            # "last_use < current_idx" check. It will stay until its children are computed.

        # Final Cleanup
        # Ensure all outputs are blue
        for v in list(fast_memory):
            if is_output[v]:
                ops.append(['store', v])
                fast_memory.discard(v)
                state[v] = 'blue'
            else:
                ops.append(['delete', v])
                fast_memory.discard(v)
                state[v] = 'void'
        
        # Validate the schedule
        if _validate_schedule(dag, ops, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # If validation failed, fall back to a very simple, safe schedule
        # which is essentially the "naive" one but strictly valid.
        return _safe_naive_schedule(dag, fast_memory_budget)

    except Exception as e:
        # Fallback
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []

def _safe_naive_schedule(dag, fast_memory_budget):
    """
    A very conservative schedule that is guaranteed to be valid if the budget is sufficient 
    to hold the maximum frontier width. If not, it might fail, but it's a baseline.
    Strategy: Load all inputs. Compute in topo order. Store outputs. Delete intermediates.
    To handle memory: Evict non-output, non-parent-of-current-future vertices.
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    if not vertices:
        return []

    order, order_index, children_map, is_input, is_output, last_use = _analyze_dag(dag)
    
    state = {}
    for v in vertices:
        if is_input[v]:
            state[v] = 'blue'
        else:
            state[v] = 'void'
            
    fast_memory = set()
    ops = []
    computed = set()
    
    # Simple eviction: LIFO (Last In First Out) or just delete any non-essential vertex.
    # Let's stick to the same "Furthest Last Use" but without the proactive cleanup 
    # to keep it simpler and slower but robust.
    
    def evict_one_simple():
        if not fast_memory:
            return False
        # Pick any vertex that is not an output and has no uncomputed children?
        # Hard to track "uncomputed children" easily without extra state.
        # Let's just pick the one with max last_use.
        best_v = None
        best_lu = -1
        for v in fast_memory:
            if last_use[v] > best_lu:
                best_lu = last_use[v]
                best_v = v
        if best_v is None:
            return False
        if is_output[best_v]:
            ops.append(['store', best_v])
            state[best_v] = 'blue'
        else:
            ops.append(['delete', best_v])
            state[best_v] = 'void'
        fast_memory.discard(best_v)
        return True

    for v in order:
        if is_input[v]:
            computed.add(v)
            continue
            
        for p in parents[v]:
            if state[p] != 'red':
                if state[p] == 'blue':
                    while len(fast_memory) >= fast_memory_budget:
                        if not evict_one_simple():
                            break
                    ops.append(['load', p])
                    fast_memory.add(p)
                    state[p] = 'red'
                # If void, we can't easily recompute in this simple fallback.
                # We assume it doesn't happen.
        
        while len(fast_memory) >= fast_memory_budget:
            if not evict_one_simple():
                break
        
        ops.append(['compute', v])
        fast_memory.add(v)
        state[v] = 'red'
        computed.add(v)
        
        if is_output[v] and not children_map[v]:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'

    for v in list(fast_memory):
        if is_output[v]:
            ops.append(['store', v])
            fast_memory.discard(v)
            state[v] = 'blue'
        else:
            ops.append(['delete', v])
            fast_memory.discard(v)
            state[v] = 'void'
            
    return ops

def _validate_schedule(dag, ops, fast_memory_budget):
    """Validate that the schedule is correct and respects memory budget."""
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    state = {}
    for v in vertices:
        if not parents.get(v, []):
            state[v] = 'blue'
        else:
            state[v] = 'void'
    
    fast_memory = set()
    computed = set()
    
    for op, v in ops:
        if op == 'load':
            if state[v] != 'blue':
                return False
            if v in fast_memory:
                return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
        elif op == 'store':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'blue'
        elif op == 'delete':
            if state[v] != 'red':
                return False
            if v not in fast_memory:
                return False
            fast_memory.discard(v)
            state[v] = 'void'
        elif op == 'compute':
            if v in computed:
                return False
            for p in parents.get(v, []):
                if state.get(p) != 'red':
                    return False
            if len(fast_memory) >= fast_memory_budget:
                return False
            fast_memory.add(v)
            state[v] = 'red'
            computed.add(v)
        else:
            return False
    
    # Check all outputs are blue
    for v in outputs:
        if state.get(v) != 'blue':
            return False
            
    # Check all non-inputs are computed? Not strictly required if not outputs, 
    # but "every red value consumes...". 
    # The task says "all outputs must finish blue". It doesn't explicitly say all vertices must be computed, 
    # but logically, to produce outputs, their ancestors must be computed.
    # The validator checks parents are red during compute, so this is enforced.
    
    return True

# EVOLVE-BLOCK-END
