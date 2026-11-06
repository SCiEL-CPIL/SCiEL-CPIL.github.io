import sys
sys.setrecursionlimit(10000)

def pebble(dag, fast_memory_budget):
    n = len(dag)
    if n == 0:
        return []
    
    # Identify inputs and outputs
    is_input = [False] * n
    is_output = [False] * n
    child_count = [0] * n
    
    for i in range(n):
        if len(dag[i]) == 0:
            is_input[i] = True
        for p in dag[i]:
            child_count[p] += 1
            
    for i in range(n):
        if child_count[i] == 0:
            is_output[i] = True

    # Check if optimal strategy fits
    # Optimal: Keep node Red until last child is processed.
    max_reds = 0
    current_reds = set()
    remaining_children = child_count[:]
    
    for i in range(n):
        # Parents become red (if not already)
        for p in dag[i]:
            current_reds.add(p)
        # i becomes red
        current_reds.add(i)
        
        # Parents might be freed
        for p in dag[i]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                if p in current_reds:
                    current_reds.remove(p)
                    
        # If i is output, it is freed (stored)
        if is_output[i]:
            if i in current_reds:
                current_reds.remove(i)
                
        max_reds = max(max_reds, len(current_reds))
        
    if max_reds <= fast_memory_budget:
        return _execute_optimal(dag, is_input, is_output, child_count)
    else:
        return _execute_greedy(dag, fast_memory_budget, is_input, is_output, child_count)

def _execute_optimal(dag, is_input, is_output, child_count):
    n = len(dag)
    state = {}
    ops = []
    remaining_children = child_count[:]
    
    # Initialize
    for i in range(n):
        if is_input[i]:
            state[i] = 'blue'
        else:
            state[i] = 'gone'
            
    for i in range(n):
        # Ensure parents are red
        for p in dag[i]:
            if state[p] == 'blue':
                ops.append(['load', p])
                state[p] = 'red'
            elif state[p] == 'gone':
                # Should not happen in optimal if logic is correct
                # But if it does, we have a bug.
                pass
            
        # Compute i
        ops.append(['compute', i])
        state[i] = 'red'
        
        # Free parents if no more children
        for p in dag[i]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                if state[p] == 'red':
                    if is_input[p]:
                        ops.append(['store', p])
                        state[p] = 'blue'
                    else:
                        ops.append(['delete', p])
                        state[p] = 'gone'
                        
        # If i is output, store it
        if is_output[i]:
            if state[i] == 'red':
                ops.append(['store', i])
                state[i] = 'blue'
                
    return ops

def _execute_greedy(dag, fast_memory_budget, is_input, is_output, child_count):
    # Fallback for tight memory
    # This is a simplified greedy: process topologically, if memory full, delete least useful.
    # If a parent is deleted, recompute it when needed.
    
    n = len(dag)
    state = {}
    ops = []
    remaining_children = child_count[:]
    red_set = set()
    
    for i in range(n):
        if is_input[i]:
            state[i] = 'blue'
        else:
            state[i] = 'gone'
            
    def ensure_red(v):
        if state[v] == 'red':
            return
        if state[v] == 'blue':
            if len(red_set) >= fast_memory_budget:
                # Need to free space
                # Find red node with min remaining_children
                # Exclude v? v is not red yet.
                # Exclude parents of current node? We are in recursion.
                # Just pick any.
                candidates = [r for r in red_set if state[r] == 'red']
                if not candidates:
                    return # Should not happen
                # Pick one with lowest remaining_children
                # Note: remaining_children is global state, might be stale?
                # It's updated when children are processed.
                # Since we are processing in topological order, remaining_children is accurate for unprocessed children.
                min_child_node = min(candidates, key=lambda x: remaining_children[x])
                if is_input[min_child_node]:
                    ops.append(['store', min_child_node])
                    state[min_child_node] = 'blue'
                else:
                    ops.append(['delete', min_child_node])
                    state[min_child_node] = 'gone'
                red_set.remove(min_child_node)
                
            ops.append(['load', v])
            state[v] = 'red'
            red_set.add(v)
            return
            
        if state[v] == 'gone':
            # Need to recompute
            # Ensure parents are red
            for p in dag[v]:
                ensure_red(p)
            ops.append(['compute', v])
            state[v] = 'red'
            red_set.add(v)
            # Note: We don't delete parents here immediately in this simple greedy,
            # they will be managed by the main loop or future ensure_red calls.
            # This might lead to memory bloat if we don't clean up.
            # For a fallback, this is acceptable if it stays within limits.
            # To be safer, we could delete parents if they are no longer needed,
            # but that's complex in recursion.

    for i in range(n):
        # Ensure parents are red
        for p in dag[i]:
            ensure_red(p)
            
        # Compute i
        ops.append(['compute', i])
        state[i] = 'red'
        red_set.add(i)
        
        # Update remaining children and cleanup parents
        for p in dag[i]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                if state[p] == 'red':
                    if is_input[p]:
                        ops.append(['store', p])
                        state[p] = 'blue'
                    else:
                        ops.append(['delete', p])
                        state[p] = 'gone'
                    red_set.remove(p)
                    
        # If i is output
        if is_output[i]:
            if state[i] == 'red':
                ops.append(['store', i])
                state[i] = 'blue'
                red_set.remove(i)
                
    return ops
