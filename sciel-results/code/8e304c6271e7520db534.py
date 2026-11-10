import random

def construct(n):
    if n <= 3:
        # Handle small cases explicitly or with the general algorithm
        pass

    # Generate all permutations of 1..n
    import itertools
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    num_perms = len(perms)
    
    # Precompute next_main and next_jump
    # next_main[u]: append missing symbol
    # next_jump[u]: append first symbol
    next_main = [0] * num_perms
    next_jump = [0] * num_perms
    
    for i, u in enumerate(perms):
        # Find missing symbol
        s = set(u)
        missing = [x for x in range(1, n + 1) if x not in s][0]
        # Next main node
        v_main = (u[1:] + (missing,))
        next_main[i] = perm_to_idx[v_main]
        
        # Next jump node
        # Append u[0]
        v_jump = (u[1:] + (u[0],))
        next_jump[i] = perm_to_idx[v_jump]

    # Find Main Cycles
    visited = [False] * num_perms
    cycles = []
    for i in range(num_perms):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = next_main[cur]
            # Verify it's a cycle (should be, as it's a functional graph on perms)
            # The last node's next_main should be the first node of the cycle
            if next_main[cycle[-1]] == cycle[0]:
                cycles.append(cycle)
    
    # Map each node to its cycle index
    node_to_cycle = [0] * num_perms
    for c_idx, cycle in enumerate(cycles):
        for node in cycle:
            node_to_cycle[node] = c_idx
            
    num_cycles = len(cycles)
    
    # Build adjacency for jumps between cycles
    # jump_adj[c_idx] = list of (exit_node_idx, next_c_idx, entry_node_idx)
    jump_adj = [[] for _ in range(num_cycles)]
    
    for c_idx, cycle in enumerate(cycles):
        for node in cycle:
            jump_target = next_jump[node]
            next_c_idx = node_to_cycle[jump_target]
            if next_c_idx != c_idx:
                jump_adj[c_idx].append((node, next_c_idx, jump_target))
            else:
                # Jump stays in same cycle?
                # If so, we can use it to change entry point?
                # But we want to visit other cycles.
                pass

    # Greedy TSP on cycles
    # We need to visit all cycles.
    # Start with a random cycle.
    visited_cycles = [False] * num_cycles
    path = [] # List of (cycle_idx, exit_node)
    
    # Pick a start
    start_c = random.randint(0, num_cycles - 1)
    visited_cycles[start_c] = True
    current_c = start_c
    
    # We need to determine the order.
    # We will store the sequence of (cycle_idx, entry_node)
    seq = [(start_c, cycles[start_c][0])]
    
    for _ in range(num_cycles - 1):
        best_next_c = -1
        best_exit_node = -1
        best_entry_node = -1
        
        # Find any unvisited cycle reachable from current_c
        # Prefer cycles with fewer options? Or just first found.
        # To improve, we could look ahead, but greedy is a start.
        
        candidates = []
        for exit_node, next_c_idx, entry_node in jump_adj[current_c]:
            if not visited_cycles[next_c_idx]:
                candidates.append((exit_node, next_c_idx, entry_node))
                
        if not candidates:
            # No direct jump to an unvisited cycle?
            # This implies we might be stuck or the graph is not connected via jumps?
            # For n=7, it should be connected.
            # If stuck, break or handle error.
            break
            
        # Heuristic: Pick the candidate that leads to a cycle with the fewest outgoing jumps to unvisited?
        # Or just pick the first one.
        # Let's pick the first one for simplicity, or randomize.
        # To make it deterministic-ish, pick min next_c_idx?
        # Let's pick the one that minimizes the number of options for the next step?
        # For now, just pick the first.
        
        exit_node, next_c_idx, entry_node = candidates[0]
        
        # Update path
        # The current cycle was entered at some node. We traversed it fully.
        # We exit at exit_node.
        # So we must ensure we can reach exit_node from the entry node of current_c.
        # Since it's a cycle, we can always reach any node.
        # But we want to minimize the traversal?
        # No, we must visit ALL nodes in the cycle.
        # So we must traverse the entire cycle.
        # The entry point determines where we start.
        # We start at entry_node.
        # We traverse the cycle until we return to entry_node?
        # No, we traverse until we reach exit_node?
        # If we traverse from entry_node to exit_node, we might miss some nodes?
        # No, a cycle is a single loop.
        # If we start at A and go to B, we visit all nodes if we go the long way?
        # Or the short way?
        # We must visit ALL nodes.
        # So we must traverse the entire cycle.
        # This means we must end up back at the entry node?
        # If we end at exit_node, and exit_node != entry_node, we have visited all nodes?
        # Yes, if we traverse the cycle in the correct direction.
        # But we have a choice of direction? No, the graph is directed.
        # So we must follow the edges.
        # If we start at entry_node, we are forced to follow the cycle.
        # We will visit all nodes and return to entry_node.
        # So we CANNOT exit at an arbitrary exit_node unless exit_node == entry_node.
        # Wait.
        # If we start at entry_node, we visit entry_node, next, ..., prev, entry_node.
        # We are back at entry_node.
        # So we can ONLY exit from entry_node?
        # No, we can choose to exit from ANY node if we don't require returning to start?
        # But we MUST visit all nodes.
        # If we stop at exit_node, we have visited all nodes IF exit_node is the last one before returning to start?
        # No.
        # If we start at A, and the cycle is A->B->C->A.
        # If we exit at B, we have visited A, B. We missed C.
        # So we MUST complete the cycle to visit all nodes.
        # This implies we must return to the entry node.
        # So we can ONLY jump from the entry node?
        # This is a constraint.
        # If we can only jump from the entry node, then the entry node of the next cycle is determined by the jump from the entry node of the current cycle.
        # So the path is determined by the initial entry node?
        # Let's check.
        # Start at C1, entry E1.
        # Traverse C1 fully. Return to E1.
        # Jump from E1 to E2 (entry of C2).
        # Traverse C2 fully. Return to E2.
        # Jump from E2 to E3.
        # ...
        # So we just need to find a path in the "Jump" graph on the set of cycles, where the nodes are the specific entry points?
        # No, the entry point of C_{i+1} is determined by the jump from the exit point of C_i.
        # And the exit point of C_i MUST be the entry point of C_i (because we must complete the cycle).
        # So the state is just the current cycle and its entry node.
        # But the entry node of the next cycle is fixed by the current entry node.
        # So the path is unique once we pick the first cycle and entry node?
        # No, because the jump from E_i goes to a specific node E_{i+1} in C_{i+1}.
        # So the sequence of cycles is deterministic given the start?
        # If so, we just need to pick a start that allows visiting all cycles.
        # If the jump graph on cycles is a single cycle, then any start works.
        # If it's not, we might get stuck.
        
        # Let's re-evaluate: Can we exit from a node OTHER than the entry node?
        # If we enter at E, we follow the cycle.
        # We visit all nodes.
        # We end up back at E.
        # So yes, we are at E.
        # So we must jump from E.
        
        # Therefore, the problem reduces to:
        # Find a path in the graph of cycles where edge C_i -> C_j exists if jump(E_i) is in C_j.
        # And E_{i+1} = jump(E_i).
        # We need to visit all cycles.
        
        # Let's build this graph.
        # For each cycle C_i, and for each possible entry node E in C_i,
        # determine where the jump goes.
        # Actually, we don't get to choose E. E is determined by the previous jump.
        # So we just need to find a start (C_start, E_start) such that the deterministic path visits all cycles.
        
        # Let's check if such a start exists.
        # We can simulate the path for each possible start.
        # There are num_cycles * n possible starts.
        # num_cycles = 720, n=7. 5040 starts.
        # Simulation cost: num_cycles steps.
        # Total: 5040 * 720 ~ 3.6M operations. Feasible.
        
        best_path = None
        best_len = float('inf')
        
        for c_start in range(num_cycles):
            for e_start_idx in range(n):
                e_start = cycles[c_start][e_start_idx]
                
                # Simulate
                visited_sim = [False] * num_cycles
                visited_sim[c_start] = True
                curr_c = c_start
                curr_e = e_start
                steps = 0
                path_sim = [(curr_c, curr_e)]
                
                while True:
                    # Jump from curr_e
                    target_node = next_jump[curr_e]
                    target_c = node_to_cycle[target_node]
                    
                    if visited_sim[target_c]:
                        # We've visited this cycle.
                        # If we've visited ALL, we are done.
                        break
                    else:
                        visited_sim[target_c] = True
                        curr_c = target_c
                        curr_e = target_node
                        path_sim.append((curr_c, curr_e))
                        steps += 1
                        if all(visited_sim):
                            break
                    # If we loop without visiting all, break
                    if steps > num_cycles:
                        break
        
                if all(visited_sim):
                    # Found a valid path
                    # Length calculation:
                    # For each cycle, we traverse n edges (to visit all nodes and return to entry).
                    # Plus 1 edge for the jump.
                    # Total edges = num_cycles * n + (num_cycles - 1)
                    # Plus the initial n symbols?
                    # Word length = n + (num_cycles - 1) * (n + 1)?
                    # Let's construct and measure.
                    
                    # Construct word
                    word = []
                    # Start with first cycle's entry node
                    # The entry node is a tuple of length n-1?
                    # No, the node is a permutation of length n?
                    # My nodes are permutations of length n.
                    # Wait, the graph nodes are (n-1)-permutations?
                    # In my code above, `perms` are length n.
                    # `next_main` maps length n perm to length n perm?
                    # `v_main = (u[1:] + (missing,))`. This is length n.
                    # So my nodes are length n permutations.
                    # But the state of the word is the last n-1 symbols.
                    # If the node is a length n permutation, it represents the last n symbols.
                    # So the "node" in the de Bruijn sense is the last n-1 symbols.
                    # But I am using length n permutations as nodes.
                    # This is equivalent.
                    # If I am at node U (length n), the next step appends x.
                    # The new node is (U[1:] + (x,)).
                    # This matches my `next_main` and `next_jump`.
                    
                    # So, to construct the word:
                    # Start with the first node U0 (length n).
                    # Append the symbols generated by the traversal.
                    # When traversing a cycle, we append the symbols that lead to the next node.
                    # For node U, the symbol appended to get to next_main[U] is `missing(U)`.
                    # The symbol appended to get to next_jump[U] is `U[0]`.
                    
                    # Path:
                    # 1. Start with U_start (length n).
                    # 2. For each cycle in the path:
                    #    a. We are at entry_node E.
                    #    b. We traverse the cycle.
                    #    c. The sequence of nodes is E, next_main[E], next_main[next_main[E]], ...
                    #    d. We append the symbols corresponding to these transitions.
                    #    e. After n transitions, we are back at E.
                    #    f. We append the jump symbol (E[0]) to go to the next entry.
                    
                    # Let's verify the length.
                    # Start: n symbols.
                    # For each of the k cycles:
                    #   n symbols appended (to complete the cycle).
                    #   1 symbol appended (to jump to next cycle).
                    # Total: n + k * (n + 1)?
                    # No, the jump to the LAST cycle is not needed?
                    # We don't need to jump after the last cycle.
                    # So: n + (k-1) * (n + 1) + n?
                    # Let's trace.
                    # Cycle 1: Append n symbols. End at E1.
                    # Jump: Append 1 symbol. End at E2 (entry of Cycle 2).
                    # Cycle 2: Append n symbols. End at E2.
                    # Jump: Append 1 symbol. End at E3.
                    # ...
                    # Cycle k: Append n symbols. End at Ek.
                    # Stop.
                    
                    # Total length = n (initial) + (k-1) * (n + 1) + n?
                    # No.
                    # Initial word is U_start (length n).
                    # Then we append symbols.
                    # For Cycle 1:
                    #   We are at E1.
                    #   We append sym1, sym2, ..., sym_n.
                    #   After sym_n, we are back at E1.
                    #   So we have added n symbols.
                    #   Current length: 2n.
                    #   Then we append jump_sym.
                    #   Current length: 2n + 1.
                    #   We are at E2.
                    # For Cycle 2:
                    #   Append n symbols.
                    #   Current length: 2n + 1 + n.
                    #   Append jump_sym.
                    #   Current length: 2n + 1 + n + 1.
                    # ...
                    # For Cycle k:
                    #   Append n symbols.
                    #   Current length: ... + n.
                    #   No jump.
                    
                    # Total = n + (k-1)*(n+1) + n = 2n + (k-1)(n+1).
                    # For n=4, k=6.
                    # 8 + 5*5 = 33.
                    # This matches the known shortest superpermutation for n=4!
                    
                    # So the formula is: 2n + (num_cycles - 1) * (n + 1).
                    # num_cycles = (n-1)!.
                    # Length = 2n + ((n-1)! - 1) * (n + 1).
                    
                    # For n=7:
                    # 14 + (720 - 1) * 8 = 14 + 719 * 8 = 14 + 5752 = 5766.
                    # This is < 5905.
                    
                    # So if we find a valid path, we get a great score.
                    
                    # Construct the actual string to be safe.
                    word = list(cycles[c_start][e_start_idx])
                    
                    curr_c = c_start
                    curr_e = cycles[c_start][e_start_idx]
                    
                    for step_idx in range(len(path_sim) - 1):
                        # Traverse current cycle
                        curr_node = curr_e
                        for _ in range(n):
                            # Get next node in main cycle
                            next_node = next_main[curr_node]
                            # The symbol appended is the last element of next_node
                            # Or calculate missing
                            s = set(curr_node)
                            missing = [x for x in range(1, n + 1) if x not in s][0]
                            word.append(missing)
                            curr_node = next_node
                        # Now curr_node should be curr_e
                        # Perform jump
                        jump_sym = curr_node[0]
                        word.append(jump_sym)
                        
                        # Move to next cycle
                        curr_c = path_sim[step_idx + 1][0]
                        curr_e = path_sim[step_idx + 1][1]
                        
                    # Traverse last cycle
                    curr_node = curr_e
                    for _ in range(n):
                        next_node = next_main[curr_node]
                        s = set(curr_node)
                        missing = [x for x in range(1, n + 1) if x not in s][0]
                        word.append(missing)
                        curr_node = next_node
                        
                    # Check if valid
                    # And return
                    return "".join(map(str, word))
                    
        # If no valid path found, fallback to original or random
        return "".join(map(str, range(1, n + 1)))

