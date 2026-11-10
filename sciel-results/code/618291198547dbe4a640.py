# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses bitset-optimized greedy construction and local search (1-move, 1-swap, 2-swap).
    """
    m = 205
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute conflict masks for each element
    # conflict_mask[i] is a bitmask where bit j is set if (i-j)%m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            mask |= (1 << j)
        conflict_mask[i] = mask
    
    def greedy_construct(order):
        """Greedy construction using bitsets."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            # Check if candidate conflicts with any selected element
            if (selected_mask & conflict_mask[candidate]) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def local_search_1move(selected_list, selected_mask):
        """Try to add any valid element. Repeat until no more can be added."""
        selected_list = list(selected_list)
        selected_mask = selected_mask
        improved = True
        while improved:
            improved = False
            for a in range(m):
                if (selected_mask & (1 << a)) == 0: # Not selected
                    if (selected_mask & conflict_mask[a]) == 0:
                        selected_mask |= (1 << a)
                        selected_list.append(a)
                        improved = True
        return selected_list, selected_mask

    def local_search_1swap(selected_list, selected_mask):
        """
        Try to swap one element out and one in.
        Accept a swap if it allows a subsequent 1-move to increase size.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        improved = True
        steps = 0
        max_steps = 5000
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            for r in list(selected_list):
                temp_mask = selected_mask & ~(1 << r)
                for a in range(m):
                    if (temp_mask & (1 << a)) == 0:
                        if (temp_mask & conflict_mask[a]) == 0:
                            # Check if this swap enables a 1-move improvement
                            new_mask = temp_mask | (1 << a)
                            # Quick check: can we add anything else?
                            can_add_more = False
                            for b in range(m):
                                if (new_mask & (1 << b)) == 0:
                                    if (new_mask & conflict_mask[b]) == 0:
                                        can_add_more = True
                                        break
                            if can_add_more:
                                # Accept this swap as it leads to improvement
                                selected_mask = new_mask
                                idx = selected_list.index(r)
                                selected_list[idx] = a
                                improved = True
                                break
                if improved:
                    break
        return selected_list, selected_mask

    def local_search_2swap(selected_list, selected_mask):
        """
        Try to remove two elements and add two.
        Look for swaps that enable subsequent 1-move improvements.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        improved = True
        steps = 0
        max_steps = 500
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            n = len(selected_list)
            if n < 2:
                break
            # Try random pairs to keep it fast
            for _ in range(200):
                r1, r2 = random.sample(selected_list, 2)
                temp_mask = selected_mask & ~(1 << r1) & ~(1 << r2)
                
                # Find candidates that can be added
                candidates = []
                for a in range(m):
                    if (temp_mask & (1 << a)) == 0:
                        if (temp_mask & conflict_mask[a]) == 0:
                            candidates.append(a)
                
                if len(candidates) < 2:
                    continue
                
                # Try pairs from candidates
                for i in range(min(len(candidates), 50)):
                    for j in range(i+1, min(len(candidates), 50)):
                        a1, a2 = candidates[i], candidates[j]
                        if a1 == a2:
                            continue
                        # Check if a1 and a2 conflict with each other
                        if (conflict_mask[a1] & (1 << a2)) or (conflict_mask[a2] & (1 << a1)):
                            continue
                        # Check if both can be added
                        new_mask = temp_mask | (1 << a1) | (1 << a2)
                        # Check if this enables a 1-move
                        can_add_more = False
                        for b in range(m):
                            if (new_mask & (1 << b)) == 0:
                                if (new_mask & conflict_mask[b]) == 0:
                                    can_add_more = True
                                    break
                        if can_add_more:
                            # Accept
                            selected_mask = new_mask
                            # Update list
                            selected_list.remove(r1)
                            selected_list.remove(r2)
                            selected_list.append(a1)
                            selected_list.append(a2)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return selected_list, selected_mask

    def simulated_annealing(selected_list, selected_mask, start_time, deadline):
        """
        Simulated annealing to escape local optima.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        current_size = len(selected_list)
        best_size = current_size
        best_list = list(selected_list)
        best_mask = selected_mask
        
        temperature = 10.0
        cooling_rate = 0.995
        min_temperature = 0.1
        
        while temperature > min_temperature and time.time() < deadline - 5:
            # Random move: swap one element
            if len(selected_list) > 0:
                r = random.choice(selected_list)
                # Find a random replacement
                non_selected = [a for a in range(m) if (selected_mask & (1 << a)) == 0]
                if non_selected:
                    a = random.choice(non_selected)
                    # Check if valid
                    temp_mask = selected_mask & ~(1 << r)
                    if (temp_mask & conflict_mask[a]) == 0:
                        new_mask = temp_mask | (1 << a)
                        new_list = list(selected_list)
                        new_list[new_list.index(r)] = a
                        new_size = len(new_list)
                        
                        # Accept if same size or better, or with probability based on temperature
                        if new_size >= current_size:
                            selected_list = new_list
                            selected_mask = new_mask
                            current_size = new_size
                            if current_size > best_size:
                                best_size = current_size
                                best_list = list(selected_list)
                                best_mask = selected_mask
                        else:
                            # Accept worse move with probability exp(-delta/T)
                            delta = current_size - new_size
                            if random.random() < np.exp(-delta / temperature):
                                selected_list = new_list
                                selected_mask = new_mask
                                current_size = new_size
            
            temperature *= cooling_rate
        
        # Try to improve from best state
        best_list, best_mask = local_search_1move(best_list, best_mask)
        if len(best_list) > best_size:
            best_size = len(best_list)
        
        return best_list, best_mask, best_size

    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_solution = None
    best_size = 0
    best_mask = 0
    
    # 1. Greedy with multiple orders
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
    ]
    
    rng = random.Random(42)
    for _ in range(30):
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        sol, mask = greedy_construct(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol[:]
            best_mask = mask
    
    # 2. Local Search on best solution
    if best_solution is not None:
        best_solution, best_mask = local_search_1move(best_solution, best_mask)
        if len(best_solution) > best_size:
            best_size = len(best_solution)
        
        # Try 1-swap to shuffle, then 1-move again
        sol_swap, mask_swap = local_search_1swap(best_solution, best_mask)
        sol_swap, mask_swap = local_search_1move(sol_swap, mask_swap)
        if len(sol_swap) > best_size:
            best_size = len(sol_swap)
            best_solution = sol_swap
            best_mask = mask_swap
            
        # Try 2-swap
        sol_2swap, mask_2swap = local_search_2swap(best_solution, best_mask)
        sol_2swap, mask_2swap = local_search_1move(sol_2swap, mask_2swap)
        if len(sol_2swap) > best_size:
            best_size = len(sol_2swap)
            best_solution = sol_2swap
            best_mask = mask_2swap

    # 3. Random Restarts with Local Search
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        order = list(range(m))
        rng.shuffle(order)
        sol, mask = greedy_construct(order)
        sol, mask = local_search_1move(sol, mask)
        
        # Probabilistically try 1-swap
        if len(sol) < best_size + 2:
            sol, mask = local_search_1swap(sol, mask)
            sol, mask = local_search_1move(sol, mask)
            
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
            best_mask = mask
            
        # Check time
        if time.time() > deadline - 15:
            break

    # 4. Simulated Annealing on best solution
    if best_solution is not None and time.time() < deadline - 10:
        sa_sol, sa_mask, sa_size = simulated_annealing(best_solution, best_mask, start_time, deadline)
        if sa_size > best_size:
            best_size = sa_size
            best_solution = sa_sol
            best_mask = sa_mask

    # Final polish
    if best_solution is not None and time.time() < deadline - 2:
        final_sol, final_mask = local_search_1move(best_solution, best_mask)
        if len(final_sol) > best_size:
            best_size = len(final_sol)
            best_solution = final_sol

    if best_solution is None:
        best_solution = [0]
    
    return np.array(best_solution, dtype=np.int64)

# EVOLVE-BLOCK-END
