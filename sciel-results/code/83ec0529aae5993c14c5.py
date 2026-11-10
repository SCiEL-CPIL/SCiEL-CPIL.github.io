# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Combines greedy initialization with a fast local search (1-swap and 1-move)
    and simulated annealing to escape local optima.
    """
    m = 205
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # For fast lookup, create a boolean array
    is_forbidden = np.zeros(m, dtype=bool)
    for d in forbidden:
        is_forbidden[d] = True
    
    def is_valid_set(selected):
        """Check if the set is valid (no forbidden differences)."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                d1 = (selected[i] - selected[j]) % m
                d2 = (selected[j] - selected[i]) % m
                if is_forbidden[d1] or is_forbidden[d2]:
                    return False
        return True
        
    def can_add(selected, candidate):
        """Check if candidate can be added to selected."""
        for s in selected:
            d1 = (candidate - s) % m
            d2 = (s - candidate) % m
            if is_forbidden[d1] or is_forbidden[d2]:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction following a given order."""
        selected = []
        for candidate in order:
            if can_add(selected, candidate):
                selected.append(candidate)
        return selected

    def local_search_1move(selected, max_iter=5000):
        """Local search with 1-move: try adding elements not in set."""
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iter):
            improved = False
            for a in range(m):
                if a in selected_set:
                    continue
                if can_add(selected, a):
                    selected.append(a)
                    selected_set.add(a)
                    improved = True
                    break
            if not improved:
                break
        
        return selected

    def local_search_1swap(selected, max_iter=2000):
        """Local search with 1-swap moves: remove one, add one."""
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iter):
            improved = False
            # Try all pairs (remove r, add a)
            for r in list(selected):
                # Try to remove r
                new_selected = [x for x in selected if x != r]
                new_set = set(new_selected)
                
                # Try to add some a not in new_selected
                for a in range(m):
                    if a in new_set:
                        continue
                    if can_add(new_selected, a):
                        # Check if this swap improves or maintains size
                        # Since we remove one and add one, size is constant.
                        # We accept if it's a different set (to escape loops) 
                        # or if we want to strictly improve, we'd need a different metric.
                        # Here we just accept valid swaps to explore the space.
                        # To avoid infinite loops, we might need a visited set or randomization.
                        # For now, accept the first valid swap.
                        selected = new_selected + [a]
                        selected_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return selected

    def simulated_annealing(initial, time_budget):
        """
        Simulated annealing to find larger square-difference-free sets.
        """
        start_time = time.time()
        rng = random.Random(123)
        
        # Start with the given initial solution
        current = set(initial)
        current_size = len(current)
        best_size = current_size
        best_set = set(current)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and (time.time() - start_time) < time_budget:
            # Generate a neighbor: remove one random element and add one random element
            if len(current) == 0:
                break
            
            rem = rng.choice(list(current))
            add = rng.randint(0, m - 1)
            
            if add == rem:
                T *= cooling
                continue
            
            # Compute new set
            new_set = set(current)
            new_set.discard(rem)
            
            # Check if add can be added
            can_add_flag = True
            for s in new_set:
                d1 = (add - s) % m
                d2 = (s - add) % m
                if is_forbidden[d1] or is_forbidden[d2]:
                    can_add_flag = False
                    break
            
            if not can_add_flag:
                T *= cooling
                continue
            
            new_set.add(add)
            new_size = len(new_set)
            
            # Acceptance criterion
            if new_size >= current_size:
                current = new_set
                current_size = new_size
                if new_size > best_size:
                    best_size = new_size
                    best_set = set(current)
            else:
                # New size is smaller; accept with probability exp(-delta/T)
                delta = current_size - new_size
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current = new_set
                    current_size = new_size
            
            T *= cooling
        
        return list(best_set)

    start_time = time.time()
    deadline = start_time + 590  # Safety margin: 590s out of 630s
    
    best_solution = None
    best_size = 0
    
    # Initialize with simple greedy
    initial = greedy_construct(list(range(m)))
    if len(initial) > best_size:
        best_size = len(initial)
        best_solution = initial[:]
    
    # Try local search on initial
    improved = local_search_1move(initial)
    if len(improved) > best_size:
        best_size = len(improved)
        best_solution = improved[:]
    
    improved = local_search_1swap(improved)
    if len(improved) > best_size:
        best_size = len(improved)
        best_solution = improved[:]
    
    # Random restarts
    rng = random.Random(42)
    restart_count = 0
    
    while time.time() < deadline - 60:  # Leave 60s for SA
        restart_count += 1
        
        # Random order
        order = list(range(m))
        rng.shuffle(order)
        solution = greedy_construct(order)
        
        # Local search
        solution = local_search_1move(solution)
        solution = local_search_1swap(solution)
        
        if len(solution) > best_size:
            best_size = len(solution)
            best_solution = solution[:]
        
        # Check time
        if time.time() > deadline - 60:
            break
    
    # Simulated Annealing on the best solution found so far
    if best_solution is not None and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            sa_solution = simulated_annealing(best_solution, remaining_time)
            if len(sa_solution) > best_size:
                best_size = len(sa_solution)
                best_solution = sa_solution[:]
    
    # Final local search on best solution
    if best_solution is not None and time.time() < deadline - 2:
        final = local_search_1move(best_solution)
        final = local_search_1swap(final)
        if len(final) > best_size:
            best_size = len(final)
            best_solution = final[:]
    
    if best_solution is None:
        best_solution = [0]
    
    return np.array(best_solution, dtype=np.int64)

# EVOLVE-BLOCK-END
