# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses a combination of greedy initialization, local search (1-swap and 1-move),
    and random restarts with time budgeting.
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
    
    def local_search_1swap(selected, max_iter=5000):
        """Local search with 1-swap moves: remove one, add one."""
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iter):
            improved = False
            # Try all pairs (remove r, add a)
            for r in list(selected):
                for a in range(m):
                    if a in selected_set:
                        continue
                    # Check if we can remove r and add a
                    new_selected = [x for x in selected if x != r]
                    if can_add(new_selected, a):
                        # Validate the whole set
                        if is_valid_set(new_selected + [a]):
                            selected = new_selected + [a]
                            selected_set = set(selected)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
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
    
    def random_order(rng):
        """Generate a random order of elements."""
        order = list(range(m))
        rng.shuffle(order)
        return order
    
    def weighted_greedy(order, weights):
        """Greedy with weights: prefer elements with higher weight."""
        # Sort by weight descending, break ties by order
        indexed = [(weights[i], i) for i in order]
        indexed.sort(key=lambda x: -x[0])
        sorted_order = [i for _, i in indexed]
        return greedy_construct(sorted_order)
    
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
    
    while time.time() < deadline:
        restart_count += 1
        
        # Random order
        order = random_order(rng)
        solution = greedy_construct(order)
        
        # Local search
        solution = local_search_1move(solution)
        solution = local_search_1swap(solution)
        
        if len(solution) > best_size:
            best_size = len(solution)
            best_solution = solution[:]
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final local search on best solution
    if best_solution is not None and time.time() < deadline - 10:
        final = local_search_1move(best_solution)
        final = local_search_1swap(final)
        if len(final) > best_size:
            best_size = len(final)
            best_solution = final[:]
    
    if best_solution is None:
        best_solution = [0]
    
    return np.array(best_solution, dtype=np.int64)

# EVOLVE-BLOCK-END
