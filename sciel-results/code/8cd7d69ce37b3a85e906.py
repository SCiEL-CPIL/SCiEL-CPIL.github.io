# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero perfect square
    modulo 205.

    This implementation uses a combination of:
    1. Precomputation of forbidden differences (nonzero quadratic residues mod 205)
    2. A greedy initial solution with multiple random orderings
    3. Local search (swap-based) to improve the solution
    4. Simulated annealing for further optimization

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    
    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    forbidden_set = set(forbidden)
    
    def is_valid_pair(a, b):
        """Check if pair (a, b) is valid (no forbidden difference)."""
        d1 = (a - b) % m
        d2 = (b - a) % m
        return d1 not in forbidden_set and d2 not in forbidden_set
    
    def is_valid_set(s):
        """Check if the set s is valid (no forbidden pairs)."""
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if not is_valid_pair(s_list[i], s_list[j]):
                    return False
        return True
    
    def greedy_solution(order):
        """Greedy construction given an ordering of candidates."""
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden_set or d2 in forbidden_set:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    def local_search(selected, max_iterations=10000):
        """
        Local search using swap moves: try to replace one element with another
        to maintain validity while potentially improving (though here we just
        try to find any valid set of same or larger size).
        """
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iterations):
            improved = False
            for i in range(len(selected)):
                # Try removing selected[i] and adding a different element
                for candidate in range(m):
                    if candidate in selected_set:
                        continue
                    # Check if candidate is compatible with all other selected elements
                    ok = True
                    for j in range(len(selected)):
                        if i == j:
                            continue
                        if not is_valid_pair(candidate, selected[j]):
                            ok = False
                            break
                    if ok:
                        # This swap maintains validity; we could do it, but size doesn't change
                        # For now, we just track that a valid alternative exists
                        pass
            
            # Try to add an element (increase size)
            added = False
            for candidate in range(m):
                if candidate in selected_set:
                    continue
                ok = True
                for s in selected:
                    if not is_valid_pair(candidate, s):
                        ok = False
                        break
                if ok:
                    selected.append(candidate)
                    selected_set.add(candidate)
                    added = True
                    break
            
            if not added:
                break
        
        return selected
    
    def simulated_annealing(initial_solution, max_iterations=50000, initial_temp=10.0, cooling_rate=0.9995, min_temp=0.01):
        """
        Simulated annealing to maximize the size of the valid set.
        Moves: add an element, remove an element, or swap.
        """
        current = list(initial_solution)
        current_set = set(current)
        best = list(current)
        best_set = set(current)
        best_size = len(current)
        
        temp = initial_temp
        
        for iteration in range(max_iterations):
            if temp < min_temp:
                break
            
            # Generate a neighbor
            move_type = np.random.choice(['add', 'remove', 'swap'], p=[0.4, 0.2, 0.4])
            
            if move_type == 'add':
                # Try to add a random element
                candidate = np.random.randint(0, m)
                if candidate in current_set:
                    continue
                # Check validity
                ok = True
                for s in current:
                    if not is_valid_pair(candidate, s):
                        ok = False
                        break
                if ok:
                    new_solution = current + [candidate]
                    new_set = current_set | {candidate}
                    delta = 1  # improvement in size
                    # Always accept improvements
                    current = new_solution
                    current_set = new_set
                    if len(current) > best_size:
                        best = list(current)
                        best_set = set(current)
                        best_size = len(current)
                else:
                    # Invalid move, reject
                    pass
                    
            elif move_type == 'remove':
                if len(current) <= 1:
                    continue
                idx = np.random.randint(0, len(current))
                removed = current[idx]
                new_solution = current[:idx] + current[idx+1:]
                new_set = current_set - {removed}
                delta = -1  # decrease in size
                # Accept with probability based on temperature
                if np.random.rand() < np.exp(-delta / temp):
                    current = new_solution
                    current_set = new_set
                    
            elif move_type == 'swap':
                if len(current) == 0:
                    continue
                idx = np.random.randint(0, len(current))
                old = current[idx]
                candidate = np.random.randint(0, m)
                if candidate == old or candidate in current_set:
                    continue
                # Check if candidate is compatible with all other elements
                ok = True
                for j in range(len(current)):
                    if j == idx:
                        continue
                    if not is_valid_pair(candidate, current[j]):
                        ok = False
                        break
                if ok:
                    new_solution = current[:idx] + [candidate] + current[idx+1:]
                    new_set = (current_set - {old}) | {candidate}
                    delta = 0  # same size
                    # Accept with some probability to explore
                    if np.random.rand() < np.exp(-delta / temp):
                        current = new_solution
                        current_set = new_set
                        if len(current) > best_size:
                            best = list(current)
                            best_set = set(current)
                            best_size = len(current)
            
            temp *= cooling_rate
        
        return best
    
    # Step 1: Generate multiple greedy solutions with different orderings
    np.random.seed(42)
    
    greedy_solutions = []
    
    # Natural order
    greedy_solutions.append(greedy_solution(list(range(m))))
    
    # Random orderings
    for _ in range(20):
        order = list(range(m))
        np.random.shuffle(order)
        greedy_solutions.append(greedy_solution(order))
    
    # Select the best greedy solution
    best_greedy = max(greedy_solutions, key=len)
    
    # Step 2: Apply local search to the best greedy solution
    locally_searched = local_search(best_greedy, max_iterations=5000)
    
    # Step 3: Apply simulated annealing starting from the locally searched solution
    sa_result = simulated_annealing(locally_searched, max_iterations=30000, initial_temp=5.0, cooling_rate=0.9998, min_temp=0.01)
    
    # Step 4: Also run SA from the original best greedy
    sa_result2 = simulated_annealing(best_greedy, max_iterations=30000, initial_temp=5.0, cooling_rate=0.9998, min_temp=0.01)
    
    # Pick the best result
    final = max(sa_result, sa_result2, locally_searched, best_greedy, key=len)
    
    # Final validation
    final_set = set(final)
    for a, b in combinations(final, 2):
        assert is_valid_pair(a, b), f"Invalid pair found: {a}, {b}"
    
    return np.array(final, dtype=np.int64)


# EVOLVE-BLOCK-END
