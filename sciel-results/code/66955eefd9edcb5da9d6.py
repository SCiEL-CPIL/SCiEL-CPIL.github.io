"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, with safety margin
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S, lookup):
        """Check if set S is sum-free using lookup set for O(1) membership."""
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True
    
    def count_violations(S, lookup):
        """Count number of sum-free violations in S."""
        S_set = set(S)
        count = 0
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    count += 1
        return count
    
    def greedy_sum_free(elements, rng):
        """Greedy construction of sum-free set."""
        S = []
        S_set = set()
        # Try elements in random order
        shuffled = elements[:]
        rng.shuffle(shuffled)
        for x in shuffled:
            # Check if adding x creates a violation
            valid = True
            for a in S:
                if a + x in S_set:
                    valid = False
                    break
                if x + a in S_set:
                    valid = False
                    break
                if x + x in S_set:
                    valid = False
                    break
            if valid:
                S.append(x)
                S_set.add(x)
        return S
    
    def local_search(elements, initial_S, rng, max_iterations=5000):
        """Local search to improve sum-free set size."""
        S = list(initial_S)
        S_set = set(S)
        removed = [x for x in elements if x not in S_set]
        best_size = len(S)
        
        # Precompute conflicts for faster checking
        def can_add(x, current_S_set):
            for a in current_S_set:
                if a + x in current_S_set:
                    return False
                if x + a in current_S_set:
                    return False
            if x + x in current_S_set:
                return False
            return True
        
        def would_create_violation_if_removed(x, current_S_set):
            """Check if removing x would allow adding some removed element."""
            # This is complex; skip for now
            return False
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            
            # Try to add back removed elements
            for i, x in enumerate(removed):
                if can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    removed.pop(i)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                    break
            
            # If no additions, try swaps: remove one element, try to add two
            if not improved and removed:
                # Try removing each element in S and see if we can add more
                for j in range(len(S)):
                    candidate_to_remove = S[j]
                    # Temporarily remove
                    S_set.remove(candidate_to_remove)
                    added_count = 0
                    new_adds = []
                    for i, x in enumerate(removed):
                        if can_add(x, S_set):
                            new_adds.append((i, x))
                        if len(new_adds) >= 2:
                            break
                    
                    if len(new_adds) > 1:
                        # Accept the swap
                        S.pop(j)
                        for idx, x in new_adds[:2]:
                            S.append(x)
                            S_set.add(x)
                            removed.pop(idx)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                    else:
                        # Restore
                        S_set.add(candidate_to_remove)
            
            if not improved:
                # No improvement, try random restarts or break
                if iteration > 1000:
                    break
        
        return S
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        rng = Random(seed + 1000)
        
        # Multiple strategies, keep best
        best_solution = []
        
        # Strategy 1: Odd numbers
        odd_set = [x for x in elements if x % 2 != 0]
        if len(odd_set) > len(best_solution):
            best_solution = odd_set
        
        # Strategy 2: Greedy with random order
        greedy_sol = greedy_sum_free(elements, rng)
        if len(greedy_sol) > len(best_solution):
            best_solution = greedy_sol
        
        # Strategy 3: Start with odds, then local search
        ls_sol = local_search(elements, odd_set, rng, max_iterations=3000)
        if len(ls_sol) > len(best_solution):
            best_solution = ls_sol
        
        # Strategy 4: Start with greedy, then local search
        ls_sol2 = local_search(elements, greedy_sol, rng, max_iterations=3000)
        if len(ls_sol2) > len(best_solution):
            best_solution = ls_sol2
        
        # Verify sum-free
        best_set = set(best_solution)
        for a in best_set:
            for b in best_set:
                if a + b in best_set:
                    # Should not happen, but fix if it does
                    best_solution = [x for x in best_solution if x != a]
                    break
        
        results.append(best_solution)
    
    return results
# EVOLVE-BLOCK-END
