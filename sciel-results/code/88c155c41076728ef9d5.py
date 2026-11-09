"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a B2[2] set in {1..1000} maximizing size.
    
    Strategy:
    1. Greedy initialization to get a good starting set.
    2. Local search with swap moves to escape local minima.
    3. Time-bounded with deadline guard.
    """
    DEADLINE = time.time() + 145  # 5 second safety margin
    
    def greedy_init():
        """Greedy construction from 1 to 1000."""
        chosen = []
        counts = {}
        for x in range(1, 1001):
            # Compute new sums
            new_sums = [2 * x] + [x + y for y in chosen]
            # Check if adding x keeps all sums <= 2
            valid = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        return chosen
    
    def get_counts(chosen):
        """Compute sum counts for a given set."""
        counts = {}
        n = len(chosen)
        for i in range(n):
            for j in range(i, n):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def try_add(chosen, counts, x, chosen_set):
        """Try to add x to the set. Returns (success, new_counts)."""
        if x in chosen_set:
            return False, counts
        new_sums = [2 * x] + [x + y for y in chosen]
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                return False, counts
        new_counts = dict(counts)
        for s in new_sums:
            new_counts[s] = new_counts.get(s, 0) + 1
        return True, new_counts
    
    def remove_element(chosen, counts, idx):
        """Remove element at idx from chosen and update counts."""
        x = chosen[idx]
        new_counts = dict(counts)
        for i in range(len(chosen)):
            if i == idx:
                continue
            s = x + chosen[i]
            new_counts[s] -= 1
            if new_counts[s] == 0:
                del new_counts[s]
        s = 2 * x
        new_counts[s] -= 1
        if new_counts[s] == 0:
            del new_counts[s]
        new_chosen = chosen[:idx] + chosen[idx+1:]
        return new_chosen, new_counts
    
    # Phase 1: Greedy initialization
    best = greedy_init()
    best_counts = get_counts(best)
    best_size = len(best)
    best_set = set(best)
    
    # Phase 2: Local search with swap moves
    random.seed(42)
    all_numbers = list(range(1, 1001))
    
    iterations = 0
    while time.time() < DEADLINE and iterations < 1000:
        iterations += 1
        improved = False
        chosen = list(best)
        counts = dict(best_counts)
        chosen_set = set(chosen)
        
        # Try swap moves: remove one element, add another
        for idx in range(len(chosen)):
            if time.time() > DEADLINE:
                break
            
            x_remove = chosen[idx]
            new_chosen, new_counts = remove_element(chosen, counts, idx)
            new_chosen_set = set(new_chosen)
            
            # Try to add a new element
            candidates = [x for x in all_numbers if x not in new_chosen_set]
            random.shuffle(candidates)
            
            for x_add in candidates:
                if time.time() > DEADLINE:
                    break
                
                success, final_counts = try_add(new_chosen, new_counts, x_add, new_chosen_set)
                if success:
                    # This swap keeps the same size, try to find an improvement
                    new_chosen_final = new_chosen + [x_add]
                    # Try to add more elements
                    final_set = set(new_chosen_final)
                    can_add_more = False
                    for x_extra in all_numbers:
                        if x_extra in final_set:
                            continue
                        success2, counts2 = try_add(new_chosen_final, final_counts, x_extra, final_set)
                        if success2:
                            can_add_more = True
                            break
                    
                    if can_add_more:
                        # This is a good move, continue building
                        best = new_chosen_final
                        best_counts = final_counts
                        best_set = final_set
                        best_size = len(best)
                        improved = True
                        break
                elif success is None:
                    pass
        
        # Try direct addition
        if not improved:
            for x in all_numbers:
                if time.time() > DEADLINE:
                    break
                if x in best_set:
                    continue
                success, new_counts = try_add(best, best_counts, x, best_set)
                if success:
                    best = best + [x]
                    best_counts = new_counts
                    best_set = set(best)
                    best_size = len(best)
                    improved = True
                    break
        
        if not improved:
            # No improvement found, try perturbation
            if random.random() < 0.3:
                idx = random.randint(0, len(best) - 1)
                perturbed, perturbed_counts = remove_element(best, best_counts, idx)
                perturbed_set = set(perturbed)
                
                # Try to add back elements
                for x in all_numbers:
                    if time.time() > DEADLINE:
                        break
                    if x in perturbed_set:
                        continue
                    success, new_counts = try_add(perturbed, perturbed_counts, x, perturbed_set)
                    if success:
                        perturbed = perturbed + [x]
                        perturbed_counts = new_counts
                        perturbed_set = set(perturbed)
                
                if len(perturbed) > best_size:
                    best = perturbed
                    best_counts = perturbed_counts
                    best_set = perturbed_set
                    best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
