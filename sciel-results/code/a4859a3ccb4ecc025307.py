"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    
    def score_solution(solution_masks):
        """Calculate total comparable pairs."""
        count = 0
        for i in range(len(solution_masks)):
            a = solution_masks[i]
            for j in range(i + 1, len(solution_masks)):
                b = solution_masks[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    def delta_swap(sol, idx, new_mask):
        """
        Calculate delta score if we replace sol[idx] with new_mask.
        sol is a list of masks.
        """
        old_mask = sol[idx]
        delta = 0
        for i in range(len(sol)):
            if i == idx:
                continue
            other = sol[i]
            
            # Check comparability of old_mask with other
            if (old_mask & other) == old_mask or (old_mask & other) == other:
                delta -= 1
                
            # Check comparability of new_mask with other
            if (new_mask & other) == new_mask or (new_mask & other) == other:
                delta += 1
        return delta

    # 1. Greedy Initialization
    current_sol = []
    used = set()
    
    # Add empty set first
    current_sol.append(0)
    used.add(0)
    
    for _ in range(m - 1):
        best_candidate = None
        best_gain = -1
        
        for mask in range(total_subsets):
            if mask in used:
                continue
            
            gain = 0
            for existing in current_sol:
                if (mask & existing) == mask or (mask & existing) == existing:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_candidate = mask
        
        if best_candidate is not None:
            current_sol.append(best_candidate)
            used.add(best_candidate)
        else:
            # Fallback: pick any unused
            for mask in range(total_subsets):
                if mask not in used:
                    current_sol.append(mask)
                    used.add(mask)
                    break

    current_score = score_solution(current_sol)
    best_sol = current_sol.copy()
    best_score = current_score

    # 2. Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    max_iters = 200000 
    
    for it in range(max_iters):
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Pick random index to replace
        idx = random.randint(0, m - 1)
        old_mask = current_sol[idx]
        
        # Pick random new mask not in used
        new_mask = None
        attempts = 0
        while new_mask is None:
            candidate = random.randint(0, total_subsets - 1)
            if candidate not in used:
                new_mask = candidate
            attempts += 1
            if attempts > 100:
                break # Safety

        if new_mask is None or new_mask == old_mask:
            continue

        d_score = delta_swap(current_sol, idx, new_mask)
        
        # Acceptance probability
        if d_score > 0 or random.random() < math.exp(d_score / T):
            # Accept
            used.remove(old_mask)
            used.add(new_mask)
            current_sol[idx] = new_mask
            current_score += d_score
            
            if current_score > best_score:
                best_score = current_score
                best_sol = current_sol.copy()
        
    # Convert best_sol (masks) to lists of integers
    result = []
    for mask in best_sol:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
