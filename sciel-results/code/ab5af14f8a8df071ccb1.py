import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute reciprocal weights
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Precompute conflict pairs for each x
    # conflicts[x] = list of (a, c) such that {a, x, c} is a 3-AP
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, N + 1):
            a = x - d
            c = x + d
            if a < 1 or c > N:
                break
            conflicts[x].append((a, c))
            
    def is_valid_addition(mask, x):
        """Check if x can be added to set represented by mask."""
        if (mask >> (x - 1)) & 1:
            return False
        
        for a, c in conflicts[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (c - 1)) & 1):
                return False
        
        return True

    def get_sum(mask):
        total = 0.0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                total += weights[x]
        return total

    def greedy_fill(mask, current_sum):
        """
        Greedily add valid elements to mask.
        Prioritizes smaller x (higher reciprocal).
        Returns updated mask and sum.
        """
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not ((mask >> (x - 1)) & 1):
                    if is_valid_addition(mask, x):
                        mask |= (1 << (x - 1))
                        current_sum += weights[x]
                        improved = True
                        break
        return mask, current_sum

    def local_search(mask, current_sum, time_limit):
        """
        Local search: try removing each element, then greedy fill.
        Accept if sum improves.
        """
        improved_global = True
        while improved_global:
            improved_global = False
            if (time.time() - start_time) > time_limit:
                break
            
            # Get list of elements in mask
            elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            
            # Shuffle to avoid bias? Or keep deterministic order?
            # Deterministic order (small to large) might get stuck if removing small x is always bad.
            # But removing small x is only bad if it doesn't unlock larger weight.
            # Let's iterate in random order to escape symmetry.
            random.shuffle(elems)
            
            for x in elems:
                if (time.time() - start_time) > time_limit:
                    return mask, current_sum
                
                # Simulate removing x
                temp_mask = mask & ~(1 << (x - 1))
                temp_sum = current_sum - weights[x]
                
                # Greedy fill
                new_mask, new_sum = greedy_fill(temp_mask, temp_sum)
                
                if new_sum > current_sum + 1e-12:
                    mask = new_mask
                    current_sum = new_sum
                    improved_global = True
                    break # Restart loop with new mask
        
        return mask, current_sum

    # Phase 1: Initial Greedy
    best_mask = 0
    best_sum = 0.0
    
    # Standard greedy: smallest first
    mask = 0
    s = 0.0
    for x in range(1, N + 1):
        if is_valid_addition(mask, x):
            mask |= (1 << (x - 1))
            s += weights[x]
    mask, s = greedy_fill(mask, s)
    
    if s > best_sum:
        best_sum = s
        best_mask = mask
        
    # Phase 2: Local Search on the greedy result
    best_mask, best_sum = local_search(best_mask, best_sum, DEADLINE - 5)
    
    # Phase 3: Iterated Local Search (Perturbation)
    # We use the remaining time for restarts/perturbations
    while (time.time() - start_time) < DEADLINE - 2:
        # Perturb current best
        current_mask = best_mask
        elems = [x for x in range(1, N + 1) if (current_mask >> (x - 1)) & 1]
        
        if not elems:
            break
        
        # Remove 1-5 random elements
        num_remove = random.randint(1, min(5, len(elems)))
        to_remove = random.sample(elems, num_remove)
        for r in to_remove:
            current_mask &= ~(1 << (r - 1))
        
        # Re-optimize
        current_sum = get_sum(current_mask)
        current_mask, current_sum = greedy_fill(current_mask, current_sum)
        
        # Local search on the perturbed state
        # Use a tighter time limit for this sub-search to allow more restarts
        current_mask, current_sum = local_search(current_mask, current_sum, DEADLINE - 2)
        
        if current_sum > best_sum + 1e-12:
            best_sum = current_sum
            best_mask = current_mask
    
    # Final polish
    best_mask, best_sum = local_search(best_mask, best_sum, DEADLINE - 1)
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    
    return result
