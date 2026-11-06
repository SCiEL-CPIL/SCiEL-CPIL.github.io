"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    def _count_violations(S):
        """
        Count the number of (y, k) violations in set S.
        A violation is: y in S, k >= 2, and there exists a subset T of S\{y} of size k
        such that sum(T) = k*y.
        Returns the total count of such violations.
        """
        n = len(S)
        if n < 3:
            return 0
        
        total_violations = 0
        S_sorted = sorted(S)
        
        for i, y in enumerate(S_sorted):
            T = S_sorted[:i] + S_sorted[i+1:]
            n_t = len(T)
            
            # DP: dp[k] is bitset of achievable sums with exactly k elements from T
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for x in T:
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            max_sum_t = sum(T)
            for k in range(2, n_t + 1):
                target = k * y
                if target > max_sum_t:
                    break
                if (dp[k] >> target) & 1:
                    total_violations += 1
                    
        return total_violations

    def _is_valid(S):
        """Check if set S has no violations."""
        return _count_violations(S) == 0

    def _greedy_construct(pool, max_size=48):
        """
        Greedily construct a valid set by adding elements one at a time.
        Returns the constructed set.
        """
        S = []
        pool = list(pool)
        random.shuffle(pool)
        
        for x in pool:
            if time.time() > deadline:
                break
            if len(S) >= max_size:
                break
            
            new_S = S + [x]
            if _is_valid(new_S):
                S.append(x)
        
        return S

    def _swap_local_search(S, max_iterations=500):
        """
        Conflict-driven swap local search.
        Iteratively tries to swap elements to reduce violations.
        Returns the improved set.
        """
        S = list(S)
        current_violations = _count_violations(S)
        
        if current_violations == 0:
            return S
        
        # Build candidate pool: all numbers not in S
        S_set = set(S)
        candidates = [x for x in range(1, 513) if x not in S_set]
        
        best_S = S
        best_violations = current_violations
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            if best_violations == 0:
                break
            
            # Try random swaps: remove a random element from S, add a random candidate
            improved = False
            
            for _ in range(20):  # Try up to 20 swaps per iteration
                if time.time() > deadline:
                    break
                
                if not candidates:
                    break
                
                # Pick a random element to remove and a random candidate to add
                remove_idx = random.randint(0, len(S) - 1)
                add_idx = random.randint(0, len(candidates) - 1)
                
                x_remove = S[remove_idx]
                x_add = candidates[add_idx]
                
                # Create new set
                new_S = S[:remove_idx] + S[remove_idx+1:] + [x_add]
                new_S_set = set(new_S)
                
                # Update candidates
                candidates[add_idx] = x_remove  # Put removed element back in candidates
                # Note: We need to be careful about duplicates, but since all elements are distinct, this is fine
                
                new_violations = _count_violations(new_S)
                
                if new_violations < best_violations:
                    best_S = new_S
                    best_violations = new_violations
                    S = new_S
                    improved = True
                    break
                else:
                    # Revert candidate change
                    candidates[add_idx] = x_add
        
            if not improved:
                # No improvement found, try a different strategy
                # Try removing the most "conflicting" element
                # For simplicity, just break if stuck
                break
        
        return best_S

    def _repair_by_removal(S, max_iterations=200):
        """
        Repair by removing elements that are involved in violations.
        """
        S = list(S)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            if _is_valid(S):
                break
            
            if len(S) < 3:
                break
            
            # Find a violation and remove an element
            # For simplicity, remove a random element
            # More sophisticated: remove the element involved in the most violations
            remove_idx = random.randint(0, len(S) - 1)
            S.pop(remove_idx)
        
        return S

    # Phase 1: Try greedy construction with different random seeds
    best_set = []
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        greedy_result = _greedy_construct(range(1, 513), max_size=48)
        
        if len(greedy_result) > len(best_set):
            best_set = greedy_result
            if len(best_set) == 48:
                break
        
        # Phase 2: Try swap local search from this starting point
        if len(greedy_result) > 0 and len(greedy_result) < 48:
            if time.time() > deadline:
                break
            
            # Try to improve via swap local search
            improved = _swap_local_search(greedy_result, max_iterations=100)
            
            if len(improved) > len(best_set):
                best_set = improved
                if len(best_set) == 48:
                    break
    
    # Phase 3: Try starting from larger random sets and repair
    if len(best_set) < 48:
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Start with a random set of size 50-60
            size = random.randint(50, 60)
            random_set = random.sample(range(1, 513), size)
            
            # Repair by removal
            repaired = _repair_by_removal(random_set, max_iterations=100)
            
            if len(repaired) > len(best_set):
                best_set = repaired
                if len(best_set) == 48:
                    break
            
            # Try swap local search from repaired set
            if len(repaired) > 0 and len(repaired) < 48:
                if time.time() > deadline:
                    break
                
                improved = _swap_local_search(repaired, max_iterations=100)
                
                if len(improved) > len(best_set):
                    best_set = improved
                    if len(best_set) == 48:
                        break
    
    # Phase 4: Final attempt with more aggressive swap search if still not at 48
    if len(best_set) < 48 and len(best_set) > 0:
        if time.time() < deadline - 5:
            improved = _swap_local_search(best_set, max_iterations=200)
            if len(improved) > len(best_set):
                best_set = improved
    
    return best_set

# EVOLVE-BLOCK-END
