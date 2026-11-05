"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    def is_smooth(n, primes):
        """Check if n is composed only of primes in the given set."""
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
    
    def get_smooth_numbers_up_to(limit, primes):
        """Get all smooth numbers up to limit."""
        smooth = []
        for n in range(2, limit + 1):
            if is_smooth(n, primes):
                smooth.append(n)
        return smooth
    
    def count_distinct_primes_in_sums(nums, primes):
        """Count distinct primes in all pairwise sums."""
        distinct = set()
        for a, b in combinations(nums, 2):
            s = a + b
            for p in primes:
                while s % p == 0:
                    distinct.add(p)
                    s //= p
            if s > 1:
                distinct.add(s)
        return len(distinct), distinct
    
    def backtrack_select(smooth_candidates, target_size, primes, deadline, current=None, start_idx=0):
        """Backtracking to select target_size numbers such that all pairwise sums are smooth."""
        if current is None:
            current = []
        
        if len(current) == target_size:
            return current
        
        remaining_needed = target_size - len(current)
        if len(smooth_candidates) - start_idx < remaining_needed:
            return None
        
        # Try adding each candidate from start_idx onwards
        for i in range(start_idx, len(smooth_candidates)):
            if time.time() > deadline:
                return None
            
            candidate = smooth_candidates[i]
            
            # Check if candidate + all existing elements are smooth
            valid = True
            for existing in current:
                s = candidate + existing
                if not is_smooth(s, primes):
                    valid = False
                    break
            
            if valid:
                result = backtrack_select(
                    smooth_candidates, target_size, primes, deadline,
                    current + [candidate], i + 1
                )
                if result is not None:
                    return result
        
        return None
    
    def greedy_local_search(initial_set, primes, deadline):
        """Improve a set by swapping elements to reduce distinct primes."""
        current = list(initial_set)
        current_score, _ = count_distinct_primes_in_sums(current, primes)
        best = current[:]
        best_score = current_score
        
        # Try random swaps to improve
        all_numbers = list(range(1, 10001))
        
        while time.time() < deadline:
            improved = False
            for _ in range(100):  # Try 100 random swaps per iteration
                if time.time() > deadline:
                    break
                
                idx = random.randint(0, len(current) - 1)
                old_val = current[idx]
                new_val = random.choice(all_numbers)
                
                if new_val == old_val or new_val in current:
                    continue
                
                # Test the swap
                test_set = current[:]
                test_set[idx] = new_val
                
                # Check all sums involving new_val
                valid = True
                for other in test_set:
                    if other == new_val:
                        continue
                    s = new_val + other
                    if not is_smooth(s, primes):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate new score
                new_score, _ = count_distinct_primes_in_sums(test_set, primes)
                
                if new_score < best_score:
                    best_score = new_score
                    best = test_set[:]
                    current = test_set[:]
                    improved = True
            
            if not improved:
                # Try a different random restart
                current = best[:]
        
        return best, best_score
    
    best_result = None
    best_score = float('inf')
    
    # Try different prime sets
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    for primes in prime_sets:
        if time.time() > deadline - 10:
            break
        
        # Get smooth numbers up to 10000
        smooth_nums = get_smooth_numbers_up_to(10000, primes)
        
        if len(smooth_nums) < 12:
            continue
        
        # Strategy 1: Backtracking to find a set where ALL pairwise sums are smooth
        result = backtrack_select(smooth_nums, 12, primes, deadline - 5)
        
        if result is not None:
            score, distinct = count_distinct_primes_in_sums(result, primes)
            if score < best_score:
                best_score = score
                best_result = result
                if score == len(primes):
                    return best_result
        
        # Strategy 2: Start with random smooth numbers and improve via local search
        if len(smooth_nums) >= 12 and time.time() < deadline - 5:
            # Random initial set
            random.seed(42)
            initial = random.sample(smooth_nums, 12)
            improved_set, improved_score = greedy_local_search(initial, primes, deadline - 2)
            
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved_set
    
    if best_result is not None:
        return best_result
    
    # Fallback: Return a simple valid set
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# EVOLVE-BLOCK-END
