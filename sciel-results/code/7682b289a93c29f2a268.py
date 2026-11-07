import time
import random

def _check_validity(subset):
    """
    Checks if the subset of integers satisfies:
    1. All non-empty subset sums are distinct.
    2. No two distinct subset sums divide each other.
    """
    k = len(subset)
    if k == 0:
        return True
    
    # Generate all subset sums
    # Using a set to automatically handle duplicates (though we check count later)
    sums = {0}
    for x in subset:
        # Create new sums by adding x to all existing sums
        # We use a list comprehension to avoid modifying the set while iterating
        new_sums = {s + x for s in sums}
        sums.update(new_sums)
    
    # Remove the empty subset sum (0)
    sums.discard(0)
    
    # Check if all sums are distinct
    # The number of non-empty subsets is 2^k - 1
    if len(sums) != (1 << k) - 1:
        return False
    
    # Check divisibility condition
    # For any s in sums, no multiple 2s, 3s, ... should be in sums
    # Optimization: Only need to check s up to max(sums) // 2
    max_sum = max(sums)
    sum_set = sums  # For O(1) lookup
    
    for s in sums:
        # If s is greater than half the max sum, it cannot have a multiple in the set
        if s > max_sum // 2:
            continue
        # Check multiples
        # m starts at 2
        for m in range(2, max_sum // s + 1):
            if s * m in sum_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    best_set = []
    best_size = 0
    candidates = list(range(1, 257))
    
    # Strategies for ordering candidates
    # 1. Random
    # 2. Descending
    # 3. Ascending
    # 4. Sorted by number of divisors (fewest first)
    
    # Precompute divisor counts for heuristic
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    div_counts = {x: count_divisors(x) for x in candidates}
    
    while time.time() < deadline:
        # Choose a strategy
        strategy = random.choice(['random', 'desc', 'asc', 'divisors'])
        
        if strategy == 'random':
            ordered = candidates.copy()
            random.shuffle(ordered)
        elif strategy == 'desc':
            ordered = sorted(candidates, reverse=True)
        elif strategy == 'asc':
            ordered = sorted(candidates)
        elif strategy == 'divisors':
            # Fewer divisors is generally better for avoiding divisibility conflicts
            ordered = sorted(candidates, key=lambda x: div_counts[x])
        
        current_set = []
        
        for x in ordered:
            # Try adding x
            new_set = current_set + [x]
            if _check_validity(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # Optional: Local Search Improvement
            # Try to add more elements if possible
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to add any remaining element
                for y in candidates:
                    if y not in current_set:
                        trial_set = current_set + [y]
                        if _check_validity(trial_set):
                            current_set = trial_set
                            improved = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set
                            break # Restart the loop to check from beginning again? 
                            # Actually, just continue the loop
                            
                # If no direct addition, try swapping?
                # This is more complex and might not be worth the time if greedy is good enough.
                # Given the theoretical bound ~10, and greedy often finds good solutions, 
                # we might skip complex swaps to keep code simple and robust.
                
    return best_set
