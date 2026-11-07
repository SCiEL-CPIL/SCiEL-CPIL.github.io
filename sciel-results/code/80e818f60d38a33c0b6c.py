import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_validity(subset_sums):
    """
    Check if no two distinct values in subset_sums divide one another.
    subset_sums is a list of sums.
    """
    if not subset_sums:
        return True
    
    # Remove duplicates (though we should prevent them in construction)
    unique_sums = list(set(subset_sums))
    n = len(unique_sums)
    
    # Sort to make divisibility check slightly faster (smaller first)
    unique_sums.sort()
    
    for i in range(n):
        a = unique_sums[i]
        for j in range(i + 1, n):
            b = unique_sums[j]
            if a == 0:
                continue
            if b % a == 0:
                return False
    return True

def _get_subset_sums(elements):
    """
    Generate all non-empty subset sums for a list of elements.
    """
    sums = set()
    n = len(elements)
    if n == 0:
        return []
    
    # Bitmask approach for small n
    # Or iterative
    current_sums = [0]
    for x in elements:
        new_sums = [s + x for s in current_sums]
        current_sums.extend(new_sums)
    
    # Exclude 0
    sums = set(current_sums)
    sums.discard(0)
    return list(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    # Candidate range
    # We want distinct integers in 1..256
    # Let's try to find a set of size 5, 6, etc.
    
    # Heuristic: Larger numbers might be safer because their multiples are further out.
    # However, small numbers create small sums which divide many things.
    # Let's try a randomized local search / backtracking hybrid.
    
    # We will attempt to build a set of size k.
    # Start with k=5 and try to go higher if time permits.
    
    max_k_to_try = 8
    
    # Pre-compute candidate pool. 
    # To speed up, we can filter candidates that are "compatible" with a base.
    # But let's just use random sampling from 1..256.
    
    random.seed(42)
    
    def backtrack(current_set, current_sums, candidates, k_target):
        """
        current_set: list of chosen numbers
        current_sums: set of all non-empty subset sums of current_set
        candidates: list of remaining candidates to try
        k_target: desired size
        """
        if len(current_set) == k_target:
            return current_set[:]
        
        if time.time() > deadline:
            return None
            
        # Try each candidate
        # Sort candidates? Maybe randomize to avoid bias.
        # We pass a shuffled list of candidates to the caller or shuffle here.
        
        for i in range(len(candidates)):
            x = candidates[i]
            
            # Check if x is already in current_set (distinctness)
            if x in current_set:
                continue
                
            # Generate new sums
            # New sums are {x} U {s + x for s in current_sums}
            new_sums = set([x])
            for s in current_sums:
                new_sums.add(s + x)
            
            # Check for duplicates in new_sums (should not happen if x is distinct and sums unique? 
            # If s1 + x == s2 + x => s1 == s2. Since current_sums are distinct, new_sums are distinct 
            # UNLESS x itself is in current_sums or s+x = x (s=0, not in set)).
            # Wait, if x is in current_sums, then x (new) == x (old).
            # So we must ensure x is not in current_sums.
            if x in current_sums:
                continue
                
            # Also ensure no s+x == x' (another new sum)? 
            # s1+x = s2+x => s1=s2. So new_sums are distinct among themselves.
            
            all_new_sums = current_sums.union(new_sums)
            
            # Check divisibility
            # We need to check if any a in all_new_sums divides any b in all_new_sums (a != b).
            # Optimization: We only need to check pairs involving at least one element from new_sums.
            # Because current_sums were already valid.
            
            valid = True
            sorted_all = sorted(all_new_sums)
            
            # For each new sum, check against all others
            for ns in new_sums:
                for os in sorted_all:
                    if ns == os:
                        continue
                    if os % ns == 0:
                        valid = False
                        break
                    if ns % os == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # If valid, recurse
            next_candidates = candidates[:i] + candidates[i+1:]
            result = backtrack(current_set + [x], all_new_sums, next_candidates, k_target)
            if result is not None:
                return result
                
        return None

    # Try to find sets of increasing size
    for k in range(5, max_k_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this k
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Generate random candidates
            # Bias towards larger numbers?
            # Let's try random from 50..256
            pool = list(range(50, 257))
            random.shuffle(pool)
            
            # Pick a subset of pool to try? 
            # Backtracking with full pool is expensive. 
            # Let's limit the candidate list size to speed up branching.
            # Or just use the top 20 random candidates?
            # Actually, for k=5, branching factor 200 is huge.
            # We need a better heuristic.
            
            # Greedy approach first:
            # Pick x1. Then pick x2 that is valid with x1. Etc.
            
            current_set = []
            current_sums = set()
            
            # Greedy step
            # Try to build a set of size k greedily from the shuffled pool
            # If greedy fails, backtrack on the last few?
            # Simpler: Randomized Greedy with Backtracking on a limited candidate set.
            
            # Let's select a smaller pool of candidates to try in this branch
            # e.g., 30 random candidates from 1..256
            trial_pool = random.sample(range(1, 257), 40)
            
            # Shuffle trial pool
            random.shuffle(trial_pool)
            
            # Use backtrack with this limited pool
            # But backtrack needs to be efficient.
            
            # Let's refine the backtrack to use the limited pool.
            res = backtrack([], set(), trial_pool, k)
            
            if res:
                best_set = res
                best_size = k
                found = True
                # If we found a valid set of size k, we can try k+1 in next loop
                # But we might want to try more attempts for k to find a "better" one?
                # No, size is the only metric.
                break
                
    return best_set

# EVOLVE-BLOCK-END
