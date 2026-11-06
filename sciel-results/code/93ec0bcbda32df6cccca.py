import itertools
import time

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    # A is a list of 12 distinct integers
    # We want the max k such that there is a Sidon subset of size k
    # Max possible k is 12
    # Min possible k is 1
    # Check from high to low
    for k in range(len(A), 0, -1):
        for subset in itertools.combinations(A, k):
            if _is_sidon(subset):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    # Initial candidate: 0..11
    current_set = list(range(12))
    current_score = _max_sidon_size(current_set)
    
    # Local search
    # We will try to swap elements to reduce the max Sidon subset size
    # We iterate through all possible swaps or single replacements
    
    # To speed up, we can precompute or use heuristics, but 12 elements is small.
    # Number of candidates for replacement: 128 - 12 = 116.
    # Number of positions: 12.
    # Total swaps to check per iteration: 12 * 116 = 1392.
    # Each check involves _max_sidon_size which is ~2000 subset checks.
    # 1392 * 2000 = 2.8 million checks per full sweep.
    # Each check is fast (set ops).
    # This might be too slow for 50s if we do many sweeps.
    
    # Optimization: Only check subsets of size K and K+1?
    # If current max is K, we only care if we can reduce it to K-1.
    # So we only need to check if there is a Sidon subset of size K.
    # If yes, score is K. If no, score is K-1 (or lower).
    
    def _has_sidon_subset_of_size(A, k):
        for subset in itertools.combinations(A, k):
            if _is_sidon(subset):
                return True
        return False

    def _exact_max_sidon_size(A):
        # Find exact max size
        for k in range(len(A), 0, -1):
            if _has_sidon_subset_of_size(A, k):
                return k
        return 1

    # Refine current_score
    current_score = _exact_max_sidon_size(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    # Simple hill climbing
    while time.time() < deadline:
        improved = False
        # Try replacing each element with any other number in [0, 128]
        for i in range(12):
            if time.time() > deadline:
                break
            old_val = current_set[i]
            for new_val in range(129):
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                
                # Create new set
                new_set = current_set[:]
                new_set[i] = new_val
                
                # Check if this reduces the max Sidon size
                # We only need to check if max size < best_score
                # If best_score is 5, we check if max size <= 4
                # i.e., check if there is NO Sidon subset of size 5
                # If there is no Sidon subset of size 5, the max is at most 4.
                
                # To be efficient, first check if there is a Sidon subset of size best_score.
                # If yes, score is at least best_score. No improvement (or same).
                # If no, score is < best_score. Improvement!
                
                if best_score > 1:
                    if _has_sidon_subset_of_size(new_set, best_score):
                        # Still has a subset of size best_score, so no improvement
                        # (It could be higher, but we are minimizing, so we don't care if it's higher than best)
                        continue
                    else:
                        # No subset of size best_score.
                        # The new max is < best_score.
                        # We accept this improvement.
                        current_set = new_set
                        best_score -= 1
                        # Update best_set
                        best_set = current_set[:]
                        improved = True
                        # Restart the inner loop or continue?
                        # Continue to see if we can improve further immediately
                        break 
                else:
                    # best_score is 1, can't improve
                    break
            if improved:
                # Restart the outer loop to re-evaluate with new set
                break
        
        if not improved:
            # No single swap improves the score
            # Try 2-swap? Or stop?
            # Given time, stop.
            break
            
    return best_set
