import itertools
import random
import time

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    # Precompute pair sums for indices
    sums = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sums[i][j] = S[i] + S[j]
            
    def is_sidon(indices):
        seen = set()
        for i_idx in range(len(indices)):
            i = indices[i_idx]
            for j_idx in range(i_idx, len(indices)):
                j = indices[j_idx]
                s = sums[i][j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    # Check sizes from 12 down to 1
    for k in range(n, 0, -1):
        for comb in itertools.combinations(range(n), k):
            if is_sidon(comb):
                return k
    return 1

def count_distinct_sums(S):
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return len(sums)

def construct():
    # Initialize with a dense set
    best_S = list(range(12))
    best_proxy = count_distinct_sums(best_S)
    
    # Local search to minimize distinct pair sums
    # This forces more collisions, reducing max Sidon subset size
    start_time = time.time()
    max_time = 50  # Leave some time for final check
    
    while time.time() - start_time < max_time:
        improved = False
        # Shuffle order of indices to avoid bias
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            # Try a few random replacements
            for _ in range(10):
                new_val = random.randint(0, 128)
                if new_val in best_S:
                    continue
                
                old_val = best_S[i]
                best_S[i] = new_val
                new_proxy = count_distinct_sums(best_S)
                
                if new_proxy < best_proxy:
                    best_proxy = new_proxy
                    improved = True
                else:
                    best_S[i] = old_val
        
        if not improved:
            break
            
    # Verify the actual max Sidon size for the best candidate found
    # It's possible a set with slightly more distinct sums has a smaller max Sidon subset
    # due to structural reasons, but distinct sums is a very strong proxy.
    # We can do a few random restarts or just trust the proxy.
    # Let's do a quick check of the final set.
    final_size = max_sidon_size(best_S)
    
    # If the proxy search didn't yield a very small size, we might want to try a few other strategies
    # But given the time constraint, returning the best proxy result is a solid strategy.
    
    return best_S
