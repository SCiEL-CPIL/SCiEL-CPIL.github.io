import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet q with covering radius radius.
    Returns a list of codewords (lists of ints).
    """
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    # 1. Generate all words in the space
    # For q=3, n=8, total words = 6561. This is small.
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # 2. Precompute neighbors for each word (distance <= radius)
    # neighbors[i] is a list of indices of words within distance 'radius' from all_words[i]
    neighbors = []
    # For efficiency, we can generate neighbors on the fly or precompute.
    # Precomputing 6561 * (1 + n*(q-1)) neighbors is feasible.
    # For q=3, n=8, radius=1: 1 + 8*2 = 17 neighbors per word.
    # Total edges ~ 6561 * 17 ~ 111k. Very small.
    
    # Helper to get neighbors of a tuple
    def get_neighbors(w):
        res = [w]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[pos] = val
                    res.append(tuple(new_w))
        return res

    # Precompute neighbor indices
    for w in all_words:
        nbrs = get_neighbors(w)
        neighbors.append([word_to_idx[nbr] for nbr in nbrs])
        
    # 3. Greedy Set Cover
    # uncovered: boolean array, True if word is not yet covered
    uncovered = np.ones(num_words, dtype=bool)
    code = []
    
    # To speed up greedy, we can track counts of how many times a candidate covers uncovered words.
    # But simple greedy: at each step, find the word that covers the most uncovered neighbors.
    # Since num_words is small, we can just iterate.
    
    while np.any(uncovered):
        if time.time() - start_time > timeout:
            break
            
        best_idx = -1
        max_covered = -1
        
        # Optimization: Only consider words that are themselves uncovered or have high potential?
        # Actually, any word can be a codeword.
        # To save time, we can randomize the order or use a heuristic.
        # Let's try a simple loop first.
        
        # For faster greedy, we can maintain a count of uncovered neighbors for each word.
        # But updating that is O(num_words) per step.
        # With 6561 words, and potentially ~400 steps, 6561*400 ~ 2.6M operations. This is fast in Python/NumPy.
        
        # Let's use a vectorized approach for the greedy step.
        # count_uncovered[i] = sum(uncovered[neighbors[i]])
        # This requires summing over a list of indices.
        
        # A faster heuristic:
        # Pick a random uncovered word, then pick a candidate that covers it and maximizes other coverage.
        # Or just scan all words.
        
        # Let's scan all words to find the one with max uncovered coverage.
        # To avoid O(N^2) in the worst case of many iterations, we can break early if we find a perfect cover (all 17 uncovered).
        
        # Vectorized coverage count:
        # This is tricky with lists of indices.
        # Let's stick to a loop but optimize it.
        
        # Actually, let's use a different greedy strategy:
        # 1. Pick a random uncovered word `u`.
        # 2. Among all words that cover `u` (i.e., `u` is in `neighbors[candidate]`), pick the one that covers the most other uncovered words.
        
        # Precompute reverse neighbors: who covers me?
        # rev_neighbors[i] = list of j such that i is in neighbors[j]
        # Since distance is symmetric, rev_neighbors[i] == neighbors[i].
        
        # So, if we pick an uncovered word `u`, the candidates are `neighbors[u]`.
        # We evaluate candidates in `neighbors[u]`.
        
        # This reduces the search space from 6561 to ~17 candidates per step!
        
        # Get a random uncovered index
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
            
        # If many uncovered, just pick one randomly to keep it simple and fast
        u_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
        
        # Candidates are neighbors of u_idx
        candidates = neighbors[u_idx]
        
        best_cand = -1
        max_cov = -1
        
        for c in candidates:
            # Count how many uncovered words are covered by c
            # neighbors[c] is the list of indices covered by c
            # We need sum(uncovered[neighbors[c]])
            # Using numpy indexing
            cov_count = np.sum(uncovered[neighbors[c]])
            if cov_count > max_cov:
                max_cov = cov_count
                best_cand = c
                if cov_count == len(neighbors[c]): # Max possible
                    break
                    
        if best_cand != -1:
            code.append(all_words[best_cand])
            # Update uncovered
            uncovered[neighbors[best_cand]] = False
            
    if time.time() - start_time > timeout:
        return code

    # 4. Local Search: Try to remove redundant codewords
    # A codeword is redundant if all its neighbors are covered by other codewords.
    
    # To check redundancy efficiently:
    # For each word in the space, count how many codewords cover it.
    # If any word has count == 1, it is only covered by that specific codeword (if that codeword is the only one).
    # Actually, if a codeword C is removed, it is safe if for all w in neighbors[C], 
    # there exists another C' in code (C' != C) such that w in neighbors[C'].
    
    # Let's maintain a coverage count array.
    # coverage[w] = number of codewords in code that cover w.
    
    coverage = np.zeros(num_words, dtype=int)
    for c in code:
        idx = word_to_idx[c]
        coverage[neighbors[idx]] += 1
        
    # Check if valid
    if np.any(coverage == 0):
        # Should not happen if greedy finished, but just in case
        pass

    improved = True
    while improved:
        if time.time() - start_time > timeout:
            break
        improved = False
        
        # Try to remove each codeword
        # We can iterate in random order to find removals faster
        indices_to_check = list(range(len(code)))
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if time.time() - start_time > timeout:
                break
                
            c = code[i]
            c_idx = word_to_idx[c]
            
            # Check if all neighbors of c are covered by others
            # A neighbor w is covered by others if coverage[w] > 1
            # (since c covers w, coverage[w] includes c. If coverage[w] > 1, some other code covers it too)
            
            # We need to check all w in neighbors[c_idx]
            # If min(coverage[neighbors[c_idx]]) > 1, then c is redundant.
            
            # Optimization: If any neighbor has coverage == 1, it MUST be c (since c covers it).
            # So if min(coverage[neighbors[c_idx]]) == 1, c is NOT redundant.
            
            min_cov = np.min(coverage[neighbors[c_idx]])
            
            if min_cov > 1:
                # c is redundant, remove it
                coverage[neighbors[c_idx]] -= 1
                code.pop(i)
                improved = True
                # After removal, we might have created new redundancies, so restart loop or continue
                # Continuing is fine, we'll check others.
                break # Restart the shuffle/check to account for index shift
                
    return code

# Fallback if the specific parameters don't match the optimized path or if greedy fails to cover
# The greedy with random pivot should always cover.

# One potential issue: The greedy step `np.sum(uncovered[neighbors[c]])` creates a temporary array.
# With 6561 words and ~400 steps, and 17 candidates per step, this is 400 * 17 * 17 operations roughly?
# No, `neighbors[c]` has length 17. `uncovered[neighbors[c]]` is a view/copy of 17 bools. Sum is fast.
# 400 steps * 17 candidates * 17 neighbors = ~115k operations. Very fast.

# The local search:
# Checking min(coverage[neighbors[c_idx]]) is 17 operations.
# If we have 400 codewords, and we check them all, that's 400 * 17 = 6800 operations.
# Even with multiple passes, it's very fast.

# The bottleneck might be the initial greedy if it takes many steps.
# But 6561 / 17 (avg coverage) ~ 386 steps.
# 386 * 17 candidates * 17 neighbors ~ 112k numpy ops. Trivial.

# So this should run well within the time limit and likely find a code close to the lower bound.
# The lower bound is 403. Greedy usually gets within 10-20% of optimal for such problems.
# 486 is the baseline. If we get < 486, we win.

# Let's add a simple random restart if the first greedy run is bad?
# No, the time limit is tight. One good run is better than multiple bad ones.
# The "random pivot" greedy is a standard heuristic (Chvatal's greedy variant).

# To be safe, let's ensure we handle the case where `code` is empty or invalid.
# The problem guarantees a solution exists.

# Final check: The function signature is `construct(q, n, radius)`.
# It returns a list of lists.

# One detail: `word_to_idx` uses tuples. `code` contains tuples from `all_words`.
# The return type should be lists? "Each codeword is a list of n integers".
# I should convert tuples to lists in the return.

def construct(q, n, radius):
    start_time = time.time()
    timeout = 110
    
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def get_neighbors(w):
        res = [w]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[pos] = val
                    res.append(tuple(new_w))
        return res

    neighbors = []
    for w in all_words:
        nbrs = get_neighbors(w)
        neighbors.append([word_to_idx[nbr] for nbr in nbrs])
        
    uncovered = np.ones(num_words, dtype=bool)
    code = []
    
    while np.any(uncovered):
        if time.time() - start_time > timeout:
            break
            
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
            
        u_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
        candidates = neighbors[u_idx]
        
        best_cand = -1
        max_cov = -1
        
        for c in candidates:
            cov_count = np.sum(uncovered[neighbors[c]])
            if cov_count > max_cov:
                max_cov = cov_count
                best_cand = c
                if cov_count == len(neighbors[c]):
                    break
                    
        if best_cand != -1:
            code.append(all_words[best_cand])
            uncovered[neighbors[best_cand]] = False
            
    if time.time() - start_time > timeout:
        return [list(c) for c in code]

    coverage = np.zeros(num_words, dtype=int)
    for c in code:
        idx = word_to_idx[c]
        coverage[neighbors[idx]] += 1
        
    improved = True
    while improved:
        if time.time() - start_time > timeout:
            break
        improved = False
        
        indices_to_check = list(range(len(code)))
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if time.time() - start_time > timeout:
                break
                
            c = code[i]
            c_idx = word_to_idx[c]
            
            min_cov = np.min(coverage[neighbors[c_idx]])
            
            if min_cov > 1:
                coverage[neighbors[c_idx]] -= 1
                code.pop(i)
                improved = True
                break
                
    return [list(c) for c in code]
