import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists as numpy arrays for fast vectorized operations
    # For q=3, n=8, each word has 1 + 8*2 = 17 neighbors
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # --- Phase 1: Fast Greedy Initialization ---
    # We use a greedy set cover approach. 
    # To speed up, we maintain a count of how many times each point is covered.
    # Initially all 0. We pick the word that covers the most *uncovered* points.
    # Instead of checking uncovered points explicitly which is O(N) per step,
    # we can just pick the word with the highest potential. 
    # However, a simpler fast greedy is often sufficient for initialization.
    
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # A simple greedy: pick word, mark covered, repeat.
    # Optimization: We can pre-calculate the "potential" of each word (number of neighbors).
    # But since all words have the same number of neighbors in a symmetric space like this,
    # the first pick is arbitrary. The subsequent picks depend on coverage.
    # A better greedy: pick word that covers most currently uncovered points.
    
    uncovered_mask = np.ones(total_words, dtype=bool)
    
    # To make this faster, we can use a heap or just argmax on the count of uncovered neighbors.
    # With N=6561 and ~400-500 iterations, a full scan is 6561 * 500 * 17 ops ~ 50M ops.
    # Python loops are slow. Let's use numpy.
    
    while np.any(uncovered_mask) and time.time() < deadline - 30.0:
        # Count how many uncovered points each word covers
        # This is effectively: for each word c, sum(uncovered_mask[neighbors[c]])
        # This is a matrix multiplication or dot product if we had an incidence matrix.
        # Since neighbors are sparse and small, a loop over code candidates might be ok, 
        # but we need to check all N words.
        
        # Vectorized approach:
        # Create a boolean matrix? No, too big (6561x6561).
        # Instead, loop through all words and compute coverage.
        # This is O(N * 17) per iteration. 6561 * 17 ~ 100k ops. 
        # 500 iterations -> 50M ops. In Python, this might be borderline slow if done in a pure Python loop.
        # Let's try to vectorize the coverage count update.
        
        # Actually, let's just use the simple greedy from the inspiration which worked well.
        # It scans all words.
        
        best_idx = -1
        best_count = -1
        
        # We can optimize by noting that if a word is already in code, we don't need to consider it?
        # No, we just don't add it again.
        
        # To speed up, we can maintain a `potential` array that decreases as points get covered.
        # potential[c] = number of neighbors of c that are still uncovered.
        # Initially potential[c] = 17 for all c.
        # When we add code word c_new, for every x in neighbors[c_new]:
        #   if x was uncovered:
        #      it is now covered.
        #      for every c such that x in neighbors[c]: potential[c] -= 1
        
        # This requires knowing which words cover x. That's just neighbors[x].
        # So:
        # 1. Pick c = argmax(potential)
        # 2. For x in neighbors[c]:
        #      if uncovered_mask[x]:
        #          uncovered_mask[x] = False
        #          for y in neighbors[x]:
        #              potential[y] -= 1
        
        # This is much faster.
        
        if not hasattr(construct, '_potential'):
            construct._potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
            construct._uncovered_mask = np.ones(total_words, dtype=bool)
            
        potential = construct._potential
        uncovered_mask = construct._uncovered_mask
        
        c = np.argmax(potential)
        if potential[c] == 0:
            break
            
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0 # Prevent re-selection
        
        neigh_c = neighbors[c]
        for x in neigh_c:
            if uncovered_mask[x]:
                uncovered_mask[x] = False
                # Update potential for all words that cover x
                # neighbors[x] contains all words that cover x
                potential[neighbors[x]] -= 1
                # Ensure we don't go below 0 due to any logic errors, though logic should be sound
                # np.maximum(potential, 0) is expensive, let's trust the logic.
        
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # --- Phase 2: Local Search with Systematic Reduction ---
    
    # Helper to compute coverage counts from scratch (for verification or reset)
    def get_counts(code):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    # Helper to reduce code by removing redundant words
    def reduce_code(code, counts):
        """
        Iteratively removes any codeword whose removal leaves all its neighbors covered by others.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                # If all neighbors of c have count > 1, we can remove c
                # because they are covered by at least one other codeword.
                if np.all(current_counts[neighbors[c]] > 1):
                    current_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    # Initialize counts
    current_counts = get_counts(current_code)
    
    # Initial reduction
    current_code, current_counts = reduce_code(current_code, current_counts)
    current_size = len(current_code)
    
    if current_size < best_size:
        best_size = current_size
        best_code = current_code[:]

    random.seed(42)
    
    # Main Local Search Loop
    # Strategy:
    # 1. Try to reduce.
    # 2. If reduced, update best.
    # 3. If not reduced, try to find a swap that enables reduction.
    #    - Pick a codeword c (randomly or systematically).
    #    - Find unique points U_c (covered only by c).
    #    - Find a replacement c' not in code that covers all U_c.
    #    - Swap c -> c'.
    #    - Try to reduce again.
    
    stuck_counter = 0

    while time.time() < deadline - 5.0:
        old_size = len(current_code)
        
        # 1. Reduction Pass
        current_code, current_counts = reduce_code(current_code, current_counts)
        current_size = len(current_code)
        
        if current_size < best_size:
            best_size = current_size
            best_code = current_code[:]
            stuck_counter = 0
            continue # If we reduced, try reducing again immediately
            
        # 2. If no reduction, try a swap
        if current_size == 0:
            break
            
        # Randomize order to avoid bias
        indices = list(range(current_size))
        random.shuffle(indices)
        
        swap_found = False
        for idx in indices:
            if time.time() > deadline - 10.0:
                break
                
            c = current_code[idx]
            # Find unique points: neighbors of c where count is 1
            u_indices = neighbors[c]
            unique_mask = current_counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # This should have been removed in reduce_code
                # Force removal just in case
                current_counts[neighbors[c]] -= 1
                current_code.pop(idx)
                current_size -= 1
                swap_found = True
                break
            
            # If too many unique points, a single swap is unlikely to cover them all
            # unless the replacement word is very good.
            # In ternary code length 8, max unique points for a "critical" word might be small.
            if len(unique_points) > 4:
                continue
                
            # Find candidates c' that cover all unique_points
            # c' must be in intersection of neighbors[unique_point] for all unique_points
            # AND c' must not be in current_code
            
            if len(unique_points) == 1:
                candidates_set = set(neighbors[unique_points[0]].tolist())
            else:
                candidates_set = set(neighbors[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors[u].tolist())
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
                
            # Filter out current codewords
            candidates_set.difference_update(current_code)
            
            if not candidates_set:
                continue
                
            # Pick a candidate
            # Heuristic: Pick one that maximizes overlap with other codewords?
            # Or just random.
            c_prime = random.choice(list(candidates_set))
            
            # Apply swap
            # 1. Remove c
            current_counts[neighbors[c]] -= 1
            current_code[idx] = c_prime
            
            # 2. Add c_prime
            current_counts[neighbors[c_prime]] += 1
            
            swap_found = True
            break
            
        if not swap_found:
            stuck_counter += 1
            # If stuck for a while, we could try a 2-opt or random perturbation,
            # but for now, let's just keep trying with different random seeds/orders.
            # The reduce_code step is key.
            
    # --- Phase 3: Final Polish ---
    # One last reduction pass on the best code found
    final_code = best_code[:]
    final_counts = get_counts(final_code)
    final_code, final_counts = reduce_code(final_code, final_counts)
    
    # Verify validity
    if not np.all(final_counts > 0):
        # Fallback to best_code if polish broke it (shouldn't happen)
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
