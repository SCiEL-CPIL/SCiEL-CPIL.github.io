import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    # Fallback for other parameters (simple greedy)
    total_words = q ** n
    if total_words > 100000:
        # Too large for simple fallback, return empty or minimal
        return []
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    ball_size = 1 + n * (q - 1)
    
    # Precompute balls
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls[i] = neighbors
        
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
        newly_covered = (~is_covered)[balls]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        is_covered[balls[best_idx]] = True
        
    return [list(all_words[c]) for c in codewords]

def _solve_ternary_8_1():
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # 1. Precompute balls efficiently using base-3 arithmetic
    # We don't need all_words array explicitly if we use indices, 
    # but we need to return vectors. 
    # Let's keep all_words for final conversion.
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute covers: list of codeword indices that cover each word
    # covers[w] is a list of c such that w is in balls[c]
    # Note: covers[w] is exactly balls[w] due to symmetry of Hamming distance
    # But let's verify or just use balls. 
    # Since distance is symmetric, w in balls[c] <=> c in balls[w].
    # So covers[w] == balls[w]. This is a huge simplification.
    
    def run_greedy_and_refine(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Greedy Initialization
        is_covered = np.zeros(total, dtype=bool)
        codewords = []
        
        # Optimization: Maintain a count of how many times each word is covered? 
        # No, just track uncovered.
        # To speed up greedy, we can use the fact that we only care about uncovered words.
        
        while not np.all(is_covered):
            if time.time() > deadline - 10:
                break
            
            # Find word covering most uncovered
            # newly_covered = (~is_covered)[balls]
            # scores = newly_covered.sum(axis=1)
            # This is O(N * BallSize). N=6561, BallSize=17. ~110k ops per step.
            # Greedy takes ~400 steps. 400 * 110k = 44M ops. Fast enough in numpy.
            
            scores = (~is_covered)[balls].sum(axis=1)
            max_score = np.max(scores)
            
            if max_score == 0:
                break
            
            # Pick randomly among best to add diversity
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            codewords.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        if len(codewords) == 0:
            return [], 0
            
        # Local Search
        # cover_counts[w] = number of codewords covering w
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
            
        improved = True
        iterations = 0
        max_iterations = 1000
        
        while improved and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant codewords
            i = 0
            while i < len(codewords):
                c = codewords[i]
                cover_counts[balls[c]] -= 1
                if not np.any(cover_counts == 0):
                    codewords.pop(i)
                    improved = True
                else:
                    cover_counts[balls[c]] += 1
                    i += 1
                    
            if not improved:
                # Phase 2: Swaps to enable removal
                # Try swapping each codeword with a non-codeword to see if it allows a removal
                # We iterate through codewords. If a swap succeeds, we restart the loop.
                swap_found = False
                for i in range(len(codewords)):
                    if swap_found:
                        break
                    c = codewords[i]
                    
                    # Find words uniquely covered by c
                    # unique_words = words w where cover_counts[w] == 1 and c covers w
                    unique_mask = (cover_counts == 1)
                    unique_in_ball = unique_mask[balls[c]]
                    unique_words = balls[c][unique_in_ball]
                    
                    if len(unique_words) == 0:
                        continue # Should have been removed, but maybe counts changed?
                    
                    # Find candidates b (not in codewords) that cover all unique_words
                    # candidates = intersection of covers[w] for w in unique_words
                    # covers[w] = balls[w]
                    
                    # Start with covers of first unique word
                    candidates = set(balls[unique_words[0]].tolist())
                    
                    # Intersect with covers of other unique words
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates &= set(balls[w].tolist())
                        
                    if not candidates:
                        continue
                        
                    # Exclude current codewords
                    current_set = set(codewords)
                    candidates -= current_set
                    
                    if not candidates:
                        continue
                        
                    # Try each candidate
                    # Randomize order of candidates?
                    candidate_list = list(candidates)
                    random.shuffle(candidate_list)
                    
                    for b in candidate_list:
                        # Tentative swap: remove c, add b
                        cover_counts[balls[c]] -= 1
                        cover_counts[balls[b]] += 1
                        
                        if not np.any(cover_counts == 0):
                            # Valid swap. Check if it enables a removal.
                            removal_found = False
                            for j in range(len(codewords)):
                                if j == i:
                                    continue
                                c_j = codewords[j]
                                cover_counts[balls[c_j]] -= 1
                                if not np.any(cover_counts == 0):
                                    # We can remove c_j
                                    removal_found = True
                                    cover_counts[balls[c_j]] += 1 # Revert check
                                    break
                                else:
                                    cover_counts[balls[c_j]] += 1
                                    
                            if removal_found:
                                # Accept swap
                                codewords[i] = b
                                improved = True
                                swap_found = True
                                break # Restart main while loop
                            else:
                                # Revert swap
                                cover_counts[balls[b]] -= 1
                                cover_counts[balls[c]] += 1
                        else:
                            # Invalid swap
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
                            
        return codewords, len(codewords)

    best_codewords = None
    best_size = total
    
    # Run multiple restarts with different seeds
    # How many restarts can we afford?
    # Greedy ~0.5s, Refine ~1-5s depending on size.
    # Let's try 5-10 restarts.
    
    seed = 42
    while time.time() < deadline - 10:
        codewords, size = run_greedy_and_refine(seed)
        if size < best_size:
            best_size = size
            best_codewords = codewords[:]
            # If we hit a very good size, we might want to polish it further?
            # The local search already does that.
            
        seed += 1
        
        # If we are getting close to the known best (486), maybe stop?
        # Best known is 486. If we hit 486, we can stop.
        if best_size <= 486:
            break
            
    if best_codewords is None:
        # Fallback: just return greedy result from last run or empty
        # This should not happen
        return []
        
    return [list(all_words[c]) for c in best_codewords]
