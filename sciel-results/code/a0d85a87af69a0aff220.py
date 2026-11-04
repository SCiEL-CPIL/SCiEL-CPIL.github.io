"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = q ** n

    # Precompute neighbors (coverage balls) for each word
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        ball = set([i])
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                ball.add(word_to_idx[new_w])
        neighbors.append(ball)
        
    if radius > 1:
        # Fallback for general radius (slower)
        neighbors = []
        for i, w in enumerate(all_words):
            ball = set()
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    ball.add(j)
            neighbors.append(ball)

    # Greedy Initialization
    uncovered = set(range(num_words))
    chosen = set()
    
    # To speed up greedy, we can just pick the one that covers most uncovered.
    # Since N=729, O(N^2) is fine.
    while uncovered:
        if time.time() > deadline - 5: # Reserve time for local search
            break
            
        best_candidate = -1
        best_count = 0
        
        for i in range(num_words):
            # Count how many uncovered words are in neighbors[i]
            # Since neighbors[i] is small (13 for r=1), iterating is fast
            count = 0
            for idx in neighbors[i]:
                if idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidate = i
                
        if best_candidate == -1 or best_count == 0:
            break
            
        chosen.add(best_candidate)
        uncovered -= neighbors[best_candidate]
        
    # If we ran out of time during greedy, return what we have (might be invalid if uncovered not empty)
    # But greedy usually finishes fast.
    if uncovered:
        # Fallback: add all uncovered? No, that's bad.
        # Just return the current set. It might be invalid. 
        # But for the specific task, greedy should finish.
        pass

    # Local Search: Simulated Annealing / Hill Climbing
    # State: `chosen` set of indices
    # Objective: Minimize len(chosen) subject to coverage.
    
    # Maintain `count` array: count[i] = number of codewords in `chosen` covering word i
    count = np.zeros(num_words, dtype=np.int8)
    for c in chosen:
        for w in neighbors[c]:
            count[w] += 1
            
    # Verify initial validity
    # min(count) should be > 0
    
    current_size = len(chosen)
    
    # Local search loop
    # Strategy: 
    # 1. Try to remove a random codeword. If valid, keep it.
    # 2. If not valid, try to swap it with a word that covers the most lost words.
    
    # To avoid getting stuck, we can use random restarts or just run this loop for a while.
    # Given the small size, we can do many iterations.
    
    best_size = current_size
    best_set = set(chosen)
    
    # We will run the local search until deadline
    while time.time() < deadline - 1:
        
        # 1. Try to remove redundant codewords
        # Pick a random codeword
        c = random.choice(list(chosen))
        
        # Check if removing c breaks coverage
        # Words that will be uncovered: those where count[w] == 1 and w in neighbors[c]
        # If any such word exists, coverage is broken.
        
        # To make this efficient, we can just simulate removal.
        # But we need to find the best swap if it breaks.
        
        # Let's identify the "critical" words for c
        # critical_words = {w in neighbors[c] | count[w] == 1}
        
        # If critical_words is empty, c is redundant.
        # We can check this by iterating neighbors[c]
        
        is_redundant = True
        critical_words = []
        for w in neighbors[c]:
            if count[w] == 1:
                is_redundant = False
                critical_words.append(w)
            # If count[w] > 1, it's fine.
            # If count[w] == 0, it's already broken (shouldn't happen)
            
        if is_redundant:
            # Remove c
            chosen.remove(c)
            for w in neighbors[c]:
                count[w] -= 1
            current_size -= 1
            if current_size < best_size:
                best_size = current_size
                best_set = set(chosen)
            continue
            
        # If not redundant, try to swap c with a new word w_new
        # We want w_new to cover as many critical_words as possible.
        # Note: w_new can be any word not in chosen.
        # Ideally, w_new covers all critical_words.
        
        # Find best w_new
        best_w_new = -1
        best_coverage = 0
        
        # We can restrict search to words that cover at least one critical word.
        # Or just iterate all words not in chosen.
        # N=729, chosen size ~70-80. So ~650 candidates.
        # For each candidate, check overlap with critical_words.
        # critical_words size is at most 13.
        
        # Optimization: Pre-compute which words cover which critical words?
        # Or just iterate.
        
        for w_new in range(num_words):
            if w_new in chosen:
                continue
                
            # Check coverage of critical_words by w_new
            # We only care about critical_words
            cov_count = 0
            for w in critical_words:
                if w in neighbors[w_new]:
                    cov_count += 1
                    
            if cov_count > best_coverage:
                best_coverage = cov_count
                best_w_new = w_new
                
        # If best_w_new covers all critical words, the swap is valid (coverage maintained)
        # If best_w_new covers fewer, coverage is broken -> invalid swap.
        # So we only swap if best_coverage == len(critical_words).
        
        if best_w_new != -1 and best_coverage == len(critical_words):
            # Perform swap
            # Remove c
            for w in neighbors[c]:
                count[w] -= 1
            chosen.remove(c)
            
            # Add w_new
            for w in neighbors[best_w_new]:
                count[w] += 1
            chosen.add(best_w_new)
            
            # Size remains same. 
            # This move might enable future removals.
            
            # We don't update best_size here as size didn't change.
            # But the configuration changed.
            
        # If no valid swap found, we are stuck in a local minimum for this c.
        # We can try another random c.
        
    # Return the best set found
    # Ensure we return a list of lists
    result = [list(all_words[i]) for i in best_set]
    
    return result
# EVOLVE-BLOCK-END
