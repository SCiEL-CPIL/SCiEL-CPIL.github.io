"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute powers of q for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls as numpy arrays for fast intersection
    # balls[i] is a numpy array of indices within radius of word i
    balls = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls.append(np.array(ball, dtype=np.int32))
    
    # Convert balls to sets for faster membership testing in some operations
    # But for gain calculation, we use set intersection with uncovered
    balls_set = [set(b) for b in balls]

    def prune_cover(indices):
        """Remove redundant codewords from a valid cover."""
        indices = list(indices)
        if not indices:
            return indices
        
        # Count how many times each word is covered
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in indices:
            for w in balls_set[idx]:
                cover_counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                # Check if this idx is necessary: does it cover any word with count 1?
                necessary = False
                for w in balls_set[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    # Remove it
                    for w in balls_set[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    best_code = None
    best_size = float('inf')
    
    # 1. Try Direct Product Construction (if applicable) as a strong initial guess
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        dp_pruned = prune_cover(dp_indices)
        if len(dp_pruned) < best_size:
            best_size = len(dp_pruned)
            best_code = dp_pruned

    # 2. Greedy Set Cover with Randomization
    # Strategy: Repeatedly pick the codeword that covers the most uncovered words.
    # To break ties and find better local optima, we randomize among top candidates.
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            # Calculate gains: number of uncovered words covered by each candidate
            # Optimization: We only need to consider candidates that cover at least one uncovered word.
            # For small spaces (<= 729), we can just iterate all.
            gains = []
            for c in range(total_words):
                gain = len(balls_set[c] & uncovered)
                if gain > 0:
                    gains.append((gain, c))
            
            if not gains:
                # Should not happen if the space is coverable
                break
                
            # Sort by gain descending
            gains.sort(key=lambda x: x[0], reverse=True)
            
            max_gain = gains[0][0]
            # Get all candidates with max gain
            top_candidates = [c for g, c in gains if g == max_gain]
            
            # Randomized selection:
            # With some probability, pick from the top 3 (or fewer) to explore slightly sub-optimal choices
            # This helps escape local minima in the greedy process.
            if len(gains) >= 3 and random.random() < 0.3:
                # Pick from top 3
                top_k = min(3, len(gains))
                top_k_indices = [c for g, c in gains[:top_k]]
                best_c = random.choice(top_k_indices)
            else:
                # Pick from all max-gain candidates
                best_c = random.choice(top_candidates)
            
            code_indices.append(best_c)
            uncovered -= balls_set[best_c]
            
            if time.time() > deadline - 15.0:
                break
        
        if uncovered:
            # Incomplete cover due to time or logic error, discard
            continue
            
        # Prune the result to remove redundancies
        pruned = prune_cover(code_indices)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned

    if best_code is None:
        # Fallback to direct product if it wasn't tried or failed
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            # Last resort: return all words (valid but huge)
            best_code = list(range(total_words))

    # Convert indices back to list of lists
    result = [list(idx_to_tuple(c)) for c in best_code]
    
    return result
# EVOLVE-BLOCK-END
