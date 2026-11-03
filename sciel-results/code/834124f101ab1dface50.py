import numpy as np
import itertools
import time
import random

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: ball_matrix[c, w] = 1 if w is within distance 1 of c
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
ball_indices = []  # list of arrays, each containing indices in the ball

for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

# Convert to list of arrays for fast indexing
ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]

# Precompute distance-2 graph adjacency for independent set approach
# Two codewords can coexist only if their balls don't conflict... actually for covering,
# we want a dominating set. Let's use a different approach: 
# For the "independent set" inspiration, we can think of it as:
# if two codewords are at distance 2, their balls overlap in exactly 1 point (the midpoint).
# This doesn't directly help for covering. Let me stick with the entropy-based approach.

# For the maximum entropy / cross-entropy approach:
# We maintain a coverage count vector. The "entropy" of the coverage distribution
# measures how evenly the 729 points are covered. We want to find codewords whose
# removal increases entropy (makes coverage more uniform) while maintaining full coverage.

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def _entropy_removal_pass(code_indices, max_iter=50):
    """
    Try to remove codewords using an entropy-based criterion.
    For each codeword, compute how much its removal would increase the 
    entropy of the coverage distribution (among still-covered points).
    Remove the codeword that maximally increases entropy while keeping full coverage.
    """
    code = list(code_indices)
    if not code:
        return code
    
    # Compute coverage counts
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_indices[c]] += 1
    
    removed_count = 0
    for _ in range(max_iter):
        if removed_count > 20:  # safety limit
            break
        best_c = None
        best_entropy_gain = 0.0
        
        for i, c in enumerate(code):
            # If we remove c, coverage drops by 1 for all points in ball(c)
            # Check if any point would become uncovered
            ball_pts = ball_indices[c]
            if np.any(cov_counts[ball_pts] <= 1):
                continue
            
            # Compute entropy gain
            # Current coverage vector (only consider points with count > 1 after removal)
            new_counts = cov_counts.copy()
            new_counts[ball_pts] -= 1
            
            # Compute entropy of new_counts distribution
            # Use normalized counts as probabilities
            total = new_counts.sum()
            if total == 0:
                continue
            probs = new_counts / total
            # Avoid log(0)
            probs = np.where(probs > 0, probs, 1e-10)
            new_entropy = -np.sum(probs * np.log2(probs))
            
            # Current entropy
            cur_total = cov_counts.sum()
            cur_probs = cov_counts / cur_total
            cur_probs = np.where(cur_probs > 0, cur_probs, 1e-10)
            cur_entropy = -np.sum(cur_probs * np.log2(cur_probs))
            
            entropy_gain = new_entropy - cur_entropy
            if entropy_gain > best_entropy_gain:
                best_entropy_gain = entropy_gain
                best_c = i
        
        if best_c is not None:
            c = code.pop(best_c)
            cov_counts[ball_indices[c]] -= 1
            removed_count += 1
        else:
            break
    
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (150s limit)
    
    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Vectorized greedy set cover with randomization
    for restart in range(100000):
        if time.time() > deadline:
            break
            
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase
        while not np.all(covered):
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            threshold = max_score * 0.85
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[ball_indices[c]] = True
            
        # Reduction Phase - basic redundancy removal
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
            
        changed = True
        while changed:
            changed = False
            indices = np.arange(len(code))
            np.random.shuffle(indices)
            
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
        
        # Entropy-based refinement: try to remove codewords that increase coverage entropy
        code = _entropy_removal_pass(code, max_iter=30)
        
        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                pass

    # Final local search on the best solution found
    if best_size > 71:
        # Convert to indices
        best_indices = []
        for cw in best_code:
            best_indices.append(word_to_idx[tuple(cw)])
        
        # Entropy-based local search with swaps
        # Try: remove c_i, add c_j, then check if we can remove another
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            # Do targeted swap-based local search
            for _ in range(int(remaining_time * 50)):  # ~50 iterations per second
                if time.time() > deadline - 5:
                    break
                
                code = list(best_indices)
                cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
                for c in code:
                    cov_counts[ball_indices[c]] += 1
                
                # Try to find a swap that enables further removal
                improved = False
                for i in range(len(code)):
                    if time.time() > deadline - 5:
                        break
                    c_i = code[i]
                    ball_i = ball_indices[c_i]
                    
                    # Check if removing c_i leaves few uncovered
                    uncovered_after = np.where(cov_counts[ball_i] == 1)[0]
                    if len(uncovered_after) > 5:
                        continue
                    
                    # Try to find a replacement c_j that covers all uncovered_after
                    # and whose ball overlaps significantly with c_i's ball
                    for j in range(NUM_WORDS):
                        if j in code:
                            continue
                        ball_j = ball_indices[j]
                        
                        # Check if ball_j covers all uncovered_after
                        if not np.all(cov_counts[ball_j[uncovered_after]] >= 0):  # always true
                            continue
                        
                        # Actually check: after removing c_i and adding c_j, 
                        # are all points still covered?
                        # Points in ball_i but not ball_j need other coverage
                        points_to_check = ball_i
                        for p in points_to_check:
                            if cov_counts[p] == 1 and p not in ball_j:
                                break
                        else:
                            # This swap maintains coverage
                            new_code = [c for idx, c in enumerate(code) if idx != i]
                            new_code.append(j)
                            
                            # Now try to remove another codeword
                            new_cov = np.zeros(NUM_WORDS, dtype=np.int32)
                            for c in new_code:
                                new_cov[ball_indices[c]] += 1
                            
                            for k in range(len(new_code)):
                                c_k = new_code[k]
                                if np.all(new_cov[ball_indices[c_k]] > 1):
                                    # Can remove c_k
                                    final_code = [c for idx, c in enumerate(new_code) if idx != k]
                                    if len(final_code) < best_size:
                                        best_size = len(final_code)
                                        best_code = [list(all_words[c]) for c in final_code]
                                        improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                
                if not improved:
                    break

    return best_code
