import numpy as np
import itertools
import time

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
ball_indices = []

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

# Precompute for each word, which codewords cover it (inverse ball)
# cover_matrix[w, c] = 1 if w is in ball[c]
cover_matrix = ball_matrix.T  # (NUM_WORDS, NUM_WORDS) where cover_matrix[w, c] = ball_matrix[c, w]

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def _reduce_code(code, ball_indices):
    """Remove redundant codewords from code. Returns reduced code."""
    if not code:
        return code
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_indices[c]] += 1
    changed = True
    while changed:
        changed = False
        for idx in range(len(code)):
            c = code[idx]
            if np.all(cov_counts[ball_indices[c]] > 1):
                cov_counts[ball_indices[c]] -= 1
                code.pop(idx)
                changed = True
                break
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy for general case
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
    
    for restart in range(100000):
        if time.time() > deadline:
            break
        
        # Temperature annealing: start broad, narrow over restarts
        temp = max(0.5, 1.0 - restart * 0.001)
        
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        
        # Greedy Phase with randomized selection
        while not np.all(covered):
            if time.time() > deadline:
                break
            scores = ball_matrix.dot(~covered)
            max_score = np.max(scores)
            if max_score == 0:
                break
                
            threshold = max_score * temp
            candidates = np.where(scores >= threshold)[0]
            c = candidates[np.random.randint(len(candidates))]
            
            code.append(c)
            covered[ball_indices[c]] = True
        
        if time.time() > deadline:
            break
        
        # Reduction Phase
        code = _reduce_code(code, ball_indices)
        
        # Local Search Phase: try replacing codewords to enable further reduction
        improved = True
        max_ls_iterations = 30
        ls_iter = 0
        while improved and ls_iter < max_ls_iterations:
            improved = False
            ls_iter += 1
            
            if time.time() > deadline:
                break
            
            if not code:
                break
            
            # Compute coverage counts
            cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
            for c in code:
                cov_counts[ball_indices[c]] += 1
            
            # Find codewords with unique coverage
            in_code_set = set(code)
            
            # Try each codeword as a removal candidate
            for idx in range(len(code)):
                c_i = code[idx]
                ball_i = ball_indices[c_i]
                unique_words = ball_i[cov_counts[ball_i] == 1]
                
                if len(unique_words) == 0:
                    # This codeword is redundant - should have been removed
                    continue
                
                # Try to find a replacement that covers all unique words
                # and allows net reduction
                # Find all codewords that cover all unique words
                # A codeword c_j covers all unique words if ball_matrix[c_j, w] == 1 for all w in unique_words
                
                # Efficient check: for each candidate, check if it covers all unique words
                # Use the cover_matrix: cover_matrix[w, c] = 1 if w is in ball[c]
                # So we need cover_matrix[unique_words, c_j].all(axis=0)
                
                # Get candidates that cover at least one unique word
                # Actually, let's be smarter: only consider codewords not already in code
                # that cover the first unique word, then check if they cover all
                
                target = unique_words[0]
                # Candidates that cover target and are not in code
                cand_mask = cover_matrix[target] == 1
                cand_mask[np.array(list(in_code_set))] = False
                candidates = np.where(cand_mask)[0]
                
                if len(candidates) == 0:
                    continue
                
                # Try up to 10 random candidates
                num_tries = min(10, len(candidates))
                tried = np.random.choice(candidates, size=num_tries, replace=False)
                
                for c_j in tried:
                    # Check if c_j covers ALL unique words of c_i
                    if not np.all(cover_matrix[unique_words, c_j]):
                        continue
                    
                    # Perform swap: remove c_i, add c_j, then reduce
                    test_code = code.copy()
                    test_code[idx] = c_j
                    
                    # Reduce
                    test_code = _reduce_code(test_code, ball_indices)
                    
                    if len(test_code) < len(code):
                        code = test_code
                        improved = True
                        break
                
                if improved:
                    break
        
        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                pass

    return best_code
