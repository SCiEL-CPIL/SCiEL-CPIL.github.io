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
# ball_matrix[c, w] = 1 if w is within distance 1 of c
# Use int32 for faster dot products if needed, but here we mainly use boolean indexing and counts
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

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
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
    deadline = start_time + 110  # 110s safety margin

    best_code = get_tetracode_code()
    best_size = len(best_code)
    
    # Convert best_code to indices for internal processing
    # all_words is the source of truth for indices
    
    # We will run local search starting from the Tetracode
    # Tetracode size is 81. We want to reduce it.
    
    for restart in range(10000): # Reduced restarts, each is heavier but starts from good seed
        if time.time() > deadline:
            break
        
        # Initialize with Tetracode indices
        # Need to map tetracode words to indices
        # This is slightly expensive, so we precompute the tetracode indices once?
        # Or just compute inside. 81 words * 110s / 10000 restarts is fine.
        
        # Actually, let's precompute the Tetracode indices outside the loop
        # But get_tetracode_code returns lists of ints (words), not indices.
        
        # Let's do it inside for now, it's small.
        t_code_words = get_tetracode_code()
        code = [word_to_idx[tuple(w)] for w in t_code_words]
        
        # Compute coverage counts
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_indices[c]] += 1
        
        # Reduction Phase: Remove redundant codewords
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
        
        # Local Search Phase: 1-Swap
        # We try to swap a codeword c_i with c_j (not in code)
        # if it allows further reduction or maintains validity while enabling reduction.
        # Actually, we just accept any swap that results in a valid code,
        # then try to reduce. If reduction succeeds, size decreases.
        
        improved = True
        max_swap_iterations = 50
        swap_iter = 0
        while improved and swap_iter < max_swap_iterations:
            improved = False
            swap_iter += 1
            
            if not code:
                break
                
            # Recalculate coverage counts if they might be stale?
            # We update them incrementally in the swap logic below,
            # but we start with a consistent state.
            
            # Pick a random codeword to swap out
            idx_to_swap = np.random.randint(len(code))
            c_i = code[idx_to_swap]
            
            # Find words covered only by c_i
            ball_i = ball_indices[c_i]
            unique_words = ball_i[cov_counts[ball_i] == 1]
            
            if len(unique_words) == 0:
                # c_i is redundant, should have been removed in Reduction phase.
                # But maybe we missed it? Remove it.
                cov_counts[ball_i] -= 1
                code.pop(idx_to_swap)
                improved = True
                continue
            
            # Pick a random unique word
            target_word = unique_words[np.random.randint(len(unique_words))]
            
            # Find candidates c_j that cover target_word and are not in code
            in_code = np.zeros(NUM_WORDS, dtype=bool)
            for c in code:
                in_code[c] = True
            
            candidates = np.where(ball_matrix[:, target_word] == 1)[0]
            candidates = candidates[~in_code[candidates]]
            
            if len(candidates) == 0:
                continue
                
            # Try a few random candidates
            num_tries = min(10, len(candidates))
            tried = np.random.choice(candidates, size=num_tries, replace=False)
            
            for c_j in tried:
                # Check if c_j covers ALL unique words of c_i
                # If yes, the swap is valid (coverage of unique words is maintained)
                # And other words were covered by others, so they remain covered.
                # So the new code is valid.
                covers_all_unique = True
                for w in unique_words:
                    if ball_matrix[c_j, w] == 0:
                        covers_all_unique = False
                        break
                        
                if not covers_all_unique:
                    continue
                
                # Perform swap
                # Update cov_counts
                cov_counts[ball_i] -= 1
                ball_j = ball_indices[c_j]
                cov_counts[ball_j] += 1
                code[idx_to_swap] = c_j
                
                # Try to reduce
                # We can run a quick reduction pass
                r_changed = True
                while r_changed:
                    r_changed = False
                    r_indices = np.arange(len(code))
                    np.random.shuffle(r_indices)
                    for r_idx in r_indices:
                        rc = code[r_idx]
                        if np.all(cov_counts[ball_indices[rc]] > 1):
                            cov_counts[ball_indices[rc]] -= 1
                            code.pop(r_idx)
                            r_changed = True
                            break
                            
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = [list(all_words[c]) for c in code]
                
                improved = True
                break # Restart swap search with new state

    return best_code
