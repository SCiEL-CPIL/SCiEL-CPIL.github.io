import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, though the task says 
    # "the evaluator passes this task's alphabet size q, length n and covering radius".
    # The task is specifically K_3(6,1).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a generic construction if parameters differ
        # Just return a trivial code (all 0s plus variations) or similar
        # For safety, return a code that is definitely valid but large
        # e.g. all words with first n-1 coords 0 and last coord 0..q-1? No.
        # Let's just use the direct sum logic if n>=4
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This only works for q=3.
            if q == 3:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # If not ternary or small n, just return a large valid code
        # Covering radius 1 means we need to cover all words.
        # A simple valid code is the whole space.
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    # Precompute neighbors for all 729 words
    # Word representation: integer 0..728
    # Convert int to base-3 tuple
    def int_to_tuple(x, n):
        return [x // (3**i) % 3 for i in range(n)]
    
    def tuple_to_int(t, n):
        return sum(t[i] * (3**i) for i in range(n))
        
    def hamming_dist(t1, t2):
        return sum(1 for a, b in zip(t1, t2) if a != b)
        
    # Precompute neighbors as numpy arrays for fast operations
    all_words = range(3**n)
    neighbors = [None] * len(all_words)
    for i in all_words:
        t = int_to_tuple(i, n)
        nb = [i]
        for pos in range(n):
            orig = t[pos]
            for new_val in range(3):
                if new_val != orig:
                    t[pos] = new_val
                    nb.append(tuple_to_int(t, n))
            t[pos] = orig
        neighbors[i] = np.array(nb, dtype=np.int32)
            
    # Initial code: Direct sum of tetracode
    tetracode_ints = []
    for a in range(3):
        for b in range(3):
            w = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            tetracode_ints.append(tuple_to_int(w, 4))
            
    initial_code = []
    for h in tetracode_ints:
        for tail in itertools.product(range(3), repeat=n - 4):
            full_t = int_to_tuple(h, 4) + list(tail)
            initial_code.append(tuple_to_int(full_t, n))
            
    code = initial_code[:]
    N = 3**n
    cover_count = np.zeros(N, dtype=np.int32)
    word_in_code = np.zeros(N, dtype=bool)
    for c in code:
        word_in_code[c] = True
        cover_count[neighbors[c]] += 1

    def update_cover_count(c, delta):
        cover_count[neighbors[c]] += delta

    best_code = code[:]
    best_size = len(code)
    
    # Local search with faster inner loops
    while time.time() < deadline:
        # 1. Try to remove redundant codewords
        # Find the redundant codeword with the largest slack (safest to remove)
        best_red = None
        best_slack = -1
        for c in code:
            min_cc = cover_count[neighbors[c]].min()
            if min_cc >= 2:
                if min_cc > best_slack:
                    best_slack = min_cc
                    best_red = c
        if best_red is not None:
            code.remove(best_red)
            word_in_code[best_red] = False
            update_cover_count(best_red, -1)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            continue  # Restart to check for new redundancies

        # 2. If no redundant, try a swap
        if not code:
            break

        c = random.choice(code)
        # Remove c
        code.remove(c)
        word_in_code[c] = False
        update_cover_count(c, -1)

        # Check validity
        uncovered_mask = (cover_count == 0)
        uncovered_count = uncovered_mask.sum()

        if uncovered_count == 0:
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            continue

        # We have uncovered words. Find candidates in the intersection of neighbors of all uncovered words.
        uncovered_list = np.where(uncovered_mask)[0]
        
        # Start with the neighbor set of the first uncovered word
        candidate_mask = np.zeros(N, dtype=bool)
        candidate_mask[neighbors[uncovered_list[0]]] = True
        
        for w in uncovered_list[1:]:
            candidate_mask[~neighbors[w]] = False
            if not candidate_mask.any():
                break

        if not candidate_mask.any():
            # Cannot swap one for one; revert
            code.append(c)
            word_in_code[c] = True
            update_cover_count(c, 1)
            continue

        # Pick the candidate that maximizes coverage improvement
        # For each candidate, score = sum of cover_count over all words it covers
        # (higher means it adds more useful coverage)
        best_cand = None
        best_score = -1
        cands = np.where(candidate_mask)[0]
        
        for cand in cands:
            # Score: how much additional coverage this candidate provides
            # Count how many words go from 0 to 1 (newly covered)
            newly_covered = ((cover_count[neighbors[cand]] == 0).sum())
            # Also consider total coverage sum for tie-breaking
            total_cov = cover_count[neighbors[cand]].sum()
            score = newly_covered * 1000 + total_cov
            # Slight preference for candidates not already in code
            if not word_in_code[cand]:
                score += 500
            if score > best_score:
                best_score = score
                best_cand = cand

        c_new = int(best_cand)
        if c_new == c:
            code.append(c)
            word_in_code[c] = True
            update_cover_count(c, 1)
            continue

        code.append(c_new)
        word_in_code[c_new] = True
        update_cover_count(c_new, 1)

    # Final polish: try a few more removals if time remains
    while time.time() < deadline - 2.0:
        best_red = None
        best_slack = -1
        for c in code:
            min_cc = cover_count[neighbors[c]].min()
            if min_cc >= 2:
                if min_cc > best_slack:
                    best_slack = min_cc
                    best_red = c
        if best_red is None:
            break
        code.remove(best_red)
        word_in_code[best_red] = False
        update_cover_count(best_red, -1)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # Convert best_code ints to lists of ints
    return [int_to_tuple(c, n) for c in best_code]
