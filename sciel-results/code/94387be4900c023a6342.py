import time
import numpy as np
import itertools
import random
from scipy.optimize import linprog

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*(q-1) = 15
    
    t_start = time.time()
    deadline = 115.0 # Safety margin (limit is 120s)
    
    # Precompute all words as indices 0..2186
    # index = d1*q^(n-1) + ... + dn
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Precompute balls: balls[i] is a list of indices of words within distance radius of word i
    balls = np.empty((N, 1 + n*(q-1)), dtype=np.int32)
    
    # Helper to get index from tuple
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx

    # Precompute balls
    for i, w in enumerate(all_words):
        ball = [i]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball
        
    # Convert balls to a sparse matrix A for LP
    # A[i, j] = 1 if word j is in ball i
    # We want to minimize sum(x_i) subject to A^T x >= 1, 0 <= x <= 1
    # This is a set cover LP relaxation.
    
    # Using scipy.sparse for efficiency
    import scipy.sparse as sp
    
    # A is N x N matrix where A[i,j] = 1 if j in ball_i
    # We need A^T for the constraints: for each word j, sum_{i: j in ball_i} x_i >= 1
    # Let's construct the constraint matrix directly:
    # Rows: words (N)
    # Cols: codewords (N)
    # M[j, i] = 1 if word j is covered by codeword i (i.e., j in ball_i)
    
    # Since balls are small (15), we can build the sparse matrix efficiently
    row_indices = []
    col_indices = []
    data = []
    
    for i in range(N):
        for j in balls[i]:
            row_indices.append(j)
            col_indices.append(i)
            data.append(1)
            
    M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    
    # Objective: minimize sum(x_i)
    c = np.ones(N)
    
    # Constraints: M @ x >= 1
    # linprog requires A_ub @ x <= b_ub
    # So: -M @ x <= -1
    A_ub = -M.toarray() # This might be memory heavy (2187x2187 ~ 4.8M floats ~ 38MB). Acceptable.
    b_ub = -np.ones(N)
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * N
    
    best_code = None
    best_len = N
    
    # Strategy 1: Solve LP Relaxation
    # Note: LP might take time. If it fails or is slow, fallback to greedy.
    # For 2187 vars and 2187 constraints, LP should be fast with HiGHS (default in recent SciPy).
    
    try:
        # Use highs method which is fast for this size
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            x = res.x
            # Round the LP solution
            # Greedy rounding: pick indices with highest x values first, 
            # but only if they cover new words.
            
            # Sort by x descending
            sorted_indices = np.argsort(x)[::-1]
            
            covered = np.zeros(N, dtype=bool)
            code = []
            
            # Pass 1: Take all x > 0.5 (or high threshold)
            # Pass 2: Greedy fill remaining uncovered
            for idx in sorted_indices:
                if time.time() - t_start > deadline - 5: break
                if x[idx] > 0.5:
                    # Check if it adds coverage
                    ball = balls[idx]
                    if not np.all(covered[ball]):
                        code.append(idx)
                        covered[ball] = True
            
            # Pass 2: Greedy fill
            while not covered.all():
                if time.time() - t_start > deadline - 5: break
                # Calculate gain
                gains = np.sum(~covered[balls], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0: break
                candidates = np.where(gains == max_gain)[0]
                # Prefer candidates with high LP value
                cand_scores = x[candidates]
                # Random tie break among top LP scores? Or just pick one with max LP?
                # Let's pick the one with max LP value, break ties randomly
                max_cand_score = np.max(cand_scores)
                top_cands = candidates[cand_scores == max_cand_score]
                idx = top_cands[random.randint(0, len(top_cands)-1)]
                
                code.append(idx)
                covered[balls[idx]] = True
                
            # Local Search: Remove redundant
            # Convert to set for fast lookup
            code_set = set(code)
            counts = np.zeros(N, dtype=np.int32)
            for idx in code:
                counts[balls[idx]] += 1
                
            changed = True
            while changed and time.time() - t_start < deadline - 5:
                changed = False
                for idx in list(code):
                    # Check if removing idx leaves all words in its ball covered by others
                    # i.e., min count in ball[idx] > 1
                    ball = balls[idx]
                    if np.min(counts[ball]) > 1:
                        # Remove
                        counts[ball] -= 1
                        code.remove(idx)
                        code_set.discard(idx)
                        changed = True
            
            if len(code) < best_len:
                best_len = len(code)
                best_code = code.copy()
                
    except Exception as e:
        pass # Fallback to pure greedy if LP fails

    # Strategy 2: Multiple Greedy Restarts (with random tie-breaking)
    # If LP didn't find something good, or just to ensure we have a solid baseline
    # The LP solution might be 186.something, rounding might get close.
    # Greedy usually gets ~240. LP guided rounding should be better.
    
    # If we have time, run a few pure greedy restarts to compare
    while time.time() - t_start < deadline - 2:
        covered = np.zeros(N, dtype=bool)
        counts = np.zeros(N, dtype=np.int32)
        code = []
        
        while not covered.all():
            gains = np.sum(~covered[balls], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0: break
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(idx)
            covered[balls[idx]] = True
            counts[balls[idx]] += 1
            
        # Local Search
        changed = True
        while changed and time.time() - t_start < deadline - 2:
            changed = False
            for idx in list(code):
                ball = balls[idx]
                if np.min(counts[ball]) > 1:
                    counts[ball] -= 1
                    code.remove(idx)
                    changed = True
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
    # If best_code is still None (shouldn't happen), fallback
    if best_code is None:
        # Simple greedy
        covered = np.zeros(N, dtype=bool)
        best_code = []
        while not covered.all():
            gains = np.sum(~covered[balls], axis=1)
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[0]
            best_code.append(idx)
            covered[balls[idx]] = True

    # Convert indices back to list of lists of integers
    result = []
    for idx in best_code:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
