import time
import numpy as np
import itertools
import random
from scipy.optimize import linprog
import scipy.sparse as sp

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*(q-1) = 15
    
    t_start = time.time()
    deadline = 115.0 # Safety margin (limit is 120s)
    
    # Precompute all words as indices 0..2186
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx

    # Precompute balls: balls[i] is a list of indices of words within distance radius of word i
    balls = np.empty((N, 1 + n*(q-1)), dtype=np.int32)
    
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
        
    # Initial Cover: Use LP relaxation + greedy rounding as in seed, but faster.
    # Build sparse matrix M
    row_indices = []
    col_indices = []
    data = []
    for i in range(N):
        for j in balls[i]:
            row_indices.append(j)
            col_indices.append(i)
            data.append(1)
            
    M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    
    best_code = None
    best_len = N
    
    # Strategy 1: LP Relaxation to get a good starting point
    try:
        c = np.ones(N)
        A_ub = -M.toarray()
        b_ub = -np.ones(N)
        bounds = [(0, 1)] * N
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            x = res.x
            sorted_indices = np.argsort(x)[::-1]
            
            covered = np.zeros(N, dtype=bool)
            code = []
            
            # Greedy rounding based on LP
            for idx in sorted_indices:
                if time.time() - t_start > deadline - 20: break
                if x[idx] > 0.5:
                    ball = balls[idx]
                    if not np.all(covered[ball]):
                        code.append(idx)
                        covered[ball] = True
            
            # Greedy fill remaining
            while not covered.all():
                if time.time() - t_start > deadline - 20: break
                gains = np.sum(~covered[balls], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0: break
                candidates = np.where(gains == max_gain)[0]
                # Prefer high LP value
                cand_scores = x[candidates]
                max_cand_score = np.max(cand_scores)
                top_cands = candidates[cand_scores == max_cand_score]
                idx = top_cands[random.randint(0, len(top_cands)-1)]
                
                code.append(idx)
                covered[balls[idx]] = True
                
            if len(code) < best_len:
                best_len = len(code)
                best_code = code.copy()
    except Exception:
        pass

    # If LP failed, use pure greedy
    if best_code is None:
        covered = np.zeros(N, dtype=bool)
        best_code = []
        while not covered.all():
            gains = np.sum(~covered[balls], axis=1)
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[0]
            best_code.append(idx)
            covered[balls[idx]] = True
        best_len = len(best_code)

    # Strategy 2: Simulated Annealing with Removals and Swaps
    # This is the materially different part: exploring the space of valid covers
    # via swaps to find removable codewords.
    
    current_code = best_code.copy()
    current_len = len(current_code)
    code_set = set(current_code)
    
    # Precompute ball lists as lists for faster iteration if needed, 
    # but numpy arrays are fine.
    
    # Maintain coverage counts
    coverage_counts = np.zeros(N, dtype=np.int32)
    for idx in current_code:
        coverage_counts[balls[idx]] += 1
    
    # Helper to check if valid
    def is_valid():
        return np.min(coverage_counts) >= 1
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    # Precompute a list of all indices for random selection
    all_indices = list(range(N))
    
    while time.time() - t_start < deadline - 5 and T > min_T:
        # Try a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try Removal
            if not current_code: continue
            idx_to_remove = random.choice(current_code)
            
            # Tentatively remove
            ball = balls[idx_to_remove]
            coverage_counts[ball] -= 1
            
            if is_valid():
                # Accept removal
                current_code.remove(idx_to_remove)
                code_set.discard(idx_to_remove)
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_code = current_code.copy()
            else:
                # Revert
                coverage_counts[ball] += 1
                
        elif move_type < 0.7:
            # Try Swap
            if not current_code: continue
            idx_to_remove = random.choice(current_code)
            idx_to_add = random.choice(all_indices)
            
            if idx_to_add in code_set:
                continue
            
            # Tentatively remove
            ball_rem = balls[idx_to_remove]
            coverage_counts[ball_rem] -= 1
            
            # Tentatively add
            ball_add = balls[idx_to_add]
            coverage_counts[ball_add] += 1
            
            if is_valid():
                # Accept swap
                current_code.remove(idx_to_remove)
                code_set.discard(idx_to_remove)
                current_code.append(idx_to_add)
                code_set.add(idx_to_add)
            else:
                # Revert
                coverage_counts[ball_add] -= 1
                coverage_counts[ball_rem] += 1
                
        else:
            # Try Add (only if we are stuck and want to explore larger sets? 
            # Usually we want to minimize, so adding is bad unless it enables a removal.
            # But SA usually allows moves that increase cost if T is high.
            # Here "cost" is size. Adding increases size by 1.
            # We can allow this if it leads to a better local structure.
            idx_to_add = random.choice(all_indices)
            if idx_to_add in code_set:
                continue
            
            ball_add = balls[idx_to_add]
            coverage_counts[ball_add] += 1
            
            # Always accept add in SA if T is high? 
            # Actually, adding doesn't help minimize directly. 
            # It might help if we then remove 2 codewords? 
            # Let's stick to Removal and Swap for now, they are more targeted.
            # Revert add
            coverage_counts[ball_add] -= 1
            
        T *= cooling
        
    # Final Local Search: Greedy Removal
    changed = True
    while changed and time.time() - t_start < deadline - 2:
        changed = False
        for idx in list(current_code):
            ball = balls[idx]
            if np.min(coverage_counts[ball]) > 1:
                coverage_counts[ball] -= 1
                current_code.remove(idx)
                code_set.discard(idx)
                current_len -= 1
                changed = True
                if current_len < best_len:
                    best_len = current_len
                    best_code = current_code.copy()

    # Ensure best_code is the smallest found
    if best_code is None:
        best_code = current_code
        
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
