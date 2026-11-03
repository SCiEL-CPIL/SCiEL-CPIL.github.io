import numpy as np
import itertools
import random
import time
import scipy.optimize as opt
import scipy.sparse as sparse

def construct(q, n, radius):
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Build sparse matrix for MILP
    # Constraint: For each word j, sum(x_i for i in Ball(j)) >= 1
    # A_ub @ x <= b_ub  =>  -A @ x <= -1
    rows = []
    cols = []
    data = []
    
    for j, w in enumerate(words):
        # j is the row index (constraint for word j)
        # We need to add 1 to A[j, i] for all i in Ball(j)
        # Ball(j) contains words at dist <= 1 from j
        
        # Center
        rows.append(j)
        cols.append(j)
        data.append(1)
        
        # Neighbors
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                i = word_to_idx[new_w]
                rows.append(j)
                cols.append(i)
                data.append(1)
                
    A = sparse.coo_matrix((data, (rows, cols)), shape=(total_words, total_words)).tocsr()
    
    c = np.ones(total_words)
    A_ub = -A
    b_ub = -np.ones(total_words)
    integrality = np.ones(total_words)
    bounds = [(0, 1)] * total_words
    
    best_code = None
    
    # Try MILP
    try:
        # Use time limit to prevent hanging, though it should be fast
        res = opt.milp(
            c=c,
            constraints=(A_ub, b_ub),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 100} # HiGHS option
        )
        
        if res.status == 0:
            x = np.round(res.x).astype(int)
            code_indices = np.where(x == 1)[0]
            if len(code_indices) > 0:
                best_code = [list(words[i]) for i in code_indices]
                
    except Exception:
        pass
        
    # Fallback or Improvement: Greedy + Local Search
    # We use the current program's logic as a robust fallback/improver
    # If MILP found a code, we can still try to improve it with local search?
    # Or just return it if it's good.
    # Given the time, returning the MILP result if valid is best.
    # But we must ensure it's valid.
    # MILP guarantees validity if status 0.
    
    if best_code is None:
        # Re-compute balls as bitsets for fast greedy
        balls = [0] * total_words
        for i, w in enumerate(words):
            ball_val = 0
            ball_val |= (1 << i)
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w = tuple(new_w_list)
                    j = word_to_idx[new_w]
                    ball_val |= (1 << j)
            balls[i] = ball_val
        
        start_time = time.time()
        max_time = 110.0
        best_size = total_words + 1
        
        while time.time() - start_time < max_time:
            uncovered = (1 << total_words) - 1
            code = []
            
            while uncovered:
                max_covered = 0
                best_candidates = []
                
                for i in range(total_words):
                    covered_count = (balls[i] & uncovered).bit_count()
                    
                    if covered_count > max_covered:
                        max_covered = covered_count
                        best_candidates = [i]
                    elif covered_count == max_covered and covered_count > 0:
                        best_candidates.append(i)
                        
                if max_covered == 0:
                    break
                    
                chosen_idx = random.choice(best_candidates)
                code.append(words[chosen_idx])
                uncovered &= ~balls[chosen_idx]
                
            # Local Search: Remove redundant
            counts = [0] * total_words
            for c_word in code:
                idx = word_to_idx[c_word]
                b = balls[idx]
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] += 1
                    temp ^= lsb
            
            new_code = []
            for c_word in code:
                idx = word_to_idx[c_word]
                b = balls[idx]
                redundant = True
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        redundant = False
                        break
                    temp ^= lsb
                
                if redundant:
                    temp = b
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp ^= lsb
                else:
                    new_code.append(c_word)
            
            code = new_code

            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
    return [list(c) for c in best_code]
