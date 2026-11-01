import time
import numpy as np
import random
import itertools
from scipy import sparse
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_lp(deadline)
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_lp(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls as a list of lists of indices
    # balls[c] = list of words w such that dist(c, w) <= 1
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    for c in range(total):
        ball = [c]
        temp = c
        for pos in range(n):
            weight = weights[pos]
            digit = (temp // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                w = c + (new_digit - digit) * weight
                ball.append(w)
        balls.append(ball)
        
    # Convert to numpy arrays for faster local search operations later
    # balls_np is (total, ball_size)
    balls_np = np.zeros((total, ball_size), dtype=np.int32)
    for i, b in enumerate(balls):
        balls_np[i] = b
        
    # Build sparse matrix M for LP
    # M[w, c] = 1 if w in balls[c]
    # We need A_ub such that A_ub @ x <= b_ub
    # Constraint: sum_{c: w in B(c)} x_c >= 1  => -sum ... <= -1
    # So A_ub = -M
    # Rows: w (0..total-1), Cols: c (0..total-1)
    
    rows = []
    cols = []
    data = []
    
    for c in range(total):
        for w in balls[c]:
            rows.append(w)
            cols.append(c)
            data.append(1.0)
            
    M = sparse.coo_matrix((data, (rows, cols)), shape=(total, total))
    M = M.tocsr()
    
    A_ub = -M
    b_ub = -np.ones(total)
    
    # Objective: minimize sum(x)
    c_obj = np.ones(total)
    
    bounds = [(0, 1) for _ in range(total)]
    
    best_size = total
    best_set = None
    
    # Try LP solve
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            x_frac = res.x
            # Rounding: Greedy based on x_frac
            # Sort codewords by x_frac descending
            order = np.argsort(-x_frac)
            
            selected = []
            covered = np.zeros(total, dtype=bool)
            
            for c in order:
                # Check if this codeword covers any uncovered word
                # If yes, add it
                # Optimization: check if any word in balls[c] is not covered
                # if not covered[balls_np[c]].all():
                #    selected.append(c)
                #    covered[balls_np[c]] = True
                
                # Using numpy for speed
                if not np.all(covered[balls_np[c]]):
                    selected.append(c)
                    covered[balls_np[c]] = True
                    
            selected = np.array(selected, dtype=np.int32)
            
            # Verify coverage (should be full by construction)
            # covered_sum = np.zeros(total, dtype=bool)
            # covered_sum[balls_np[selected]] = True
            # if not np.all(covered_sum):
            #     # Fallback to full set if something went wrong
            #     selected = np.arange(total, dtype=np.int32)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected.copy()
                
    except Exception as e:
        # Fallback to random greedy if LP fails
        pass

    # If LP didn't yield a result or failed, use random greedy as in seed
    if best_set is None or time.time() < deadline - 10.0:
         # Run the original greedy + local search as a fallback or refinement
         # We can use the LP solution as a start if available, otherwise random greedy
         
         if best_set is not None:
             current_best = best_set
         else:
             # Random Greedy
             np.random.seed(42)
             random.seed(42)
             selected = []
             uncovered = np.ones(total, dtype=bool)
             while np.any(uncovered) and time.time() < deadline - 5.0:
                 counts = uncovered[balls_np].sum(axis=1)
                 max_count = np.max(counts)
                 if max_count == 0: break
                 candidates = np.where(counts == max_count)[0]
                 best_idx = candidates[np.random.randint(len(candidates))]
                 selected.append(best_idx)
                 uncovered[balls_np[best_idx]] = False
             current_best = np.array(selected, dtype=np.int32)
             if len(current_best) < best_size:
                 best_size = len(current_best)
                 best_set = current_best.copy()
             else:
                 current_best = best_set
                 
         # Local Search on current_best
         if time.time() < deadline - 5.0 and current_best is not None:
             selected = current_best.copy()
             
             # Pruning
             cover_count = np.zeros(total, dtype=np.int32)
             cover_count[balls_np[selected]] += 1
             
             changed = True
             while changed and time.time() < deadline - 2.0:
                 mins = np.min(cover_count[balls_np[selected]], axis=1)
                 removable_mask = mins > 1
                 if not np.any(removable_mask):
                     break
                 remove_indices = selected[removable_mask]
                 selected = selected[~removable_mask]
                 cover_count[balls_np[remove_indices]] -= 1
                 changed = True
                 
             # Swaps
             # Precompute sets for swap check
             ball_sets = [set(row.tolist()) for row in balls_np]
             
             for _ in range(50): # More swaps if time permits
                 if time.time() > deadline - 2.0:
                     break
                 if len(selected) == 0:
                     break
                 
                 c_idx = np.random.randint(len(selected))
                 c = selected[c_idx]
                 c_prime = np.random.randint(total)
                 if c_prime == c:
                     continue
                 
                 bc = ball_sets[c]
                 bcp = ball_sets[c_prime]
                 
                 valid = True
                 for w in bc:
                     if w not in bcp:
                         if cover_count[w] == 1:
                             valid = False
                             break
                 
                 if not valid:
                     continue
                 
                 cover_count[balls_np[c]] -= 1
                 cover_count[balls_np[c_prime]] += 1
                 selected[c_idx] = c_prime
                 
                 # Re-prune
                 changed = True
                 while changed and time.time() < deadline - 2.0:
                     mins = np.min(cover_count[balls_np[selected]], axis=1)
                     removable_mask = mins > 1
                     if not np.any(removable_mask):
                         break
                     remove_indices = selected[removable_mask]
                     selected = selected[~removable_mask]
                     cover_count[balls_np[remove_indices]] -= 1
                     changed = True
                     
             if len(selected) < best_size:
                 best_size = len(selected)
                 best_set = selected.copy()

    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
        
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
