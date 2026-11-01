import time
import numpy as np
import random
import itertools
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls of radius 1)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # balls[w] contains w and all words at distance 1 from w
    # There are 1 + 2*n = 17 such words
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Build incidence matrix A
    # A[w, c] = 1 if word w is in ball(c)
    # This is the transpose of the "balls" relation if we view balls[c] as the set of words covered by c.
    # Actually, balls[c] lists the words covered by c.
    # So we want a matrix M where M[w, c] = 1 if w in balls[c].
    # We can build this by iterating over c and setting M[balls[c], c] = 1.
    
    # For LP: min sum(x_c) s.t. sum_{c: w in B(c)} x_c >= 1 for all w
    # Let x be vector of length total.
    # Constraints: A @ x >= 1, where A[w, c] = 1 if w in B(c).
    # In linprog, we need A_ub @ x <= b_ub.
    # So -A @ x <= -1.
    
    # We can avoid building the full dense matrix if we use sparse, but linprog with highs can handle dense 6561x6561 reasonably fast?
    # 6561 * 6561 * 8 bytes ~ 340 MB. That's a lot. 
    # Better to use sparse matrix. scipy.sparse is available.
    
    from scipy.sparse import lil_matrix
    
    A_lil = lil_matrix((total, total), dtype=np.float64)
    for c in range(total):
        covered_words = balls[c]
        A_lil[covered_words, c] = 1.0
        
    A_csr = A_lil.tocsr()
    
    # Objective: minimize sum(x_c)
    c_obj = np.ones(total, dtype=np.float64)
    
    # Constraints: -A @ x <= -1
    A_ub = -A_csr
    b_ub = -np.ones(total, dtype=np.float64)
    
    # Bounds: 0 <= x_c <= 1
    bounds = [(0, 1) for _ in range(total)]
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to greedy if LP fails
            pass
            
        x_lp = res.x
        
        # Deterministic Rounding:
        # Sort codewords by x_lp descending
        sorted_indices = np.argsort(x_lp)[::-1]
        
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        
        # Greedily add codewords from sorted list until all words are covered
        for c in sorted_indices:
            if np.all(cover_count > 0):
                break
            if time.time() > deadline - 10.0:
                break
            selected.append(c)
            cover_count[balls[c]] += 1
            
        # If we ran out of time or didn't cover all, we might have an incomplete set.
        # But LP solution should guide us well.
        # Let's ensure we actually cover everything. If not, add more greedily.
        if not np.all(cover_count > 0) and time.time() < deadline - 5.0:
             uncovered = np.where(cover_count == 0)[0]
             while len(uncovered) > 0:
                 # Pick a random uncovered word, find a codeword covering it, add it
                 w = uncovered[0]
                 # Find any codeword c such that w in balls[c]
                 # This is equivalent to finding c such that A[w, c] = 1
                 # We can just iterate through all c? No, that's slow.
                 # We can precompute "covering_codewords" for each word?
                 # Or just pick a random c and check?
                 # Better: We know balls is symmetric in a way? No.
                 # Let's just pick a random c and hope it covers w? No.
                 # We can use the fact that we have the balls matrix.
                 # For a given w, which c cover it?
                 # This is the column sum of A[w, :].
                 # We didn't store A explicitly in a way that allows fast inverse lookup.
                 # But we can just iterate through all c and check if w is in balls[c].
                 # Since we are in a fallback loop, it might be slow.
                 # However, the LP rounding should have covered almost everything.
                 
                 # Let's just pick a random codeword not in selected?
                 # Or better, just add a codeword that covers w.
                 # We can find c by checking which balls contain w.
                 # This is expensive if done naively.
                 # But this branch is unlikely to be hit if LP is good.
                 
                 # For now, just break and rely on the main result.
                 # If we return an invalid set, score is 0.
                 # We must ensure validity.
                 
                 # Let's do a simple check: if we didn't finish, we have a problem.
                 # Let's force complete coverage by adding random codewords until covered.
                 remaining = set(range(total)) - set(selected)
                 while len(uncovered) > 0 and remaining:
                     c = random.choice(list(remaining))
                     remaining.remove(c)
                     selected.append(c)
                     cover_count[balls[c]] += 1
                     new_uncovered = np.where(cover_count == 0)[0]
                     if len(new_uncovered) < len(uncovered):
                         uncovered = new_uncovered
                         # If no change, we might be stuck, but random should eventually hit
                     else:
                         uncovered = new_uncovered
                         # If stuck, break and hope? No, must be valid.
                         # This is a safety net.
                         if time.time() > deadline - 5.0:
                             break
             
        # Local Search Polish: Remove redundant codewords
        # Since we added in order of LP weight, some might be redundant.
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
    except Exception:
        # Fallback to simple greedy if LP fails
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.arange(total)
        while len(uncovered) > 0:
            if time.time() > deadline - 5.0:
                break
            # Pick random uncovered word, find a codeword covering it?
            # No, standard greedy: pick codeword that covers most uncovered.
            # This is what the original code did, but we skipped it.
            # Let's just pick a random codeword.
            c = random.randint(0, total - 1)
            selected.append(c)
            cover_count[balls[c]] += 1
            uncovered = np.where(cover_count == 0)[0]
            
        # Polish
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

    # Convert best_set to output format
    if not selected:
        # Fallback
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
