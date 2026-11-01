"""Ternary covering code via LP column generation + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    N = q ** n
    if N <= 0:
        return []

    # Convert index to base-q digits (most significant first)
    def idx_to_digits(idx):
        digits = [0] * n
        rem = idx
        for j in range(n - 1, -1, -1):
            digits[j] = rem % q
            rem //= q
        return digits

    # Precompute balls as sets (for N=2187 this is manageable)
    # ball[c] = set of words within Hamming distance `radius` of c
    balls = [set() for _ in range(N)]
    
    if radius == 1:
        for c in range(N):
            digits = idx_to_digits(c)
            balls[c].add(c)
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    # Compute new index
                    # index = sum(digits[j] * q^(n-1-j))
                    # delta = (new_val - orig) * q^(n-1-pos)
                    delta = (new_val - orig) * (q ** (n - 1 - pos))
                    new_idx = c + delta
                    if 0 <= new_idx < N:
                        balls[c].add(new_idx)
    else:
        # General radius: generate all words within distance
        # For small N, brute force is okay
        def hamming_dist(i, j):
            d = 0
            a, b = i, j
            for _ in range(n):
                if (a % q) != (b % q):
                    d += 1
                a //= q
                b //= q
            return d

        for c in range(N):
            for w in range(N):
                if hamming_dist(c, w) <= radius:
                    balls[c].add(w)

    # Convert balls to sorted lists for faster iteration if needed, but sets are fine for N=2187
    # For LP, we need a matrix or a way to evaluate constraints.
    # Let's use a sparse representation or just iterate.
    
    # --- Phase 1: LP Relaxation with Column Generation ---
    # Variables: x_c for c in 0..N-1
    # Minimize sum(x_c)
    # Subject to: sum_{c: w in ball(c)} x_c >= 1 for all w
    #             0 <= x_c <= 1
    
    # We will maintain a set of "active" columns (codewords in the master problem).
    # Initially, start with a greedy solution to have a feasible master.
    
    # Greedy initial solution
    uncovered = set(range(N))
    initial_code = []
    while uncovered:
        if time.time() > deadline - 20:
            break
        best_c = -1
        max_cov = 0
        # Sample a subset of candidates if N is large? For 2187, full scan is ~2187*|uncovered| ops.
        # Let's just do full scan but optimize.
        for c in range(N):
            if c in initial_code:
                continue
            cov = len(balls[c] & uncovered)
            if cov > max_cov:
                max_cov = cov
                best_c = c
        if best_c == -1 or max_cov == 0:
            break
        initial_code.append(best_c)
        uncovered -= balls[best_c]
        
    active_cols = set(initial_code)
    
    # Solve LP using scipy.optimize.linprog
    from scipy.optimize import linprog
    
    def solve_lp(active_cols_list):
        m = len(active_cols_list)
        if m == 0:
            return None, None
        
        # Objective: minimize sum(x)
        c_obj = np.ones(m)
        
        # Constraints: A_ub @ x <= b_ub
        # We have >= constraints. linprog handles <= by default.
        # So we convert -sum(x_c) <= -1
        # A_ub is N x m
        # A_ub[w, i] = -1 if word w is in ball(active_cols_list[i]) else 0
        
        A_ub = np.zeros((N, m), dtype=np.int8)
        for i, c in enumerate(active_cols_list):
            for w in balls[c]:
                A_ub[w, i] = -1
                
        b_ub = np.ones(N, dtype=np.int8) * -1
        
        bounds = [(0, 1)] * m
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0:
            return None, None
        return res.x, res
        
    # Pricing subproblem:
    # Given dual variables y_w >= 0 for each word w,
    # find c that minimizes x_c - sum_{w in ball(c)} y_w
    # Since x_c >= 0, the minimum reduced cost is min(0, min_c (1 - sum_{w in ball(c)} y_w))?
    # Actually, the reduced cost for column c is 1 - sum_{w in ball(c)} y_w.
    # If this is < 0, we add column c.
    # We want to find c that minimizes this value.
    
    # Extract duals from linprog result
    # In HiGHS, duals for inequality constraints are in res.ineqlin.marginals
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 30:
            break
            
        active_list = list(active_cols)
        x_sol, lp_res = solve_lp(active_list)
        if x_sol is None:
            break
            
        # Get duals
        # The constraints are A_ub @ x <= b_ub, which correspond to -sum(x_c) <= -1
        # The dual variable associated with constraint i is lambda_i.
        # In the standard form min c^T x s.t. A x <= b, the dual is max b^T y s.t. A^T y <= c, y <= 0.
        # Wait, linprog returns marginals for the constraints as defined.
        # If constraint is -sum(x) <= -1, the marginal is the shadow price.
        # Let y_w be the dual variable for the constraint "word w is covered".
        # The Lagrangian is sum(x_c) + sum_w y_w (1 - sum_{c:w in ball(c)} x_c)
        # = sum_c x_c (1 - sum_{w in ball(c)} y_w) + sum_w y_w
        # For the LP relaxation, the reduced cost of column c is 1 - sum_{w in ball(c)} y_w.
        
        # scipy linprog `res.ineqlin.marginals` gives the dual values for A_ub x <= b_ub.
        # Since our constraint is -sum(x) <= -1, the marginal should be <= 0?
        # Let's check the sign.
        # If we relax the constraint (make it easier), the objective decreases.
        # So the shadow price of a resource (coverage requirement) should be positive in the "maximize profit" sense,
        # but in minimization with <= constraints, the dual variables are typically <= 0.
        # Let's assume y_w = -res.ineqlin.marginals[w] to get positive weights.
        
        y = -res.ineqlin.marginals
        # y should be >= 0. Clip small negatives.
        y = np.maximum(y, 0)
        
        # Find column with most negative reduced cost
        best_rc = 0.0 # Current best reduced cost (we want to find < 0)
        best_c = -1
        
        # We can speed this up by precomputing something?
        # For each c, rc = 1 - sum(y[w] for w in balls[c])
        # This is O(N * |ball|) per iteration. |ball| is 1 + n(q-1) = 1 + 7*2 = 15 for q=3, n=1.
        # Wait, radius 1, q=3. Ball size is 1 + 7*(3-1) = 15.
        # So N * 15 = 2187 * 15 ~ 32k operations. Very fast.
        
        for c in range(N):
            if c in active_cols:
                continue
            s = 0.0
            for w in balls[c]:
                s += y[w]
            rc = 1.0 - s
            if rc < best_rc:
                best_rc = rc
                best_c = c
                
        if best_c == -1 or best_rc >= -1e-6:
            break # Optimal or no improvement
        
        active_cols.add(best_c)
        
    # Extract integer solution from LP
    # Round x_sol to get initial code for local search
    # We need to map active_cols back to x_sol
    active_list = list(active_cols)
    x_sol, _ = solve_lp(active_list)
    
    # Greedy rounding: pick columns with x > 0.5? Or sort by x descending?
    # Let's try a heuristic: start with columns where x > 0.5, then repair.
    lp_code = []
    if x_sol is not None:
        for i, c in enumerate(active_list):
            if x_sol[i] > 0.5:
                lp_code.append(c)
    
    # If LP code is not valid or empty, fall back to greedy
    def is_valid(code):
        cov = set()
        for c in code:
            cov |= balls[c]
        return len(cov) == N

    if not is_valid(lp_code):
        # Repair: add codewords to cover missing words
        covered = set()
        for c in lp_code:
            covered |= balls[c]
        missing = set(range(N)) - covered
        while missing:
            if time.time() > deadline - 10:
                break
            # Pick word w in missing, find best c to cover it
            # Actually, just pick any c that covers a missing word.
            # To be efficient, pick c that covers most missing words.
            best_c = -1
            max_cov = 0
            for c in range(N):
                if c in lp_code:
                    continue
                cov = len(balls[c] & missing)
                if cov > max_cov:
                    max_cov = cov
                    best_c = c
            if best_c == -1:
                break
            lp_code.append(best_c)
            missing -= balls[best_c]
            
    # --- Phase 2: Local Search ---
    # Use the LP solution as a seed for local search.
    # This local search is similar to the previous one but faster due to better start.
    
    code_set = lp_code
    
    # Build state
    counts = np.zeros(N, dtype=np.int32)
    for c in code_set:
        for w in balls[c]:
            counts[w] += 1
            
    # Helper: remove redundant
    def try_remove(i):
        c = code_set[i]
        # Check if all w in balls[c] have counts[w] > 1
        for w in balls[c]:
            if counts[w] == 1:
                return False
        # Remove
        for w in balls[c]:
            counts[w] -= 1
        code_set.pop(i)
        return True

    # Helper: swap
    def try_swap(i, new_c):
        old_c = code_set[i]
        if old_c == new_c or new_c in code_set:
            return False
        # Check validity: words in balls[old_c] \ balls[new_c] must have count >= 2
        for w in balls[old_c]:
            if w not in balls[new_c]:
                if counts[w] < 2:
                    return False
        # Perform swap
        for w in balls[old_c]:
            counts[w] -= 1
        for w in balls[new_c]:
            counts[w] += 1
        code_set[i] = new_c
        return True

    best_size = len(code_set)
    best_code = code_set.copy()
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Remove redundant
        changed = True
        while changed:
            changed = False
            for i in range(len(code_set) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                if try_remove(i):
                    improved = True
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = code_set.copy()
                        
        # 2. Swaps
        if time.time() < deadline - 10:
            # Try random swaps
            for _ in range(20): # passes
                if time.time() > deadline - 5:
                    break
                indices = list(range(len(code_set)))
                random.shuffle(indices)
                for i in indices:
                    if time.time() > deadline - 5:
                        break
                    # Try a few random candidates
                    for _ in range(5):
                        new_c = random.randint(0, N - 1)
                        if try_swap(i, new_c):
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
    # Final check
    if not is_valid(best_code):
        # Fallback to lp_code if it was valid, or greedy
        if is_valid(lp_code):
            best_code = lp_code
        else:
            # Re-run greedy
            uncovered = set(range(N))
            final_code = []
            while uncovered:
                best_c = -1
                max_cov = 0
                for c in range(N):
                    if c in final_code:
                        continue
                    cov = len(balls[c] & uncovered)
                    if cov > max_cov:
                        max_cov = cov
                        best_c = c
                if best_c == -1:
                    break
                final_code.append(best_c)
                uncovered -= balls[best_c]
            best_code = final_code

    # Convert to output format
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
            
    return [list(all_words[i]) for i in best_code]
# EVOLVE-BLOCK-END
