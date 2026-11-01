import time
import numpy as np
import random
import sys
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    
    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: all words (worst case) or greedy
        total = q ** n
        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = q ** i
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)
        
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(len(balls)):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break
        
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    
    # 1. Precompute balls and incidence structure
    # balls[x] is a list of y such that dist(x, y) <= 1
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)
    
    # Build incidence matrix A where A[y, x] = 1 if y in balls[x]
    # Note: We want to cover every y. So sum_x A[y, x] * z[x] >= 1
    # A is 6561 x 6561. This is too large to build explicitly in dense form for memory/time?
    # 6561*6561 * 1 byte ~ 43 MB. Manageable.
    # But linprog with 6561 variables and 6561 constraints might be slow.
    # Let's try to solve the LP relaxation.
    # Minimize sum(z) subject to A^T z >= 1, 0 <= z <= 1
    # Actually, scipy linprog handles sparse matrices efficiently.
    
    # Construct sparse A
    from scipy.sparse import lil_matrix, csr_matrix
    A_lil = lil_matrix((total, total), dtype=np.float64)
    for x in range(total):
        for y in balls[x]:
            A_lil[y, x] = 1.0
    
    A_csr = A_lil.tocsr()
    
    # Objective: minimize sum(z)
    c = np.ones(total, dtype=np.float64)
    
    # Constraints: A_csr @ z >= 1  =>  -A_csr @ z <= -1
    A_ub = -A_csr
    b_ub = -np.ones(total, dtype=np.float64)
    
    # Bounds: 0 <= z <= 1
    bounds = [(0, 1)] * total
    
    # Solve LP
    # Use 'highs' method which is fast
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={"presolve": True})
        if res.success:
            z_lp = res.x
            # Get dual variables (shadow prices) for the constraints
            # res.ineqlin.marginals contains the dual variables for A_ub z <= b_ub
            # Since A_ub = -A, the duals correspond to the covering constraints
            # The magnitude of the duals indicates how "tight" or "important" the constraint is.
            # High dual value (negative in minimization with <= constraints? Let's check sign)
            # In standard form min c^T x s.t. A_ub x <= b_ub, duals y >= 0.
            # The value is the change in objective per unit change in b_ub.
            # Here b_ub is -1.
            # Let's just use the magnitude of marginals as a priority score.
            # Actually, for covering problems, the dual variables of the covering constraints
            # often correlate with the "difficulty" of covering that point.
            # However, a simpler heuristic derived from LP is to round z_lp.
            # But we need a valid cover.
            
            # Strategy: Use LP solution to guide greedy selection.
            # Sort codewords by z_lp value descending? Or use duals?
            # Let's try a hybrid:
            # 1. Start with greedy to get a baseline.
            # 2. Use LP duals to break ties or weight the greedy.
            
            # Actually, a direct rounding of LP might not yield a valid cover.
            # We will use the LP duals to define a priority order for a greedy algorithm.
            # The dual variable for constraint y (covering point y) tells us the "cost" of covering y.
            # Points with high duals are "expensive" to cover, so we should prioritize codewords that cover them.
            
            # Let's extract duals. In HiGHS, res.ineqlin.marginals gives duals for A_ub x <= b_ub.
            # Since A_ub = -A, the duals are for -A z <= -1.
            # Let's just use the absolute values or the raw values.
            
            duals = res.ineqlin.marginals
            # The duals are typically non-negative for <= constraints in minimization.
            # Higher dual means covering that point is more critical.
            
            # Create a priority score for each codeword x.
            # Score(x) = sum_{y in balls[x]} duals[y]
            # This approximates the "value" of selecting x.
            
            scores = np.zeros(total, dtype=np.float64)
            # Efficient calculation: scores = A_csr.T @ duals
            # A_csr.T has shape (total, total)
            scores = A_csr.T @ duals
            
            # Now run a greedy algorithm, but instead of max uncovered count,
            # we use a weighted score: score[x] * (number of uncovered points in balls[x])
            # Or simply sort by score and pick if it covers new points?
            # Let's stick to standard greedy but use scores to break ties or weight.
            
            # Alternative: Use the LP solution z_lp directly.
            # Threshold z_lp at 0.5? No, that's not guaranteed valid.
            
            # Let's try: Greedy selection where the "gain" of adding x is
            # gain(x) = sum_{y in balls[x] \cap Uncovered} (1 + duals[y])
            # This biases towards covering "hard" points.
            
        else:
            # Fallback to standard greedy if LP fails
            duals = np.ones(total, dtype=np.float64)
            scores = np.array([len(b) for b in balls], dtype=np.float64)
            
    except Exception:
        duals = np.ones(total, dtype=np.float64)
        scores = np.array([len(b) for b in balls], dtype=np.float64)
    
    # 2. Greedy Construction with LP-guided weighting
    covered = np.zeros(total, dtype=bool)
    selected = []
    
    # Precompute ball indices for fast access
    # We need to quickly check how many uncovered points a ball covers.
    # We can maintain a count of uncovered points per codeword.
    # uncovered_count[x] = number of y in balls[x] such that not covered[y]
    
    uncovered_count = np.array([len(b) for b in balls], dtype=np.int32)
    
    # We also need to know the "dual-weighted" uncovered count to use as priority.
    # weighted_uncovered_count[x] = sum_{y in balls[x] \cap Uncovered} duals[y]
    # This is expensive to update. 
    # Simplification: Just use standard greedy (max uncovered) but if LP succeeded,
    # we can initialize the selection order or use a modified gain.
    
    # Let's try a simpler approach:
    # Use the LP solution to identify a "core" set of codewords with z_lp > 0.5.
    # Then fill the rest with greedy.
    
    if 'z_lp' in locals() and res.success:
        # Get indices where z_lp > 0.5
        # This might be a large set.
        # Let's just use the greedy approach but with a twist:
        # Sort candidates by their LP value descending? No, greedy is dynamic.
        
        # Let's stick to the weighted greedy idea but implement it efficiently.
        # We'll recompute the weighted gain only for the top K candidates in each step?
        # That might be too slow.
        
        # Let's just do standard greedy first to get a valid solution,
        # then use LP insights for local search?
        
        # Actually, let's try a different rounding:
        # Sort all codewords by z_lp descending.
        # Iterate through this sorted list.
        # If adding this codeword covers any new points, add it.
        # This is a "greedy by LP order" heuristic.
        
        sorted_indices = np.argsort(z_lp)[::-1]
        for idx in sorted_indices:
            if time.time() > deadline - 10:
                break
            # Check if this codeword covers any uncovered point
            # We need to check if any y in balls[idx] is not covered
            # This is O(17) per check. 6561 * 17 ~ 100k ops. Fast.
            add = False
            for y in balls[idx]:
                if not covered[y]:
                    add = True
                    break
            if add:
                selected.append(idx)
                for y in balls[idx]:
                    covered[y] = True
            if np.all(covered):
                break
        
        # If this didn't cover everything (should not happen if we iterate all),
        # fallback to standard greedy for remaining
        if not np.all(covered):
            # Standard greedy for remaining uncovered
            while not np.all(covered) and time.time() < deadline - 5:
                # Find best codeword
                best_idx = -1
                best_count = 0
                for x in range(total):
                    if x in selected:
                        continue
                    count = 0
                    for y in balls[x]:
                        if not covered[y]:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_idx = x
                if best_idx == -1:
                    break
                selected.append(best_idx)
                for y in balls[best_idx]:
                    covered[y] = True
    
    else:
        # Standard Greedy
        while not np.all(covered) and time.time() < deadline - 10:
            max_idx = np.argmax(uncovered_count)
            if uncovered_count[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for x in range(total):
                        if y in balls[x]:
                            uncovered_count[x] -= 1
    
    # 3. Local Search to improve
    current_set = set(selected)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in current_set:
        for y in balls[c]:
            cover_count[y] += 1
    
    def is_redundant(c):
        for y in balls[c]:
            if cover_count[y] == 1:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for c in candidates:
            if c not in current_set:
                continue
            if is_redundant(c):
                for y in balls[c]:
                    cover_count[y] -= 1
                current_set.remove(c)
                improved = True
    
    # 4. Try swaps: remove one, add one (if it reduces size? No, swap is 1-for-1)
    # We want to reduce size. So only removals are useful for size reduction.
    # But sometimes removing one requires adding two? No, that increases size.
    # So local search for size reduction is just finding redundant codewords.
    
    # To get smaller than greedy, we might need to find a non-greedy structure.
    # The LP-guided greedy might have found a better structure than pure greedy.
    
    best_solution = list(current_set)
    
    # Convert to list of digit lists
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)
    
    return result
