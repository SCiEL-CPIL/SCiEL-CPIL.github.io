import time
from math import log
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def score_set(U):
    """
    Calculate the score for a given set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    # Efficiently compute sums and diffs
    # Using list comprehension for speed, though sets are still O(n^2)
    # For small n (up to ~20-30), this is fast enough.
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Strategy: Use MILP to find sets with high difference/sum ratio.
    # We search over a bounded range of maximum element M_max.
    # Variables: x_i for i in 0..M_max, binary.
    # x_0 = 1 (fixed).
    
    # To make MILP tractable, we need to linearize the objective.
    # The score is 1 + log(R) / log(D).
    # R = |U-U| / |U+U|.
    # Let N_diff = |U-U|, N_sum = |U+U|.
    # We want to maximize log(N_diff) - log(N_sum).
    # This is non-linear.
    
    # Alternative: Maximize N_diff - C * N_sum for some C?
    # Or simply maximize N_diff subject to N_sum <= K?
    
    # Let's try a different formulation.
    # We know that for "good" sets (like Sidon or similar), N_diff is large relative to N_sum.
    # Actually, |U-U| is always >= |U+U| for sets containing 0? No.
    # If U = {0, 1, 2}, U+U={0,1,2,3,4} (5), U-U={-2,-1,0,1,2} (5). Ratio 1.
    # If U = {0, 1, 3}, U+U={0,1,2,3,4,6} (6), U-U={-3,-2,-1,0,1,2,3} (7). Ratio 7/6.
    
    # The term log(D) in the denominator suggests we want small max(U) for a given ratio.
    # So we want to maximize Ratio / log(D).
    
    # Let's use a heuristic MILP objective:
    # Maximize: (Number of distinct differences) - lambda * (Number of distinct sums)
    # And constrain max(U) to be small.
    
    # However, counting distinct sums/diffs is non-linear (0-1 variables for each possible sum/diff).
    # Let y_s be 1 if sum s is in U+U.
    # Let z_d be 1 if diff d is in U-U.
    
    # Constraints:
    # For each possible sum s: y_s <= sum_{i+j=s} x_i x_j  (non-linear product)
    # Linearization: x_i x_j <= x_i, x_i x_j <= x_j, x_i x_j >= x_i + x_j - 1.
    # This introduces M^2 variables. If M=50, M^2=2500. Manageable.
    
    # Let's try M_max = 30.
    # Variables:
    # x[0..M] : binary, x[0]=1.
    # y[0..2M] : binary, y[s]=1 if s in U+U.
    # z[-M..M] : binary, z[d]=1 if d in U-U.
    # t[i][j] : binary, t[i][j]=1 if x_i=1 and x_j=1.
    
    # Objective: Maximize sum(z) - 2 * sum(y)
    # (Weighting sums higher to penalize them, as we want ratio > 1)
    
    # Actually, the score is log(R)/log(D).
    # Let's just maximize R = N_diff / N_sum? No, log is concave.
    # Let's maximize N_diff - N_sum. This encourages large gaps.
    
    # Let's implement this MILP.
    
    M_max = 40
    n_vars_x = M_max + 1
    n_vars_y = 2 * M_max + 1
    n_vars_z = 2 * M_max + 1
    n_vars_t = n_vars_x * n_vars_x
    
    total_vars = n_vars_x + n_vars_y + n_vars_z + n_vars_t
    
    # Index mapping
    def idx_x(i): return i
    def idx_y(s): return n_vars_x + s
    def idx_z(d): return n_vars_x + n_vars_y + (d + M_max)
    def idx_t(i, j): return n_vars_x + n_vars_y + n_vars_z + i * n_vars_x + j
    
    c = np.zeros(total_vars)
    
    # Objective: Maximize N_diff - 1.5 * N_sum
    # Coefficients for z (diffs)
    for d in range(-M_max, M_max + 1):
        c[idx_z(d)] = 1.0
    # Coefficients for y (sums)
    for s in range(0, 2 * M_max + 1):
        c[idx_y(s)] = -1.5
    
    # Constraints:
    # 1. x_0 = 1
    # 2. x_i + x_j - 1 <= t_ij <= x_i, t_ij <= x_j
    # 3. y_s >= sum_{i+j=s} t_ij - (n_vars_x - 2)  (If any pair sums to s, y_s can be 1. Actually y_s >= t_ij is not enough if multiple pairs. We need y_s >= t_ij for all i,j? No.
    #    We need: y_s = 1 if ANY t_ij=1 where i+j=s.
    #    Constraint: y_s >= t_ij for all i,j with i+j=s.
    #    Constraint: y_s <= sum_{i+j=s} t_ij (This is not linear in that form directly, but y_s <= sum t_ij works if we force y_s to be 0 if sum is 0? No.
    #    If sum t_ij = 0, then y_s <= 0 => y_s=0. Correct.
    #    If sum t_ij >= 1, y_s can be 1. Since we are maximizing sum(z) - sum(y), we want y_s to be as small as possible. So y_s will be 0 unless forced to be 1.
    #    Wait, the objective penalizes y_s. So the solver will set y_s=0 unless it is forced to be 1.
    #    How is it forced? By the constraint y_s >= t_ij.
    #    So if any t_ij=1, y_s must be 1.
    #    This is correct.
    
    # Similarly for z_d:
    # z_d >= t_ij for all i,j with i-j=d.
    # z_d <= sum_{i-j=d} t_ij.
    # Since we maximize sum(z), we want z_d=1 if possible.
    # If any t_ij=1, z_d >= 1 => z_d=1.
    
    # Let's build the constraint matrix A_ub and b_ub, or use LinearConstraint.
    # It's easier to build lists of rows.
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # Constraint 1: x_0 = 1
    row = np.zeros(total_vars)
    row[idx_x(0)] = 1
    A_rows.append(row)
    b_lower.append(1)
    b_upper.append(1)
    
    # Constraints for t_ij
    for i in range(n_vars_x):
        for j in range(n_vars_x):
            ti = idx_t(i, j)
            xi = idx_x(i)
            xj = idx_x(j)
            
            # t_ij <= x_i
            row = np.zeros(total_vars)
            row[ti] = 1
            row[xi] = -1
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(0)
            
            # t_ij <= x_j
            row = np.zeros(total_vars)
            row[ti] = 1
            row[xj] = -1
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(0)
            
            # t_ij >= x_i + x_j - 1
            row = np.zeros(total_vars)
            row[ti] = -1
            row[xi] = 1
            row[xj] = 1
            A_rows.append(row)
            b_lower.append(-1)
            b_upper.append(np.inf)
            
    # Constraints for y_s
    for s in range(2 * M_max + 1):
        ys = idx_y(s)
        # y_s <= sum_{i+j=s} t_ij
        row = np.zeros(total_vars)
        row[ys] = 1
        pairs = []
        for i in range(n_vars_x):
            j = s - i
            if 0 <= j < n_vars_x:
                row[idx_t(i, j)] = -1
                pairs.append((i, j))
        
        if pairs:
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(0)
            
        # y_s >= t_ij for each pair
        for i, j in pairs:
            row = np.zeros(total_vars)
            row[ys] = 1
            row[idx_t(i, j)] = -1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)
            
    # Constraints for z_d
    for d in range(-M_max, M_max + 1):
        zd = idx_z(d)
        # z_d <= sum_{i-j=d} t_ij
        row = np.zeros(total_vars)
        row[zd] = 1
        pairs = []
        for i in range(n_vars_x):
            j = i - d
            if 0 <= j < n_vars_x:
                row[idx_t(i, j)] = -1
                pairs.append((i, j))
        
        if pairs:
            A_rows.append(row)
            b_lower.append(-np.inf)
            b_upper.append(0)
            
        # z_d >= t_ij for each pair
        for i, j in pairs:
            row = np.zeros(total_vars)
            row[zd] = 1
            row[idx_t(i, j)] = -1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)
            
    if not A_rows:
        return best_set
        
    A = np.array(A_rows)
    lb = np.array(b_lower)
    ub = np.array(b_upper)
    
    # Bounds for variables
    # x: binary
    # y, z: binary
    # t: binary
    bounds = Bounds(
        lb=np.zeros(total_vars),
        ub=np.ones(total_vars)
    )
    
    # Integrality
    integrality = np.ones(total_vars)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            x_sol = res.x[:n_vars_x]
            U = [i for i in range(n_vars_x) if x_sol[i] > 0.5]
            
            if 0 in U and len(U) > 1:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
        elif res.status == 1: # Feasible
            x_sol = res.x[:n_vars_x]
            U = [i for i in range(n_vars_x) if x_sol[i] > 0.5]
            if 0 in U and len(U) > 1:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                    
    except Exception as e:
        # Fallback if MILP fails
        pass

    # Local search on the best set found so far
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 2 and iteration < 50:
            improved = False
            iteration += 1
            n = len(current)
            
            # Try replacing each element
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old_val = current[i]
                # Search range
                search_range = max(10, int(sqrt(old_val)))
                for delta in range(-search_range, search_range + 1):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 25 and time.time() < deadline - 2:
                max_val = max(current)
                # Try values in gaps
                candidates = set()
                for i in range(n):
                    for j in range(n):
                        mid = (current[i] + current[j]) // 2
                        if current[i] < mid < current[j]:
                            candidates.add(mid)
                candidates.add(max_val + 1)
                candidates.add(max_val + 2)
                
                for new_val in candidates:
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                        
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 2:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
