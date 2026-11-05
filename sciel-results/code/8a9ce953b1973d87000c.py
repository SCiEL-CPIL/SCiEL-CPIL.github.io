"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We want to minimize max(A) for a Sidon set A containing 0,
    # such that {a_i - a_j} contains 1..30.
    
    # Lower bound on diameter:
    # If we use k elements, we have k(k-1)/2 differences.
    # To cover 30 differences, we need k(k-1)/2 >= 30 => k >= 8 (28 diffs, not enough), k=9 (36 diffs).
    # So we need at least 9 elements.
    # With 9 elements, the max difference M must be at least large enough to hold 30 distinct values.
    # Trivially M >= 30.
    # But Sidon constraints force M to be larger.
    
    best_solution = None
    best_max = float('inf')
    
    # To speed up search, we can fix the size of the set?
    # Let's try sizes 9, 10, 11.
    
    def check_sidon_and_cover(A):
        if not A or 0 not in A:
            return False, 0, 0
        if len(A) > 32:
            return False, 0, 0
        
        diffs = set()
        max_val = max(A)
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False, 0, 0
                diffs.add(d)
        
        covered = sum(1 for d in range(1, 31) if d in diffs)
        if covered == 30:
            return True, max_val, covered
        return False, max_val, covered

    # Use ILP to find the optimal set for a fixed size k
    def solve_ilp(k, max_limit=1000):
        """
        Solve for a Sidon set of size k containing 0, covering 1..30, minimizing max element.
        Variables:
        x_i (binary) for i in 1..max_limit-1 (0 is fixed to be in the set)
        y_d (binary) for d in 1..max_limit-1 indicating if difference d is present
        
        Objective: Minimize sum(x_i * i) + large_penalty * (max_element_indicator)
        Actually, minimizing max element is equivalent to minimizing the sum if we force a compact structure,
        but strictly minimizing max(A) requires a variable M such that x_i <= M for all i, and minimize M.
        
        Let's define:
        x_i for i=1..L (L is a guess for upper bound on max element, e.g., 100)
        M: the maximum element.
        
        Constraints:
        1. Sum(x_i) = k - 1 (since 0 is fixed)
        2. x_i <= M for all i (implicitly true if M >= L, but we want to minimize M)
           Actually, if we minimize M subject to x_i * i <= M * x_i? No.
           Constraint: x_i <= M for all i? No, M is a variable.
           Constraint: i * x_i <= M for all i? Yes. If x_i=1, then i <= M.
           
        3. Sidon Property:
           For any distinct pairs (i,j) and (p,q) with i<j, p<q, and {i,j} != {p,q}:
           x_i + x_j + x_p + x_q <= 2  (to prevent both pairs existing)
           Wait, this is O(L^4). Too slow for L=100.
           
           Alternative formulation for Sidon:
           Let z_{i,j} be 1 if both i and j are in the set.
           z_{i,j} <= x_i, z_{i,j} <= x_j
           z_{i,j} >= x_i + x_j - 1
           
           Difference d = j - i.
           For each d, sum_{i} z_{i, i+d} <= 1.
           This ensures no two pairs have the same difference.
           
        4. Coverage:
           For each d in 1..30: sum_{i} z_{i, i+d} >= 1.
           
        Variables:
        x_1...x_L (binary)
        z_{i,j} for 1<=i<j<=L (binary)
        M (integer, lower bound 0, upper bound L)
        
        Total vars: L + L(L-1)/2 + 1.
        For L=100: 100 + 4950 + 1 ~ 5000 vars.
        Constraints:
        - Sum x_i = k-1 (1)
        - i * x_i <= M (L)
        - z_{i,j} <= x_i (L^2/2)
        - z_{i,j} <= x_j (L^2/2)
        - z_{i,j} >= x_i + x_j - 1 (L^2/2)
        - Sum_i z_{i, i+d} <= 1 for d=1..L-1 (L)
        - Sum_i z_{i, i+d} >= 1 for d=1..30 (30)
        
        Total constraints ~ 3 * L^2/2 + L + 31 ~ 1.5 L^2.
        For L=100, ~15000 constraints.
        SciPy milp can handle this size quickly.
        """
        
        # Determine L (upper bound for max element)
        # Start with a reasonable guess. Theoretical lower bound for k=9 is around 30-40?
        # Let's try increasing L if no solution is found.
        
        L = 100 # Initial guess for max element bound
        
        # Indices
        # x_i corresponds to index i (1..L)
        # z_{i,j} corresponds to pair (i,j) with 1<=i<j<=L
        
        n_x = L
        n_z = L * (L - 1) // 2
        n_M = 1
        
        total_vars = n_x + n_z + n_M
        
        # Variable indices
        # x_i: 0 .. L-1
        # z_{i,j}: L .. L + n_z - 1
        # M: L + n_z
        
        def z_idx(i, j):
            # i < j
            # Count pairs (i', j') with i' < j'
            # For a fixed i, the number of pairs is (L-1) + (L-2) + ... + (L-i)
            # Let's precompute offsets
            pass

        # Precompute z indices
        z_offsets = {}
        count = 0
        for i in range(1, L + 1):
            for j in range(i + 1, L + 1):
                z_offsets[(i, j)] = L + count
                count += 1
        
        M_idx = L + n_z
        
        # Objective: Minimize M
        c = np.zeros(total_vars)
        c[M_idx] = 1.0
        
        # Bounds
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        lb[M_idx] = 0
        ub[M_idx] = L
        
        # Integrality
        integrality = np.ones(total_vars)
        
        # Constraints
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        # 1. Sum(x_i) = k - 1
        row = np.zeros(total_vars)
        for i in range(1, L + 1):
            row[i - 1] = 1.0
        A_rows.append(row)
        lb_rows.append(k - 1)
        ub_rows.append(k - 1)
        
        # 2. i * x_i <= M  =>  i * x_i - M <= 0
        for i in range(1, L + 1):
            row = np.zeros(total_vars)
            row[i - 1] = i
            row[M_idx] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0)
        
        # 3. z_{i,j} <= x_i  =>  z_{i,j} - x_i <= 0
        # 4. z_{i,j} <= x_j  =>  z_{i,j} - x_j <= 0
        # 5. z_{i,j} >= x_i + x_j - 1  =>  -z_{i,j} + x_i + x_j >= 1  =>  x_i + x_j - z_{i,j} >= 1
        for i in range(1, L + 1):
            for j in range(i + 1, L + 1):
                idx_z = z_offsets[(i, j)]
                
                # z <= x_i
                row = np.zeros(total_vars)
                row[idx_z] = 1.0
                row[i - 1] = -1.0
                A_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(0)
                
                # z <= x_j
                row = np.zeros(total_vars)
                row[idx_z] = 1.0
                row[j - 1] = -1.0
                A_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(0)
                
                # x_i + x_j - z >= 1
                row = np.zeros(total_vars)
                row[i - 1] = 1.0
                row[j - 1] = 1.0
                row[idx_z] = -1.0
                A_rows.append(row)
                lb_rows.append(1.0)
                ub_rows.append(np.inf)
        
        # 6. Sidon: For each d in 1..L-1, sum_{i} z_{i, i+d} <= 1
        for d in range(1, L):
            row = np.zeros(total_vars)
            for i in range(1, L - d + 1):
                j = i + d
                if j <= L:
                    row[z_offsets[(i, j)]] = 1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(1.0)
            
        # 7. Coverage: For each d in 1..30, sum_{i} z_{i, i+d} >= 1
        for d in range(1, 31):
            row = np.zeros(total_vars)
            for i in range(1, L - d + 1):
                j = i + d
                if j <= L:
                    row[z_offsets[(i, j)]] = 1.0
            A_rows.append(row)
            lb_rows.append(1.0)
            ub_rows.append(np.inf)
            
        A_array = np.array(A_rows)
        lb_array = np.array(lb_rows)
        ub_array = np.array(ub_rows)
        
        constraints = LinearConstraint(A_array, lb_array, ub_array)
        bounds = Bounds(lb, ub)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # Extract solution
                x_vals = res.x[:n_x]
                M_val = int(round(res.x[M_idx]))
                
                A = [0]
                for i in range(1, L + 1):
                    if x_vals[i - 1] > 0.5:
                        A.append(i)
                        
                # Verify
                is_valid, max_val, covered = check_sidon_and_cover(A)
                if is_valid:
                    return A, max_val
            else:
                return None, float('inf')
                
        except Exception as e:
            return None, float('inf')
            
    # Try different sizes k
    # k=9 is minimal. k=10 might allow smaller M? Unlikely but possible.
    # Let's iterate k from 9 to 12.
    
    for k in range(9, 13):
        # Try increasing L bounds
        # Start with a lower bound estimate.
        # For k=9, M is likely < 100.
        L_guesses = [50, 80, 120, 200]
        
        for L in L_guesses:
            if time.time() > deadline:
                break
                
            A, max_val = solve_ilp(k, L)
            if A is not None:
                if max_val < best_max:
                    best_max = max_val
                    best_solution = A[:]
                    
            # If we found a solution with M close to L, maybe we need larger L?
            # But we are minimizing M, so if it finds a solution, M is optimal for that L bound?
            # No, if M < L, it's optimal. If M == L, it might be better with larger L?
            # No, if M is bounded by L, and we minimize M, the solver will find the smallest M <= L.
            # So if it finds M, it's the best for that k.
            # We don't need to try larger L for the same k unless it failed.
            # However, if it failed, we might need larger L to accommodate the set.
            
            if A is not None:
                break # Found optimal for this k
            
            if time.time() > deadline:
                break
                
    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if ILP fails or times out
        def greedy_construct():
            a = [0]
            used = set()
            for d in range(1, 31):
                if d in used:
                    continue
                found = False
                candidates = set()
                for ai in a:
                    candidates.add(ai + d)
                    if ai - d > 0:
                        candidates.add(ai - d)
                
                for x in sorted(candidates):
                    if x in a:
                        continue
                    new_diffs = set()
                    valid = True
                    for ai in a:
                        diff = abs(x - ai)
                        if diff in used or diff in new_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    if valid:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    # Add a large number
                    x = max(a) + d + 1
                    new_diffs = set()
                    valid = True
                    for ai in a:
                        diff = abs(x - ai)
                        if diff in used or diff in new_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    if valid:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                    else:
                        # Emergency
                        a.append(max(a) + 1000)
                        break
            return a
            
        return greedy_construct()

# EVOLVE-BLOCK-END
