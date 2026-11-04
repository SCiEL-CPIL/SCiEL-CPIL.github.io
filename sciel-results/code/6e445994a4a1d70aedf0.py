"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 18

    # Build the schoolbook decomposition as a guaranteed valid fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    # Precompute the 18 (i,j,k) index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    # Precompute row indices for each triple
    triple_rows = [(i * m + j, j * p + k, k * n + i) for (i, j, k) in triples]

    def check_exact(U, V, W, rank):
        """Check if U, V, W form a valid exact decomposition using Fraction arithmetic."""
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        for (u_idx, v_idx, w_idx) in triple_rows:
            total = Fraction(0)
            for c in range(rank):
                total += U_frac[u_idx][c] * V_frac[v_idx][c] * W_frac[w_idx][c]
            if total != 1:
                return False
        return True

    def check_float(U, V, W, rank, tol=1e-9):
        """Quick float check."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for (u_idx, v_idx, w_idx) in triple_rows:
            val = np.dot(np.dot(U_arr[u_idx, :], V_arr[v_idx, :]), W_arr[w_idx, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    def error_metric(U, V, W, rank):
        """Compute total absolute error across all triples (float, fast)."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        total_err = 0.0
        for (u_idx, v_idx, w_idx) in triple_rows:
            val = np.dot(np.dot(U_arr[u_idx, :], V_arr[v_idx, :]), W_arr[w_idx, :])
            total_err += abs(val - 1.0)
        return total_err

    def solve_linear_for_W(U, V, rank):
        """
        Given U and V, try to solve for W exactly.
        For each triple (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        This is a system of n*p equations in n*p*rank unknowns (W has p*n rows, rank cols).
        Actually W is (p*n, rank). The constraints are:
        For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        Let's reorganize: for fixed k and i, the row W[k*n+i,:] appears.
        But the constraint couples all (i,j,k). Let's think of it as:
        For each (i,j,k): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        
        This is a linear system in the entries of W. Let's set up the matrix.
        Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
        Total variables: p*n*rank
        Equations: n*m*p
        
        Since n*m*p = p*n*rank only if m=rank... no.
        n*m*p = 18, p*n*rank = 6*rank.
        For rank=3: 18 equations, 18 variables -> square system!
        For rank=4: 18 equations, 24 variables -> underdetermined.
        
        Let's solve the square system for rank=3 first using Fraction.
        """
        num_eqs = n * m * p  # 18
        num_vars = p * n * rank  # 6 * rank
        
        if num_vars < num_eqs:
            return None  # underdetermined in wrong direction (more eqs than vars)
        
        # Build the linear system A * x = b
        # x is flattened W: x[k*n*rank + i*rank + c] = W[k*n+i, c]
        # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # => sum_c (U[i*m+j,c] * V[j*p+k,c]) * x[k*n*rank + i*rank + c] = 1
        
        # We'll use numpy for the linear solve, then verify with Fraction
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for eq_idx, (i, j, k) in enumerate(triples):
            u_idx = i * m + j
            v_idx = j * p + k
            w_row_base = k * n * rank + i * rank
            for c in range(rank):
                coeff = U[u_idx][c] * V[v_idx][c]
                A[eq_idx, w_row_base + c] = coeff
        
        # Check if the system is consistent and solvable
        try:
            # Use least squares / solve
            if num_vars == num_eqs:
                x = np.linalg.solve(A, b)
            else:
                # Underdetermined: find minimum norm solution
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-7:
                    return None
        
        except np.linalg.LinAlgError:
            return None
        
        # Round to nearest half-integer
        W_candidate = [[0] * rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    val = x[k * n * rank + i * rank + c]
                    # Round to nearest 0.5
                    rounded = round(val * 2) / 2
                    W_candidate[k * n + i][c] = rounded
        
        # Verify with exact arithmetic
        if check_exact(U, V, W_candidate, rank):
            return W_candidate
        
        # Try rounding to nearest integer
        W_candidate2 = [[0] * rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    val = x[k * n * rank + i * rank + c]
                    rounded = round(val)
                    W_candidate2[k * n + i][c] = rounded
        if check_exact(U, V, W_candidate2, rank):
            return W_candidate2
        
        return None

    def solve_linear_for_U(V, W, rank):
        """Given V and W, solve for U."""
        num_eqs = n * m * p
        num_vars = n * m * rank
        
        if num_vars < num_eqs:
            return None
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for eq_idx, (i, j, k) in enumerate(triples):
            v_idx = j * p + k
            w_idx = k * n + i
            u_row_base = i * m * rank + j * rank
            for c in range(rank):
                coeff = V[v_idx][c] * W[w_idx][c]
                A[eq_idx, u_row_base + c] = coeff
        
        try:
            if num_vars == num_eqs:
                x = np.linalg.solve(A, b)
            else:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-7:
                    return None
        except np.linalg.LinAlgError:
            return None
        
        U_candidate = [[0] * rank for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    val = x[i * m * rank + j * rank + c]
                    rounded = round(val * 2) / 2
                    U_candidate[i * m + j][c] = rounded
        
        if check_exact(U_candidate, V, W, rank):
            return U_candidate
        
        U_candidate2 = [[0] * rank for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    val = x[i * m * rank + j * rank + c]
                    rounded = round(val)
                    U_candidate2[i * m + j][c] = rounded
        if check_exact(U_candidate2, V, W, rank):
            return U_candidate2
        
        return None

    def solve_linear_for_V(U, W, rank):
        """Given U and W, solve for V."""
        num_eqs = n * m * p
        num_vars = m * p * rank
        
        if num_vars < num_eqs:
            return None
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for eq_idx, (i, j, k) in enumerate(triples):
            u_idx = i * m + j
            w_idx = k * n + i
            v_row_base = j * p * rank + k * rank
            for c in range(rank):
                coeff = U[u_idx][c] * W[w_idx][c]
                A[eq_idx, v_row_base + c] = coeff
        
        try:
            if num_vars == num_eqs:
                x = np.linalg.solve(A, b)
            else:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-7:
                    return None
        except np.linalg.LinAlgError:
            return None
        
        V_candidate = [[0] * rank for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    val = x[j * p * rank + k * rank + c]
                    rounded = round(val * 2) / 2
                    V_candidate[j * p + k][c] = rounded
        
        if check_exact(U, V_candidate, W, rank):
            return V_candidate
        
        V_candidate2 = [[0] * rank for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    val = x[j * p * rank + k * rank + c]
                    rounded = round(val)
                    V_candidate2[j * p + k][c] = rounded
        if check_exact(U, V_candidate2, W, rank):
            return V_candidate2
        
        return None

    def local_search(U, V, W, rank, max_iters=5000, timeout=15):
        """Local search to fix violations using half-integer steps."""
        start = time.time()
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        current_err = error_metric(U, V, W, rank)
        
        for it in range(max_iters):
            if time.time() - start > timeout:
                break
            if current_err < 1e-9:
                if check_exact(U, V, W, rank):
                    return U, V, W, True
                break
            
            # Find the best single-element perturbation
            best_err = current_err
            best_change = None
            
            # Try perturbing random elements
            for _ in range(30):
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, nm - 1)
                    col = random.randint(0, rank - 1)
                    old_val = U[row][col]
                elif mat_choice == 'V':
                    row = random.randint(0, mp - 1)
                    col = random.randint(0, rank - 1)
                    old_val = V[row][col]
                else:
                    row = random.randint(0, pn - 1)
                    col = random.randint(0, rank - 1)
                    old_val = W[row][col]
                
                delta = random.choice([-1, -0.5, 0.5, 1])
                new_val = old_val + delta
                
                if mat_choice == 'U':
                    U[row][col] = new_val
                elif mat_choice == 'V':
                    V[row][col] = new_val
                else:
                    W[row][col] = new_val
                
                new_err = error_metric(U, V, W, rank)
                
                if new_err < best_err:
                    best_err = new_err
                    best_change = (mat_choice, row, col, old_val, new_val)
                else:
                    # Revert
                    if mat_choice == 'U':
                        U[row][col] = old_val
                    elif mat_choice == 'V':
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
            
            if best_change is None:
                break
            
            mat_choice, row, col, old_val, new_val = best_change
            if mat_choice == 'U':
                U[row][col] = new_val
            elif mat_choice == 'V':
                V[row][col] = new_val
            else:
                W[row][col] = new_val
            current_err = best_err
        
        return U, V, W, check_exact(U, V, W, rank)

    deadline = time.time() + 110  # Safety margin

    # Strategy 1: Try to solve linear systems for lower ranks
    for target_rank in range(schoolbook_rank - 1, 1, -1):
        if time.time() > deadline - 20:
            break
        
        num_attempts = 10
        for attempt in range(num_attempts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with half-integers
            U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(target_rank)] for _ in range(pn)]
            
            # Quick check
            if check_exact(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
            
            # Try solving for W given U, V
            W_sol = solve_linear_for_W(U, V, target_rank)
            if W_sol is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W_sol
                break
            
            # Try solving for U given V, W
            U_sol = solve_linear_for_U(V, W, target_rank)
            if U_sol is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V, W
                break
            
            # Try solving for V given U, W
            V_sol = solve_linear_for_V(U, W, target_rank)
            if V_sol is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U, V_sol, W
                break
            
            # Local search
            U_ls, V_ls, W_ls, success = local_search(U, V, W, target_rank, max_iters=3000, timeout=5)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                break
        
        if best_rank < schoolbook_rank:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
