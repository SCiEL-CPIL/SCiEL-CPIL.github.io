import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def verify_decomposition_np(U, V, W):
        """Verify using numpy arrays."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def current_solution():
        """Rank 50 Strassen-based solution."""
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def compute_errors(U, V, W, r):
        """Compute all tensor entry errors."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    errors[i, j, k] = val - 1.0
        return errors

    def find_2x3x3_decomposition(max_time=30):
        """Find a rank-10 decomposition of 2×3×3 matrix multiplication."""
        if time.time() > start_time + max_time:
            return None
        
        # 2×3×3: U is 6×10, V is 9×10, W is 6×10
        # Entries: U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
        # Equations: sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1 for all i,j,k
        
        rng = np.random.RandomState(42)
        
        for trial in range(20):
            if time.time() > start_time + max_time:
                break
            
            # Initialize with random sparse structure
            U = np.zeros((6, 10), dtype=np.float64)
            V = np.zeros((9, 10), dtype=np.float64)
            W = np.zeros((6, 10), dtype=np.float64)
            
            # Give each column a few nonzero entries
            for c in range(10):
                # U: pick 2-3 rows
                u_idx = rng.choice(6, size=rng.randint(2, 4), replace=False)
                for idx in u_idx:
                    U[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
                
                v_idx = rng.choice(9, size=rng.randint(2, 4), replace=False)
                for idx in v_idx:
                    V[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
                
                w_idx = rng.choice(6, size=rng.randint(2, 4), replace=False)
                for idx in w_idx:
                    W[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
            
            # Iterative refinement
            for it in range(5000):
                if time.time() > start_time + max_time:
                    break
                
                # Compute errors for 2×3×3
                errs = np.zeros((2, 3, 3))
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            errs[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i]) - 1.0
                
                max_err = np.max(np.abs(errs))
                if max_err < 1e-10:
                    # Check if we can round to integers/half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    ok = True
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                val = np.sum(U_r[i*3+j] * V_r[j*3+k] * W_r[k*2+i])
                                if abs(val - 1) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        return U_r, V_r, W_r
                    break
                
                # Find worst equation
                worst = np.unravel_index(np.argmax(np.abs(errs)), errs.shape)
                wi, wj, wk = worst
                
                # Try to fix by perturbing U[wi*3+wj, :], V[wj*3+wk, :], W[wk*2+wi, :]
                u_row = wi * 3 + wj
                v_row = wj * 3 + wk
                w_row = wk * 2 + wi
                
                improved = False
                for _ in range(20):
                    mat = rng.randint(3)
                    col = rng.randint(10)
                    if mat == 0:
                        old = U[u_row, col]
                        U[u_row, col] = old + rng.choice([-1.0, 1.0])
                    elif mat == 1:
                        old = V[v_row, col]
                        V[v_row, col] = old + rng.choice([-1.0, 1.0])
                    else:
                        old = W[w_row, col]
                        W[w_row, col] = old + rng.choice([-1.0, 1.0])
                    
                    # Recompute max error
                    errs_new = np.zeros((2, 3, 3))
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                errs_new[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i]) - 1.0
                    
                    new_max = np.max(np.abs(errs_new))
                    if new_max < max_err * 0.95:
                        max_err = new_max
                        improved = True
                        break
                    else:
                        if mat == 0:
                            U[u_row, col] = old
                        elif mat == 1:
                            V[v_row, col] = old
                        else:
                            W[w_row, col] = old
                
                if not improved:
                    # Broader perturbation
                    mat = rng.randint(3)
                    if mat == 0:
                        row = rng.randint(6)
                        col = rng.randint(10)
                        U[row, col] += rng.choice([-1.0, 1.0])
                    elif mat == 1:
                        row = rng.randint(9)
                        col = rng.randint(10)
                        V[row, col] += rng.choice([-1.0, 1.0])
                    else:
                        row = rng.randint(6)
                        col = rng.randint(10)
                        W[row, col] += rng.choice([-1.0, 1.0])
        
        return None

    def find_2x3x3_decomposition_v2(max_time=30):
        """Alternative: use structured initialization based on known patterns."""
        if time.time() > start_time + max_time:
            return None
        
        # Try a more structured approach
        # The 2×3×3 multiplication: C = A*B
        # A is 2×3, B is 3×3
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        # Let's try to find a solution with specific structure
        # Assume each rank-1 term involves a subset of the 18 products
        
        rng = np.random.RandomState(123)
        
        # Try different initializations
        for trial in range(50):
            if time.time() > start_time + max_time:
                break
            
            U = np.zeros((6, 10), dtype=np.float64)
            V = np.zeros((9, 10), dtype=np.float64)
            W = np.zeros((6, 10), dtype=np.float64)
            
            # Each term c: choose which (i,j) pairs in U, (j,k) in V, (k,i) in W
            for c in range(10):
                # U: pick 2-3 (i,j) pairs
                u_pairs = rng.choice(6, size=rng.randint(2, 4), replace=False)
                for idx in u_pairs:
                    U[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0, 3.0, -3.0])
                
                v_pairs = rng.choice(9, size=rng.randint(2, 4), replace=False)
                for idx in v_pairs:
                    V[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0, 3.0, -3.0])
                
                w_pairs = rng.choice(6, size=rng.randint(2, 4), replace=False)
                for idx in w_pairs:
                    W[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0, 3.0, -3.0])
            
            # Gradient-like refinement
            for it in range(10000):
                if time.time() > start_time + max_time:
                    break
                
                errs = np.zeros((2, 3, 3))
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            errs[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i]) - 1.0
                
                max_err = np.max(np.abs(errs))
                if max_err < 1e-8:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    ok = True
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                val = np.sum(U_r[i*3+j] * V_r[j*3+k] * W_r[k*2+i])
                                if abs(val - 1) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        return U_r, V_r, W_r
                    break
                
                worst = np.unravel_index(np.argmax(np.abs(errs)), errs.shape)
                wi, wj, wk = worst
                u_row = wi * 3 + wj
                v_row = wj * 3 + wk
                w_row = wk * 2 + wi
                
                # Try multiple perturbations
                best_new_err = max_err
                best_change = None
                
                for _ in range(30):
                    mat = rng.randint(3)
                    col = rng.randint(10)
                    delta = rng.choice([-2.0, -1.0, 1.0, 2.0])
                    
                    if mat == 0:
                        old = U[u_row, col]
                        U[u_row, col] = old + delta
                    elif mat == 1:
                        old = V[v_row, col]
                        V[v_row, col] = old + delta
                    else:
                        old = W[w_row, col]
                        W[w_row, col] = old + delta
                    
                    errs_new = np.zeros((2, 3, 3))
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                errs_new[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i]) - 1.0
                    
                    new_max = np.max(np.abs(errs_new))
                    
                    if new_max < best_new_err:
                        best_new_err = new_max
                        best_change = (mat, u_row, v_row, w_row, col, delta, old)
                    
                    # Revert
                    if mat == 0:
                        U[u_row, col] = old
                    elif mat == 1:
                        V[v_row, col] = old
                    else:
                        W[w_row, col] = old
                
                if best_change is not None and best_new_err < max_err * 0.98:
                    mat, u_row, v_row, w_row, col, delta, old = best_change
                    if mat == 0:
                        U[u_row, col] += delta
                    elif mat == 1:
                        V[v_row, col] += delta
                    else:
                        W[w_row, col] += delta
                    max_err = best_new_err
                else:
                    # Broader random perturbation
                    mat = rng.randint(3)
                    row = rng.randint((6, 9, 6)[mat])
                    col = rng.randint(10)
                    delta = rng.choice([-1.0, 1.0])
                    if mat == 0:
                        U[row, col] += delta
                    elif mat == 1:
                        V[row, col] += delta
                    else:
                        W[row, col] += delta
        
        return None

    def find_2x3x3_decomposition_v3(max_time=30):
        """Try to find 2×3×3 rank-10 using a different strategy: 
        Start from schoolbook (rank 18) and try to merge terms."""
        if time.time() > start_time + max_time:
            return None
        
        # Schoolbook: 18 terms, one for each (i,j,k)
        # U[i*3+j, c] = 1 for the (i,j) pair, V[j*3+k, c] = 1 for (j,k), W[k*2+i, c] = 1 for (k,i)
        # We want to merge 18 terms into 10 terms
        
        # This is equivalent to finding a rank-10 decomposition
        # Let's try a direct numerical approach with more structure
        
        rng = np.random.RandomState(7)
        
        # Try: assume the solution has a certain sparsity pattern
        # Each term c has support on a subset of U rows, V rows, W rows
        
        for trial in range(30):
            if time.time() > start_time + max_time:
                break
            
            # Random sparse initialization with larger values
            U = np.zeros((6, 10), dtype=np.float64)
            V = np.zeros((9, 10), dtype=np.float64)
            W = np.zeros((6, 10), dtype=np.float64)
            
            for c in range(10):
                u_count = rng.randint(2, 5)
                u_idx = rng.choice(6, size=u_count, replace=False)
                for idx in u_idx:
                    U[idx, c] = rng.choice([-2.0, -1.0, 1.0, 2.0, 3.0, -3.0])
                
                v_count = rng.randint(2, 5)
                v_idx = rng.choice(9, size=v_count, replace=False)
                for idx in v_idx:
                    V[idx, c] = rng.choice([-2.0, -1.0, 1.0, 2.0, 3.0, -3.0])
                
                w_count = rng.randint(2, 5)
                w_idx = rng.choice(6, size=w_count, replace=False)
                for idx in w_idx:
                    W[idx, c] = rng.choice([-2.0, -1.0, 1.0, 2.0, 3.0, -3.0])
            
            # Simulated annealing style
            temperature = 1.0
            for it in range(20000):
                if time.time() > start_time + max_time:
                    break
                
                temperature *= 0.9999
                
                errs = np.zeros((2, 3, 3))
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            errs[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i]) - 1.0
                
                max_err = np.max(np.abs(errs))
                if max_err < 1e-8:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    ok = True
                    for i in range(2):
                        for j in range(3):
                            for k in range(3):
                                val = np.sum(U_r[i*3+j] * V_r[j*3+k] * W_r[k*2+i])
                                if abs(val - 1) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        return U_r, V_r, W_r
                    break
                
                # Pick a random variable to perturb
                mat = rng.randint(3)
                if mat == 0:
                    row = rng.randint(6)
                    col = rng.randint(10)
                    old = U[row, col]
                    U[row, col] = old + rng.choice([-2.0, -1.0, 1.0, 2.0])
                elif mat == 1:
                    row = rng.randint(9)
                    col = rng.randint(10)
                    old = V[row, col]
                    V[row, col] = old + rng.choice([-2.0, -1.0, 1.0, 2.0])
                else:
                    row = rng.randint(6)
                    col = rng.randint(10)
                    old = W[row, col]
                    W[row, col] = old + rng.choice([-2.0, -1.0, 1.0, 2.0])
                
                # Recompute error
                errs_new = np.zeros((2, 3, 3))
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            errs_new[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i]) - 1.0
                
                new_max = np.max(np.abs(errs_new))
                
                # Accept if better or with probability exp(-(new-old)/T)
                if new_max <= max_err:
                    max_err = new_max
                else:
                    if rng.random() < np.exp(-(new_max - max_err) / (temperature + 1e-10)):
                        max_err = new_max
                    else:
                        # Revert
                        if mat == 0:
                            U[row, col] = old
                        elif mat == 1:
                            V[row, col] = old
                        else:
                            W[row, col] = old
        
        return None

    # First, get the known rank-50 solution as fallback
    known_sol = current_solution()
    best_result = known_sol
    best_rank = 50
    
    # Try to find rank-10 decomposition of 2×3×3
    decomp_2x3x3 = None
    for finder in [find_2x3x3_decomposition_v2, find_2x3x3_decomposition, find_2x3x3_decomposition_v3]:
        if time.time() > start_time + 90:
            break
        decomp_2x3x3 = finder(max_time=25)
        if decomp_2x3x3 is not None:
            break
    
    if decomp_2x3x3 is not None:
        U_233, V_233, W_233 = decomp_2x3x3
        # Tile it 3 times for 2×3×9
        # For block q (q=0,1,2): columns 3q, 3q+1, 3q+2 of B
        # U is shared (same A matrix)
        # V: for block q, V[j*3+k, c] maps to V[j*9+(3q+k), c*3+q_offset]
        # W: for block q, W[k*2+i, c] maps to W[(3q+k)*2+i, c*3+q_offset]
        
        rank_233 = 10
        total_rank = 3 * rank_233  # 30
        
        U_full = np.zeros((6, total_rank), dtype=np.float64)
        V_full = np.zeros((27, total_rank), dtype=np.float64)
        W_full = np.zeros((18, total_rank), dtype=np.float64)
        
        for q in range(3):
            offset = q * rank_233
            # U is the same for all blocks (A is shared)
            U_full[:, offset:offset+rank_233] = U_233
            
            # V: block q uses rows j*9+(3q+k) for j,k in 0..2
            for j in range(3):
                for k in range(3):
                    V_full[j*9 + (3*q + k), offset:offset+rank_233] = V_233[j*3 + k]
            
            # W: block q uses rows (3q+k)*2+i for k,i
            for k in range(3):
                for i in range(2):
                    W_full[(3*q + k)*2 + i, offset:offset+rank_233] = W_233[k*2 + i]
        
        # Verify
        if verify_decomposition_np(U_full, V_full, W_full):
            # Round to half-integers
            U_r = np.round(U_full * 2) / 2
            V_r = np.round(V_full * 2) / 2
            W_r = np.round(W_full * 2) / 2
            
            if verify_decomposition_np(U_r, V_r, W_r):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = 30
    
    # If we didn't find rank 30, try to find rank 40, 45, etc.
    if best_rank == 50 and time.time() < start_time + 100:
        # Try to find a lower rank directly for 2×3×9
        for target_rank in [45, 40, 35]:
            if time.time() > start_time + 110:
                break
            
            rng = np.random.RandomState(target_rank)
            U = np.zeros((6, target_rank), dtype=np.float64)
            V = np.zeros((27, target_rank), dtype=np.float64)
            W = np.zeros((18, target_rank), dtype=np.float64)
            
            for c in range(target_rank):
                u_count = rng.randint(1, 4)
                u_idx = rng.choice(6, size=u_count, replace=False)
                for idx in u_idx:
                    U[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
                
                v_count = rng.randint(1, 4)
                v_idx = rng.choice(27, size=v_count, replace=False)
                for idx in v_idx:
                    V[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
                
                w_count = rng.randint(1, 4)
                w_idx = rng.choice(18, size=w_count, replace=False)
                for idx in w_idx:
                    W[idx, c] = rng.choice([-1.0, 1.0, 2.0, -2.0])
            
            # Local search
            for it in range(5000):
                if time.time() > start_time + 115:
                    break
                
                errs = compute_errors(U, V, W, target_rank)
                max_err = np.max(np.abs(errs))
                
                if max_err < 1e-10:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify_decomposition_np(U_r, V_r, W_r):
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = target_rank
                    break
                
                worst = np.unravel_index(np.argmax(np.abs(errs)), errs.shape)
                wi, wj, wk = worst
                u_row = wi * m + wj
                v_row = wj * p + wk
                w_row = wk * n + wi
                
                improved = False
                for _ in range(15):
                    mat = rng.randint(3)
                    col = rng.randint(target_rank)
                    if mat == 0:
                        old = U[u_row, col]
                        U[u_row, col] = old + rng.choice([-1.0, 1.0])
                    elif mat == 1:
                        old = V[v_row, col]
                        V[v_row, col] = old + rng.choice([-1.0, 1.0])
                    else:
                        old = W[w_row, col]
                        W[w_row, col] = old + rng.choice([-1.0, 1.0])
                    
                    errs_new = compute_errors(U, V, W, target_rank)
                    new_max = np.max(np.abs(errs_new))
                    
                    if new_max < max_err * 0.95:
                        max_err = new_max
                        improved = True
                        break
                    else:
                        if mat == 0:
                            U[u_row, col] = old
                        elif mat == 1:
                            V[v_row, col] = old
                        else:
                            W[w_row, col] = old
                
                if not improved:
                    mat = rng.randint(3)
                    row = rng.randint((6, 27, 18)[mat])
                    col = rng.randint(target_rank)
                    if mat == 0:
                        U[row, col] += rng.choice([-1.0, 1.0])
                    elif mat == 1:
                        V[row, col] += rng.choice([-1.0, 1.0])
                    else:
                        W[row, col] += rng.choice([-1.0, 1.0])
            
            if best_rank < 50:
                break
    
    return best_result
