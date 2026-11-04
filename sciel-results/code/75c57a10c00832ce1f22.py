"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from scipy.optimize import minimize

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def build_2x2x3_rank11():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3 (4 schoolbook + 7 Strassen)."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # Schoolbook part: 4 terms for the first column of B
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[3][3] = 1; W[3][3] = 1
        
        # Strassen part for columns 1,2 of B (mapping 2x2x2 to positions)
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}  # V row mapping: 0->1, 1->2, 2->4, 3->5
        w2_map = {0: 2, 1: 4, 2: 3, 3: 5}  # W row mapping: 0->2, 1->4, 2->3, 3->5
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base + c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0:
                    V[v2_map[r]][base + c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0:
                    W[w2_map[r]][base + c] += W2[r][c]

        return U, V, W

    def tensor_error_2x2x3(U, V, W):
        """Compute the error of a 2x2x3 decomposition. U:(4,r), V:(6,r), W:(6,r)
        W indexed by (k,i) where k in 0..2, i in 0..1 -> index = k*2+i"""
        U_arr = np.asarray(U, dtype=float)
        V_arr = np.asarray(V, dtype=float)
        W_arr = np.asarray(W, dtype=float)
        r = U_arr.shape[1]
        errors = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_row = i * 2 + j
                    v_row = j * 3 + k
                    w_row = k * 2 + i
                    val = np.sum(U_arr[u_row] * V_arr[v_row] * W_arr[w_row])
                    errors.append(val - 1.0)
        return np.array(errors)

    def polish_2x2x3(U0, V0, W0, max_iters=200, tol=1e-10):
        """Polish a near-valid 2x2x3 decomposition using gradient descent."""
        U = np.array(U0, dtype=float)
        V = np.array(V0, dtype=float)
        W = np.array(W0, dtype=float)
        r = U.shape[1]
        
        def objective(x):
            U_f = x[:4*r].reshape(4, r)
            V_f = x[4*r:4*r+6*r].reshape(6, r)
            W_f = x[4*r+6*r:].reshape(6, r)
            errs = tensor_error_2x2x3(U_f, V_f, W_f)
            return np.sum(errs**2)
        
        def gradient(x):
            U_f = x[:4*r].reshape(4, r)
            V_f = x[4*r:4*r+6*r].reshape(6, r)
            W_f = x[4*r+6*r:].reshape(6, r)
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            f0 = objective(x)
            for idx in range(len(x)):
                x_pert = x.copy()
                x_pert[idx] += eps
                grad[idx] = (objective(x_pert) - f0) / eps
                x_pert[idx] -= 2*eps
                grad[idx] = (f0 - objective(x_pert)) / eps
            return grad
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B
        result = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                         options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'maxfun': 5000})
        
        if result.fun < 1e-8:
            U_out = result.x[:4*r].reshape(4, r)
            V_out = result.x[4*r:4*r+6*r].reshape(6, r)
            W_out = result.x[4*r+6*r:].reshape(6, r)
            # Round to nearest half-integer
            U_out = np.round(U_out * 2) / 2
            V_out = np.round(V_out * 2) / 2
            W_out = np.round(W_out * 2) / 2
            if np.allclose(tensor_error_2x2x3(U_out, V_out, W_out), 0, atol=1e-9):
                return U_out.tolist(), V_out.tolist(), W_out.tolist()
            # Try without rounding
            if np.allclose(tensor_error_2x2x3(U_out, V_out, W_out), 0, atol=1e-9):
                return U_out.tolist(), V_out.tolist(), W_out.tolist()
        return None

    def try_find_2x2x3_lower_rank(max_rank=10):
        """Try to find a lower-rank decomposition for 2x2x3 using structured search + polish."""
        if time.time() > deadline - 30:
            return None, None, None, max_rank + 1
        
        random.seed(42)
        np.random.seed(42)
        
        # Try ranks from 8 to max_rank
        for rank in range(8, max_rank + 1):
            if time.time() > deadline - 10:
                break
            
            # Multiple random restarts with polish
            for restart in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Initialize with structured patterns
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(4, rank))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(6, rank))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(6, rank))
                
                # Check if already valid
                if np.allclose(tensor_error_2x2x3(U, V, W), 0, atol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist(), rank
                
                # Try to polish
                result = polish_2x2x3(U, V, W, max_iters=50)
                if result is not None:
                    return result[0], result[1], result[2], rank
        
        return None, None, None, max_rank + 1

    def build_4x4x6(U_sub, V_sub, W_sub):
        """Build 4x4x6 from 2x2x3 sub-decomposition using Strassen block recursion."""
        n, m, p = 4, 4, 6
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        # Strassen-like 7-term decomposition for 4x4*4x6 using 2x2x3 blocks
        # We partition: A is 4x4 into 2x2 blocks, B is 4x6 into 2x3 blocks
        # Block indices: A[bi,bj], B[bj,bk], C[bi,bk] where bi,bj in {0,1}, bk in {0,1,2}
        # But we use 7 Strassen terms for the 2x2x2 part and handle the extra column separately
        
        # Actually, let's use a cleaner approach:
        # C = A * B where A is 4x4, B is 4x6
        # Partition A into [[A00, A01], [A10, A11]] (each 2x2)
        # Partition B into [[B00, B01, B02], [B10, B11, B12]] (each 2x3)
        # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12],
        #      [A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12]]
        
        # Each C block is 2x3, computed as sum of two 2x2*2x3 products
        # We can use 7 Strassen terms for the 2x2x2 part and 4 schoolbook for the third column
        # But that gives 7*2 + 4*2 = 20 per row of blocks... too much
        
        # Better: use the 2x2x3 sub-decomposition directly
        # For each of the 4 block-pair combinations, we need one 2x2x3 computation
        # But Strassen reduces the 2x2x2 part from 8 to 7 multiplications
        
        # Let me use a different approach: direct block decomposition
        # C00 = A00*B00 + A01*B10 (two 2x2x3 products)
        # C01 = A00*B01 + A01*B11
        # C02 = A00*B02 + A01*B12
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # C12 = A10*B02 + A11*B12
        
        # That's 12 separate 2x2x3 computations, each of rank rank_sub
        # Total rank = 12 * rank_sub. For rank_sub=11, that's 132. Worse than current.
        
        # The current approach uses 7 terms which is better. Let me keep it.
        
        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
        w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff

                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff

                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    def verify(U, V, W, n, m, p):
        """Fast verification using NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Build the tensor check: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # Vectorized: create all index combinations
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        
        # Compute all dot products at once
        # prod[r] = sum_r U[u_rows, r] * V[v_rows, r] * W[w_rows, r]
        # This is: (U[u_rows] * V[v_rows] * W[w_rows]).sum(axis=1)
        prod = (U_arr[u_rows] * V_arr[v_rows] * W_arr[w_rows]).sum(axis=1)
        
        return np.allclose(prod, 1.0, atol=1e-9)

    # First try to find a lower-rank 2x2x3 decomposition
    U_sub, V_sub, W_sub, sub_rank = try_find_2x2x3_lower_rank(max_rank=10)
    
    if U_sub is None:
        # Fall back to rank-11
        U_sub, V_sub, W_sub = build_2x2x3_rank11()
        sub_rank = 11

    U, V, W = build_4x4x6(U_sub, V_sub, W_sub)

    # Verify the decomposition
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
