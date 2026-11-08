import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4
    
    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off + 1 * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def find_rank_with_fixed_V(R, max_attempts=10, timeout=25):
        """
        Fix V to a canonical sparse form (one non-zero per row, values in {-1, 1}),
        then solve for U and W using alternating least squares with snapping.
        """
        if time.time() > deadline - timeout:
            return None
        
        best_result = None
        
        for attempt in range(max_attempts):
            if time.time() > deadline - timeout:
                break
            
            np.random.seed(123 + attempt * 7)
            
            # Create V with exactly one non-zero per row
            V = np.zeros((V_dim, R))
            # Assign each row to a random column
            row_assignments = np.random.randint(0, R, size=V_dim)
            for r in range(V_dim):
                col = row_assignments[r]
                V[r, col] = 1.0 if np.random.random() < 0.5 else -1.0
            
            # Initialize U and W
            U = np.random.randn(U_dim, R) * 0.1
            W = np.random.randn(W_dim, R) * 0.1
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline - timeout:
                    break
                
                # Solve for U given V and W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each a: T[a,:,:].flatten() = M_a @ U[a,:]
                # where M_a[b*W_dim+c, r] = V[b,r] * W[c,r]
                
                U_new = np.zeros((U_dim, R))
                for a in range(U_dim):
                    M_a = np.zeros((V_dim * W_dim, R))
                    for b in range(V_dim):
                        for c in range(W_dim):
                            M_a[b * W_dim + c, :] = V[b, :] * W[c, :]
                    try:
                        U_new[a, :] = np.linalg.lstsq(M_a, T[a, :, :].flatten(), rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                
                # Solve for W given U and V
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each c: T[:,:,c].flatten() = M_c @ W[c,:]
                # where M_c[a*V_dim+b, r] = U[a,r] * V[b,r]
                
                W_new = np.zeros((W_dim, R))
                for c in range(W_dim):
                    M_c = np.zeros((U_dim * V_dim, R))
                    for a in range(U_dim):
                        for b in range(V_dim):
                            M_c[a * V_dim + b, :] = U[a, :] * V[b, :]
                    try:
                        W_new[c, :] = np.linalg.lstsq(M_c, T[:, :, c].flatten(), rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                
                diff_U = np.linalg.norm(U_new - U)
                diff_W = np.linalg.norm(W_new - W)
                U = U_new
                W = W_new
                
                if diff_U + diff_W < 1e-9:
                    break
            
            # Check loss
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            loss = np.sum((prod - T) ** 2)
            
            if loss < 1e-4:
                # Try to snap
                valid, U_s, V_s, W_s = try_snap_validate(U, V, W)
                if valid:
                    best_result = (U_s, V_s, W_s)
                    break
                else:
                    # Try more aggressive snapping
                    for snap_factor in [4, 8, 16]:
                        U_snap = np.round(U * snap_factor) / snap_factor
                        V_snap = np.round(V * snap_factor) / snap_factor
                        W_snap = np.round(W * snap_factor) / snap_factor
                        if validate(U_snap, V_snap, W_snap):
                            best_result = (U_snap, V_snap, W_snap)
                            break
                    if best_result:
                        break
        
        return best_result

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    
    # Try to find lower rank using fixed-V approach
    for target_rank in [45, 44, 43, 42]:
        if time.time() > deadline - 20:
            break
        
        result = find_rank_with_fixed_V(target_rank, max_attempts=8, timeout=18)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
