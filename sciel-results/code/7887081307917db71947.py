"""Combined program: structural rank-32 base with local search for rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    # Build the rank-32 base decomposition using Strassen blocks
    # Block 1: cols 0-3 -> two Strassen-7 blocks (rank 14)
    # Block 2: cols 4-8 -> Strassen(4,5) + schoolbook(6) + Strassen(7,8) (rank 7+4+7=18)
    # Total: 32

    rank_base = 32
    U = [[0.0] * rank_base for _ in range(n * m)]   # 4 x 32
    V = [[0.0] * rank_base for _ in range(m * p)]   # 18 x 32
    W = [[0.0] * rank_base for _ in range(p * n)]   # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b00+b11) -> C00, C11
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M2: (a10+a11)b00 -> C10, -C11
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        # M3: a00(b01-b11) -> C01, C11
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M4: a11(b10-b00) -> C00, C10
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        # M5: (a00+a01)b11 -> -C00, C01
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        # M6: (a10-a00)(b00+b01) -> C11
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7: (a01-a11)(b10+b11) -> C00
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_trivial(6)
    add_strassen(7, 8)

    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        """Check tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_arr.shape[1]):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr, tol=1e-8):
        """Vectorized verification."""
        r = U_arr.shape[1]
        # Build tensor via einsum
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        # Check against ones tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1.0) > tol:
                        return False
        return True

    # Local search to reduce rank by 1
    # We try to drop one term and refine the others using a simple iterative method.
    
    def try_drop_term(drop_idx):
        if time.time() > deadline - 10:
            return None
        
        # Create copies of U, V, W with the drop_idx column removed
        keep = [i for i in range(rank_base) if i != drop_idx]
        U_work = U_np[:, keep].copy()
        V_work = V_np[:, keep].copy()
        W_work = W_np[:, keep].copy()
        
        # Target tensor
        T = np.ones((n, m, p))
        
        # Iterative refinement: fix two matrices, solve for the third
        # We'll use a simple approach: 
        # 1. Fix V and W, solve for U via least squares
        # 2. Fix U and W, solve for V via least squares
        # 3. Fix U and V, solve for W via least squares
        # Repeat a few times.
        
        # To solve for U given V, W:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For a fixed (i,k), let u_vec = U[i*m:(i*m+m), :] (size m x R)
        # This is coupled. Let's vectorize.
        # Reshape T to (n*m*p, 1)
        T_vec = T.reshape(-1)
        
        # Construct the linear system for U.
        # The coefficient for U[i*m+j, r] in the equation for T[i,j,k] is V[j*p+k, r] * W[k*n+i, r]
        # This is a 3D structure. We can build a large matrix A_U of size (n*m*p) x (n*m*R)
        # But n*m*p = 36 and n*m*R = 4*31 = 124. This is small enough.
        
        def solve_for_U(V_mat, W_mat):
            # A_U shape: (36, 124)
            A_U = np.zeros((n*m*p, n*m*len(keep)))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for r_idx, r in enumerate(keep):
                            idx_u = (i*m + j) * len(keep) + r_idx
                            A_U[idx_t, idx_u] = V_mat[j*p+k, r_idx] * W_mat[k*n+i, r_idx]
            
            try:
                U_sol, _, _, _ = np.linalg.lstsq(A_U, T_vec, rcond=None)
            except:
                return None
            return U_sol.reshape(n*m, len(keep))
        
        def solve_for_V(U_mat, W_mat):
            # A_V shape: (36, m*p*R) = (36, 18*31=558)
            A_V = np.zeros((n*m*p, m*p*len(keep)))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for r_idx, r in enumerate(keep):
                            idx_v = (j*p + k) * len(keep) + r_idx
                            A_V[idx_t, idx_v] = U_mat[i*m+j, r_idx] * W_mat[k*n+i, r_idx]
            
            try:
                V_sol, _, _, _ = np.linalg.lstsq(A_V, T_vec, rcond=None)
            except:
                return None
            return V_sol.reshape(m*p, len(keep))
        
        def solve_for_W(U_mat, V_mat):
            # A_W shape: (36, p*n*R) = (36, 18*31=558)
            A_W = np.zeros((n*m*p, p*n*len(keep)))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for r_idx, r in enumerate(keep):
                            idx_w = (k*n + i) * len(keep) + r_idx
                            A_W[idx_t, idx_w] = U_mat[i*m+j, r_idx] * V_mat[j*p+k, r_idx]
            
            try:
                W_sol, _, _, _ = np.linalg.lstsq(A_W, T_vec, rcond=None)
            except:
                return None
            return W_sol.reshape(p*n, len(keep))
        
        # Run a few iterations
        for _ in range(10):
            if time.time() > deadline - 10:
                return None
            U_work = solve_for_U(V_work, W_work)
            if U_work is None: continue
            V_work = solve_for_V(U_work, W_work)
            if V_work is None: continue
            W_work = solve_for_W(U_work, V_work)
            if W_work is None: continue
            
        # Verify the result
        if verify_fast(U_work, V_work, W_work, tol=1e-6):
            # Convert back to lists
            U_list = U_work.tolist()
            V_list = V_work.tolist()
            W_list = W_work.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    for r in range(rank_base):
        if time.time() > deadline - 5:
            break
        result = try_drop_term(r)
        if result:
            return result

    # Return the base decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
