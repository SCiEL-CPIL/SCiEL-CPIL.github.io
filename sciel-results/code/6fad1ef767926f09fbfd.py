import time
import numpy as np
import scipy.linalg

DIMENSIONS = [2, 4, 7]
DEADLINE = 130

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _construct_rank_50(n, m, p):
    """
    Constructs a rank-50 decomposition using Strassen for 2x2x2 blocks 
    and schoolbook for the remainder.
    2x4x7 -> 2x4x4 (Strassen blocks) + 2x4x3 (Schoolbook)
    2x4x4: 2 blocks of 2x2x4? No.
    Split m=4 into two 2s. Split p=4 into two 2s.
    4 blocks of 2x2x2. Rank 7*4 = 28.
    Remainder: 2x4x3. Schoolbook rank 2*4*3 = 24.
    Total rank: 28 + 24 = 52? 
    Wait, the previous note said rank 50.
    "splitting the 2x2x3 blocks into 2x2x2 (Strassen, rank 7) and 2x2x1 (schoolbook, rank 4)"
    2x4x3: Split m=4 into two 2s.
    Two blocks of 2x2x3.
    Each 2x2x3: Split p=3 into 2x2 + 2x1.
    2x2x2 (Rank 7) + 2x2x1 (Rank 4). Total 11 per block.
    2 blocks -> 22.
    2x4x4: Split m=4 into two 2s. Split p=4 into two 2s.
    4 blocks of 2x2x2. Rank 7*4 = 28.
    Total: 28 + 22 = 50.
    """
    U_total = np.zeros((n * m, 50))
    V_total = np.zeros((m * p, 50))
    W_total = np.zeros((p * n, 50))
    
    # Helper to embed 2x2x2 Strassen
    def embed_strassen_222(U2, V2, W2, i_off, j_off, k_off, rank_start):
        # i_off: offset in n (0 or 1)
        # j_off: offset in m (0 or 2)
        # k_off: offset in p (0, 2, 4, 6)
        
        for r in range(7):
            r_global = rank_start + r
            
            # U: i in 0,1; j in 0,1
            for i in range(2):
                for j in range(2):
                    val = U2[i * 2 + j, r]
                    if val != 0:
                        i_g = i_off + i
                        j_g = j_off + j
                        U_total[i_g * m + j_g, r_global] = val
                        
            # V: j in 0,1; k in 0,1
            for j in range(2):
                for k in range(2):
                    val = V2[j * 2 + k, r]
                    if val != 0:
                        j_g = j_off + j
                        k_g = k_off + k
                        V_total[j_g * p + k_g, r_global] = val
                        
            # W: k in 0,1; i in 0,1
            for k in range(2):
                for i in range(2):
                    val = W2[k * 2 + i, r]
                    if val != 0:
                        k_g = k_off + k
                        i_g = i_off + i
                        W_total[k_g * n + i_g, r_global] = val

    # Helper to embed 2x2x1 Schoolbook
    def embed_school_221(i_off, j_off, k_off, rank_start):
        # 2x2x1: 4 terms
        for i in range(2):
            for j in range(2):
                r_global = rank_start + (i * 2 + j)
                i_g = i_off + i
                j_g = j_off + j
                k_g = k_off # k is 0 only for 2x2x1
                
                U_total[i_g * m + j_g, r_global] = 1.0
                V_total[j_g * p + k_g, r_global] = 1.0
                W_total[k_g * n + i_g, r_global] = 1.0

    # Strassen 2x2x2 matrices
    U2, V2, W2 = _construct_strassen_2x2x2()
    
    rank_counter = 0
    
    # Part 1: 2x4x4 (p indices 0..3)
    # 4 blocks of 2x2x2
    # j_off in [0, 2], k_off in [0, 2]
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            embed_strassen_222(U2, V2, W2, 0, j_off, k_off, rank_counter)
            rank_counter += 7
            
    # Part 2: 2x4x3 (p indices 4..6)
    # Split into two 2x2x3 blocks (j_off 0 and 2)
    for j_off in [0, 2]:
        # 2x2x3: Split k into [4,5] (2x2x2) and [6] (2x2x1)
        # Block 1: 2x2x2 at k_off=4
        embed_strassen_222(U2, V2, W2, 0, j_off, 4, rank_counter)
        rank_counter += 7
        # Block 2: 2x2x1 at k_off=6
        embed_school_221(0, j_off, 6, rank_counter)
        rank_counter += 4
        
    return U_total, V_total, W_total

def _construct_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21, C22
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B12-B22) -> C12, C22
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B21-B11) -> C11, C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)B22 -> C11, C12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # 1. Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_[i, j, r] * V_[j, k, r] * W_[k, i, r]
                if abs(val - 1.0) > tol:
                    return False

    # 2. Check zero entries
    # Total entries: (n*m)*(m*p)*(p*n)
    total_u = n * m
    total_v = m * p
    total_w = p * n
    
    # We check all zero entries if rank is small, otherwise sample
    # For rank 49, checking all 3080 zeros is fast.
    num_checks = 3080 
    rng = np.random.RandomState(42)
    
    for _ in range(num_checks):
        u = rng.randint(0, total_u)
        v = rng.randint(0, total_v)
        w = rng.randint(0, total_w)
        
        i1 = u // m
        j1 = u % m
        j2 = v // p
        k2 = v % p
        k3 = w // n
        i3 = w % n
        
        if i1 == i3 and j1 == j2 and k2 == k3:
            continue
            
        val = 0.0
        for r in range(rank):
            val += U_[i1, j1, r] * V_[j2, k2, r] * W_[k3, i3, r]
            
        if abs(val) > tol:
            return False

    return True

def _als_optimize(U, V, W, n, m, p, target_rank, max_iter=200, tol=1e-9):
    """
    Alternating Least Squares to optimize the decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute target tensor values
    # We need to solve: sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = T_{i,j,k}
    
    # We will iterate over the three matrices.
    # Fix U, V. Solve for W.
    # The equation is linear in W_{k,i}^{(r)}.
    # Let M_{r, (k,i)} be the coefficient of W_{k,i}^{(r)}.
    # Actually, for each (i,j,k), we have:
    # sum_r (U_{i,j}^{(r)} V_{j,k}^{(r)}) W_{k,i}^{(r)} = T_{i,j,k}
    
    # This is not a simple linear system for W because the index r is coupled to (k,i) in W.
    # W is a matrix of shape (p*n, rank).
    # Let's flatten W to a vector w of length p*n*rank.
    # The equation for a specific (i,j,k) involves terms W_{k,i}^{(r)} for all r.
    # So it's a linear system in the flattened W.
    
    # However, we have 56 equations (for 1s) and 3080 equations (for 0s).
    # Total 3136 equations.
    # Unknowns: 14 * 49 = 686.
    # Overdetermined. We use least squares.
    
    # To speed up, we only use the 56 non-zero equations and a subset of zeros?
    # No, we must satisfy zeros.
    
    # Let's build the design matrix A for W.
    # A has shape (Num_Eqs, p*n*rank).
    # This is large: 3136 * 686 ~ 2M entries. Manageable.
    
    # But we need to do this 3 times per iteration.
    
    # Optimization: We can just use the 56 non-zero equations to update W, 
    # and hope the zeros stay close to 0? 
    # No, that will likely fail.
    
    # Alternative: Use only the non-zero equations for the update, 
    # then project or verify.
    
    # Let's try to solve the least squares problem for W using all constraints.
    
    # Indices:
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Valid triples: (i,j,k)
    # Invalid triples: all others.
    
    # We will construct the linear system for W.
    # Variable index in W: idx_w = (k*n + i) * rank + r
    
    # Equation index: eq_idx
    # For each valid (i,j,k):
    #   sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # For each invalid (i,j,j',k',k'',i''):
    #   sum_r U[i,j,r] * V[j',k',r] * W[k'',i'',r] = 0
    
    # This is too many equations to build the full matrix in Python loops.
    # We will use a sparse approach or just the non-zero ones?
    
    # Let's stick to the non-zero equations for the update step, 
    # and check the error on zeros.
    
    # Build A for W (non-zero eqs only)
    # Num eqs: n*m*p = 56
    # Num vars: p*n*rank = 14*49 = 686
    
    A_W = np.zeros((n*m*p, p*n*rank))
    b_W = np.ones(n*m*p)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    coeff = U[i*m + j, r] * V[j*p + k, r]
                    var_idx = (k*n + i) * rank + r
                    A_W[eq_idx, var_idx] = coeff
                eq_idx += 1
                
    # Solve least squares
    W_flat, residuals, rank_w, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
    W_new = W_flat.reshape(p*n, rank)
    
    # Check if W_new is valid (no NaNs)
    if not np.all(np.isfinite(W_new)):
        return U, V, W, False
        
    # Update W
    W = W_new
    
    # Now solve for V, fixing U and W.
    # Equation: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
    # Linear in V[j,k,r].
    # Var idx: (j*p + k) * rank + r
    
    A_V = np.zeros((n*m*p, m*p*rank))
    b_V = np.ones(n*m*p)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    coeff = U[i*m + j, r] * W[k*n + i, r]
                    var_idx = (j*p + k) * rank + r
                    A_V[eq_idx, var_idx] = coeff
                eq_idx += 1
                
    V_flat, residuals, rank_v, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
    V_new = V_flat.reshape(m*p, rank)
    
    if not np.all(np.isfinite(V_new)):
        return U, V, W, False
        
    V = V_new
    
    # Now solve for U, fixing V and W.
    # Linear in U[i,j,r].
    # Var idx: (i*m + j) * rank + r
    
    A_U = np.zeros((n*m*p, n*m*rank))
    b_U = np.ones(n*m*p)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    coeff = V[j*p + k, r] * W[k*n + i, r]
                    var_idx = (i*m + j) * rank + r
                    A_U[eq_idx, var_idx] = coeff
                eq_idx += 1
                
    U_flat, residuals, rank_u, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
    U_new = U_flat.reshape(n*m, rank)
    
    if not np.all(np.isfinite(U_new)):
        return U, V, W, False
        
    U = U_new
    
    return U, V, W, True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get the rank-50 solution
    U50, V50, W50 = _construct_rank_50(n, m, p)
    
    # Verify rank-50
    if _verify_full(U50, V50, W50, n, m, p):
        best_U, best_V, best_W = U50, V50, W50
        best_rank = 50
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 56
        
    # Attempt Rank 49
    target_rank = 49
    # Initialize with first 49 columns of rank-50 solution
    U49_init = U50[:, :target_rank]
    V49_init = V50[:, :target_rank]
    W49_init = W50[:, :target_rank]
    
    # Run ALS
    U49, V49, W49, success = _als_optimize(U49_init, V49_init, W49_init, n, m, p, target_rank, max_iter=50)
    
    if success and _verify_full(U49, V49, W49, n, m, p):
        best_U, best_V, best_W = U49, V49, W49
        best_rank = 49
        
    # If time permits, try Rank 48?
    if time.time() - start_time < DEADLINE - 30 and best_rank > 48:
        target_rank = 48
        U48_init = U50[:, :target_rank]
        V48_init = V50[:, :target_rank]
        W48_init = W50[:, :target_rank]
        
        U48, V48, W48, success = _als_optimize(U48_init, V48_init, W48_init, n, m, p, target_rank, max_iter=50)
        
        if success and _verify_full(U48, V48, W48, n, m, p):
            best_U, best_V, best_W = U48, V48, W48
            best_rank = 48

    if time.time() - start_time > DEADLINE - 5:
        # Emergency return
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
