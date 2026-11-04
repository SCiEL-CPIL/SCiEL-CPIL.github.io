import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def _construct_strassen_28():
    """
    Returns the rank 28 Strassen-based decomposition.
    """
    n, m, p = 2, 4, 4
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen patterns for 2x2
    # a, b, c dicts mapping local index (r*2+c) to coeff
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] x B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] x B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] x B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] x B[2:4,2:4] -> C[:,2:4]
    ]
    
    col_idx = 0
    for (a_cs, b_rs, b_cs, c_cs) in blocks:
        for pattern in patterns:
            for local_idx, coeff in pattern['a'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                U[local_r * m + (a_cs + local_c), col_idx] = coeff
            for local_idx, coeff in pattern['b'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                V[(b_rs + local_r) * p + (b_cs + local_c), col_idx] = coeff
            for local_idx, coeff in pattern['c'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                W[(c_cs + local_c) * n + local_r, col_idx] = coeff
            col_idx += 1
            
    return U, V, W

def _construct_strassen_24():
    """
    Returns a rank 24 decomposition using Strassen for the 2x2 diagonal blocks
    and sharing the off-diagonal cross terms.
    """
    n, m, p = 2, 4, 4
    R = 24
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen patterns for 2x2
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    # Diagonal blocks: (0,0) and (2,2)
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    diag_blocks = [
        (0, 0, 0, 0),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for (a_cs, b_rs, b_cs, c_cs) in diag_blocks:
        for pattern in patterns:
            for local_idx, coeff in pattern['a'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                U[local_r * m + (a_cs + local_c), col_idx] = coeff
            for local_idx, coeff in pattern['b'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                V[(b_rs + local_r) * p + (b_cs + local_c), col_idx] = coeff
            for local_idx, coeff in pattern['c'].items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                W[(c_cs + local_c) * n + local_r, col_idx] = coeff
            col_idx += 1
            
    # Off-diagonal blocks:
    # C[:, 0:2] += A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] += A[:, 0:2] * B[0:2, 2:4]
    
    # We need to compute these with rank 4 total (2 for each) to keep total rank 24.
    # Wait, 14 (diag) + 4 (off) = 18? No.
    # 2x4x4 rank is at least 2*4 = 8? No.
    # Rank of 2x4x4 matmul is known to be 24? Or 26?
    # Let's just try to fill the remaining 10 slots with a valid decomposition for the off-diagonals.
    
    # Actually, the standard "Strassen for 2x4x4" might not exist at 24.
    # Let's stick to the known 28 or try to find 27/26 via search.
    # This function is a placeholder for a potential 24 solution if found.
    # For now, return None to indicate failure, or a dummy that fails verification.
    # The caller will verify.
    
    # Let's try a simple construction for the off-diagonals to see if it works.
    # C[i, 0:2] += sum_{j=2,3} A[i, j] B[j, 0:2]
    # This is a 2x2x2 matmul for each column block?
    # A_off is 2x2, B_off is 2x2.
    # We can use rank 4 for each? That would be 14+8=22?
    # No, we only have 24-14=10 slots.
    # So we need rank 5 for each off-diagonal block?
    # Rank of 2x2x2 is 7 (Strassen). So we can't do it in 5.
    # So rank 24 is likely impossible with this block structure.
    
    # Let's just return None.
    return None

def _solve_224_rank(R, max_iters=1000, timeout=10):
    """
    Attempts to find a rank R decomposition for 2x2x4 matrix multiplication.
    Returns (U, V, W) if found, else None.
    U: (4, R), V: (8, R), W: (8, R)
    """
    n, m, p = 2, 2, 4
    # T_{ijk} = 1
    T = np.ones((n, m, p))
    
    # Variables: flatten U, V, W
    # U: 2x2, V: 2x4, W: 4x2
    # Total vars per term: 4 + 8 + 8 = 20
    # Total vars: 20 * R
    
    def func(x):
        U_flat = x[:4*R]
        V_flat = x[4*R:12*R]
        W_flat = x[12*R:]
        
        U = U_flat.reshape(4, R)
        V = V_flat.reshape(8, R)
        W = W_flat.reshape(8, R)
        
        # Reshape for einsum
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        return (T_recon - T).flatten()
    
    # Initial guess: Random
    best_residual = np.inf
    best_x = None
    
    # Use a few random restarts
    for _ in range(5):
        x0 = np.random.randn(20 * R)
        try:
            res = least_squares(func, x0, method='trf', max_nfev=1000)
            if res.cost < best_residual:
                best_residual = res.cost
                best_x = res.x
                if best_residual < 1e-12:
                    break
        except Exception:
            continue
            
    if best_x is None or best_residual > 1e-9:
        return None
        
    # Check if solution is close to integers
    U = best_x[:4*R].reshape(4, R)
    V = best_x[4*R:12*R].reshape(8, R)
    W = best_x[12*R:].reshape(8, R)
    
    # Snap to nearest integer
    U_int = np.round(U).astype(int)
    V_int = np.round(V).astype(int)
    W_int = np.round(W).astype(int)
    
    # Verify integer solution
    U_r = U_int.reshape(n, m, R)
    V_r = V_int.reshape(m, p, R)
    W_r = W_int.reshape(p, n, R)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if np.allclose(T_recon, T, atol=1e-9):
        return U_int, V_int, W_int
        
    # If not integer, try half-integers?
    # Multiply by 2 and check
    U_half = np.round(U * 2).astype(int)
    V_half = np.round(V * 2).astype(int)
    W_half = np.round(W * 2).astype(int)
    
    U_r = U_half.reshape(n, m, R)
    V_r = V_half.reshape(m, p, R)
    W_r = W_half.reshape(p, n, R)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # If we used half integers, the product is scaled by 8?
    # No, the sum is $\sum u v w$. If u,v,w are doubled, sum is 8x.
    # We need T_recon to be 1.
    # So if we find a solution with half-integers, the sum will be 8.
    # We need to divide the result? No, the interface requires the matrices themselves.
    # If the matrices are half-integers, the product is 8x the target.
    # This is invalid unless we can scale the whole thing?
    # The task says "integer/half-integer entries".
    # If entries are half-integers, let $U = U'/2$.
    # Then $\sum (U'/2) (V'/2) (W'/2) = \frac{1}{8} \sum U' V' W'$.
    # We need this to be 1. So $\sum U' V' W' = 8$.
    # So we check if T_recon (using U', V', W') is 8.
    
    if np.allclose(T_recon, 8 * np.ones((n, m, p)), atol=1e-9):
        return U_half / 2.0, V_half / 2.0, W_half / 2.0
        
    return None

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """
    Verifies if U, V, W correctly decompose the n x m x p matrix multiplication tensor.
    """
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_true = np.ones((n, m, p))
    return np.allclose(T_recon, T_true, atol=1e-9)

def construct():
    t0 = time.time()
    deadline = 145.0
    
    # Get the baseline rank 28 solution
    U28, V28, W28 = _construct_strassen_28()
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try to find a lower rank solution for the sub-problem 2x2x4
    # If we find rank 13 for 2x2x4, we can build rank 26 for 2x4x4.
    # The 2x4x4 problem splits into two 2x2x4 problems.
    
    # Try ranks 13, 12, 11 for 2x2x4
    for r in [13, 12, 11]:
        if time.time() - t0 > deadline - 20:
            break
            
        # We need to solve for 2x2x4
        # U: 4xr, V: 8xr, W: 8xr
        sol = _solve_224_rank(r, timeout=10)
        
        if sol is not None:
            U_sub, V_sub, W_sub = sol
            # Construct full 2x4x4 solution
            # Left part (cols 0-1 of B, 0-1 of C) and Right part (cols 2-3 of B, 2-3 of C)
            # A is split into A_L (cols 0-1) and A_R (cols 2-3)
            
            U_full = np.zeros((8, 2*r), dtype=float)
            V_full = np.zeros((16, 2*r), dtype=float)
            W_full = np.zeros((8, 2*r), dtype=float)
            
            # Left half (indices 0,1 for A cols, 0,1 for B rows, 0,1 for C cols)
            # A_L is rows 0,1 of A. In U_full, these are rows 0,1 (i=0, j=0) and 2,3 (i=1, j=0)?
            # U index: i*m + j. m=4.
            # A_L corresponds to j=0,1.
            # So rows 0,1 (i=0, j=0,1) and 4,5 (i=1, j=0,1) in U_full?
            # Wait, A is 2x4.
            # A[0,0], A[0,1] -> U[0, :], U[1, :]
            # A[1,0], A[1,1] -> U[4, :], U[5, :]
            
            # B_L is rows 0,1 of B.
            # B[0,0], B[0,1] -> V[0, :], V[1, :]
            # B[1,0], B[1,1] -> V[4, :], V[5, :]
            
            # C_L is cols 0,1 of C.
            # C[0,0], C[0,1] -> W[0, :], W[1, :]
            # C[1,0], C[1,1] -> W[4, :], W[5, :]
            
            # Map U_sub (4xr) to U_full
            # U_sub rows: 0 (i=0,j=0), 1 (i=0,j=1), 2 (i=1,j=0), 3 (i=1,j=1)
            # U_full rows:
            # 0 (i=0,j=0) <- U_sub[0]
            # 1 (i=0,j=1) <- U_sub[1]
            # 4 (i=1,j=0) <- U_sub[2]
            # 5 (i=1,j=1) <- U_sub[3]
            
            U_full[0, :r] = U_sub[0]
            U_full[1, :r] = U_sub[1]
            U_full[4, :r] = U_sub[2]
            U_full[5, :r] = U_sub[3]
            
            # Right half (indices 2,3 for A cols, 2,3 for B rows, 2,3 for C cols)
            # A_R corresponds to j=2,3.
            # A[0,2], A[0,3] -> U[2, :], U[3, :]
            # A[1,2], A[1,3] -> U[6, :], U[7, :]
            
            # B_R is rows 2,3 of B.
            # B[2,0]... wait.
            # B is 4x4.
            # B_L is B[0:2, 0:2].
            # B_R is B[2:4, 2:4]? No.
            # C_R = A_R B_R.
            # C[0, 2:4] = A[0, 2:4] B[2:4, 2:4].
            # So B_R is B[2:4, 2:4].
            # V indices for B[2:4, 2:4]:
            # B[2,2], B[2,3] -> V[2*4+2, :], V[2*4+3, :] -> V[10, :], V[11, :]
            # B[3,2], B[3,3] -> V[3*4+2, :], V[3*4+3, :] -> V[14, :], V[15, :]
            
            # C_R is C[0:2, 2:4].
            # C[0,2], C[0,3] -> W[2*2+0, :], W[2*2+1, :] -> W[4, :], W[5, :]
            # C[1,2], C[1,3] -> W[3*2+0, :], W[3*2+1, :] -> W[6, :], W[7, :]
            
            # U_sub mapping for right half:
            # U_sub[0] -> U_full[2]
            # U_sub[1] -> U_full[3]
            # U_sub[2] -> U_full[6]
            # U_sub[3] -> U_full[7]
            
            U_full[2, r:] = U_sub[0]
            U_full[3, r:] = U_sub[1]
            U_full[6, r:] = U_sub[2]
            U_full[7, r:] = U_sub[3]
            
            # V_sub mapping for left half:
            # V_sub rows: 0 (j=0,k=0), 1 (j=0,k=1), 2 (j=1,k=0), 3 (j=1,k=1)
            # V_full rows for B_L:
            # 0 (j=0,k=0) <- V_sub[0]
            # 1 (j=0,k=1) <- V_sub[1]
            # 4 (j=1,k=0) <- V_sub[2]
            # 5 (j=1,k=1) <- V_sub[3]
            
            V_full[0, :r] = V_sub[0]
            V_full[1, :r] = V_sub[1]
            V_full[4, :r] = V_sub[2]
            V_full[5, :r] = V_sub[3]
            
            # V_sub mapping for right half:
            # V_full rows for B_R:
            # 10 (j=2,k=2) <- V_sub[0]
            # 11 (j=2,k=3) <- V_sub[1]
            # 14 (j=3,k=2) <- V_sub[2]
            # 15 (j=3,k=3) <- V_sub[3]
            
            V_full[10, r:] = V_sub[0]
            V_full[11, r:] = V_sub[1]
            V_full[14, r:] = V_sub[2]
            V_full[15, r:] = V_sub[3]
            
            # W_sub mapping for left half:
            # W_sub rows: 0 (k=0,i=0), 1 (k=1,i=0), 2 (k=0,i=1), 3 (k=1,i=1)
            # W_full rows for C_L:
            # 0 (k=0,i=0) <- W_sub[0]
            # 1 (k=1,i=0) <- W_sub[1]
            # 4 (k=0,i=1) <- W_sub[2]
            # 5 (k=1,i=1) <- W_sub[3]
            
            W_full[0, :r] = W_sub[0]
            W_full[1, :r] = W_sub[1]
            W_full[4, :r] = W_sub[2]
            W_full[5, :r] = W_sub[3]
            
            # W_sub mapping for right half:
            # W_full rows for C_R:
            # 4 (k=2,i=0) <- W_sub[0]
            # 5 (k=3,i=0) <- W_sub[1]
            # 6 (k=2,i=1) <- W_sub[2]
            # 7 (k=3,i=1) <- W_sub[3]
            
            W_full[4, r:] = W_sub[0]
            W_full[5, r:] = W_sub[1]
            W_full[6, r:] = W_sub[2]
            W_full[7, r:] = W_sub[3]
            
            # Verify
            if _verify_decomposition(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2*r
                break # Found a lower rank, stop searching for even lower? Or continue?
                # Continue to see if we can go lower?
                # If 13 works, try 12?
                
    # If no lower rank found, return 28
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
