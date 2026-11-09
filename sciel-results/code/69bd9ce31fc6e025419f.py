"""Matrix multiplication 4x4x6 tensor decomposition search."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen 2x2x2 decomposition (rank 7)."""
    # Strassen's algorithm: C = A*B where A,B,C are 2x2
    # Uses 7 multiplications instead of 4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Tensor entries: U[i,j], V[j,k], W[k,i] for i,j in {0,1}, k in {0,1}
    # Each rank-1 component is a triple of 2x2 matrices
    # U components (for A), V components (for B), W components (for C)
    
    # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    # M2: (A21+A22)B11 -> C21 += 1
    # M3: A11(B21-B22) -> C12 += 1
    # M4: A22(B12-B11) -> C11 += 1, C21 += 1
    # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
    # M6: (A21-A11)(B11+B12) -> C22 += 1
    # M7: (A12-A22)(B21+B22) -> C11 += 1
    
    U = np.array([
        [1, 0, 0, 0, 1, 0, 0],   # A11: M1, M5
        [0, 0, 1, 0, 0, 1, 0],   # A12: M3, M7
        [0, 1, 0, 0, 0, 1, 0],   # A21: M2, M6
        [1, 1, 0, 1, 0, 0, 0],   # A22: M1, M2, M4
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],   # B11: M1, M2, M4, M6
        [0, 0, 1, 1, 0, 0, 1],   # B12: M3, M4, M7
        [0, 0, 1, 0, 1, 0, 0],   # B21: M3, M5, M7
        [1, 0, 0, 0, 1, 0, 0],   # B22: M1, M5
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11: M1, M4, -M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C12: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C21: M2, M4
        [1, 0, 0, 0, 0, 1, 0],   # C22: M1, M6
    ], dtype=float)
    
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], r))
    V = np.zeros((V1.shape[0] * V2.shape[0], r))
    W = np.zeros((W1.shape[0] * W2.shape[0], r))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            U[:, c] = np.kron(U1[:, c1], U2[:, c2])
            V[:, c] = np.kron(V1[:, c1], V2[:, c2])
            W[:, c] = np.kron(W1[:, c1], W2[:, c2])
    
    return U, V, W

def _build_4x4x4_decomposition():
    """Build a rank-49 decomposition for 4x4x4 using Kronecker of Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    U4, V4, W4 = _kron_decompose(U2, V2, W2, U2, V2, W2)
    return U4, V4, W4

def _build_4x4x6_decomposition():
    """
    Build a decomposition for 4x4x6.
    Strategy: 4x4x6 = (2x2x3) ⊗ (2x2x2) using Kronecker structure.
    Or: treat as two 4x4x3 blocks.
    
    Actually, let's use: 4x4x6 where we split the 6 into 3+3.
    C(4x6) = A(4x4) * B(4x6)
    Split B into B1(4x3), B2(4x3)
    C = [A*B1 | A*B2]
    
    For each 4x4x3 block, we can use a decomposition.
    4x4x3: A(4x4)*B(4x3)=C(4x3)
    This can be seen as (2x2x3) ⊗ (2x2x2) ? No, that gives 2x2x3 and 2x2x2.
    
    Let me think differently. Use the known fact that rank(4x4x4) <= 18.
    For 4x4x6, we can potentially use rank <= 18 * 2 = 36 if we can share structure,
    or rank <= 18 + 18 = 36 by treating as two independent 4x4x3 problems.
    
    Actually, a simpler approach: use the Kronecker decomposition.
    4x4x6 = (2x2) x (2x3) for the first two dims... 
    
    Let me just use a practical approach: create a 4x4x6 decomposition by
    taking a 4x4x4 decomposition and extending it.
    """
    # Start with 4x4x4 rank-49 decomposition from Kronecker of Strassen
    U444, V444, W444 = _build_4x4x4_decomposition()
    
    # For 4x4x6, we need to extend the output dimension from 4 to 6.
    # The tensor is T[i,j,k] for i in 0..3, j in 0..3, k in 0..5
    # We can think of this as: for k in 0..3, use the 4x4x4 decomposition
    # for k in 4,5, we need additional terms.
    
    # Alternative: use a 2x2x3 decomposition for the "inner" product and
    # Kronecker with 2x2x2.
    
    # Let's construct a 2x2x3 decomposition first (rank <= 12 schoolbook)
    # A(2x2) * B(2x3) = C(2x3)
    # Schoolbook: 2*2*3 = 12 terms
    # Try to find a lower rank decomposition
    
    # For now, use schoolbook for 2x2x3 and Kronecker with Strassen 2x2x2
    # This gives rank 12 * 7 = 84, which is better than 96 schoolbook
    
    U223 = np.zeros((2*2, 12))
    V223 = np.zeros((2*3, 12))
    W223 = np.zeros((3*2, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U223[i*2+j, col] = 1
                V223[j*3+k, col] = 1
                W223[k*2+i, col] = 1
                col += 1
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # Kronecker: (2x2x3) ⊗ (2x2x2) gives 4x4x6
    # U: (2*2) x (2*2) = 4 x 4 rows, rank 12*7 = 84
    # V: (2*3) x (2*2) = 6 x 4 rows, rank 84
    # W: (3*2) x (2*2) = 6 x 4 rows, rank 84
    U446, V446, W446 = _kron_decompose(U223, V223, W223, U2, V2, W2)
    
    return U446, V446, W446

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96

    # Build schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    # Build structured decomposition (Kronecker of Strassen)
    U_struct, V_struct, W_struct = _build_4x4x6_decomposition()
    struct_rank = U_struct.shape[1]
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = schoolbook_rank
    
    # Check if structured decomposition is valid
    if _verify(U_struct, V_struct, W_struct, n, m, p, tol=1e-8):
        best_U, best_V, best_W = U_struct.copy(), V_struct.copy(), W_struct.copy()
        best_rank = struct_rank

    # Precompute index arrays for fast vectorized evaluation
    idx_i = np.repeat(np.arange(n), m*p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n*m)

    # Row indices for U, V, W
    U_rows = idx_i * m + idx_j
    V_rows = idx_j * p + idx_k
    W_rows = idx_k * n + idx_i

    def compute_error_vec(U, V, W):
        """Compute sum of squared errors using vectorized operations."""
        U_gathered = U[U_rows]  # (nmp, rank)
        V_gathered = V[V_rows]  # (nmp, rank)
        W_gathered = W[W_rows]  # (nmp, rank)
        vals = np.sum(U_gathered * V_gathered * W_gathered, axis=1)
        err = np.sum((1.0 - vals)**2)
        return err, vals

    def compute_gradients(U, V, W, vals):
        """Compute gradients of error w.r.t. U, V, W."""
        dvals = -2.0 * (1.0 - vals)  # (nmp,)
        
        U_gathered = U[U_rows]
        V_gathered = V[V_rows]
        W_gathered = W[W_rows]
        
        dU_contrib = dvals[:, None] * V_gathered * W_gathered  # (nmp, rank)
        dU = np.zeros_like(U)
        np.add.at(dU, U_rows, dU_contrib)
        
        dV_contrib = dvals[:, None] * U_gathered * W_gathered
        dV = np.zeros_like(V)
        np.add.at(dV, V_rows, dV_contrib)
        
        dW_contrib = dvals[:, None] * U_gathered * V_gathered
        dW = np.zeros_like(W)
        np.add.at(dW, W_rows, dW_contrib)
        
        return dU, dV, dW

    def gradient_descent(rank, U_init=None, V_init=None, W_init=None, max_iters=5000, seed=42, lr_init=0.05):
        """Try to find a rank-r decomposition via gradient descent."""
        rng = np.random.RandomState(seed)
        
        if U_init is not None:
            U = U_init[:, :rank].copy() if U_init.shape[1] >= rank else np.hstack([U_init, rng.randn(n*m, rank - U_init.shape[1]) * 0.5])
            V = V_init[:, :rank].copy() if V_init.shape[1] >= rank else np.hstack([V_init, rng.randn(m*p, rank - V_init.shape[1]) * 0.5])
            W = W_init[:, :rank].copy() if W_init.shape[1] >= rank else np.hstack([W_init, rng.randn(p*n, rank - W_init.shape[1]) * 0.5])
        elif rank >= schoolbook_rank:
            U = U_sb[:, :rank].copy()
            V = V_sb[:, :rank].copy()
            W = W_sb[:, :rank].copy()
            U += rng.randn(n*m, rank) * 0.01
            V += rng.randn(m*p, rank) * 0.01
            W += rng.randn(p*n, rank) * 0.01
        else:
            U = rng.randn(n*m, rank) * 0.5
            V = rng.randn(m*p, rank) * 0.5
            W = rng.randn(p*n, rank) * 0.5
        
        err, vals = compute_error_vec(U, V, W)
        best_err = err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            dU, dV, dW = compute_gradients(U, V, W, vals)
            
            U_norm = np.linalg.norm(U)
            V_norm = np.linalg.norm(V)
            W_norm = np.linalg.norm(W)
            
            U -= lr * dU / max(U_norm, 1e-8) * U_norm
            V -= lr * dV / max(V_norm, 1e-8) * V_norm
            W -= lr * dW / max(W_norm, 1e-8) * W_norm
            
            new_err, new_vals = compute_error_vec(U, V, W)
            
            if new_err < err:
                err = new_err
                vals = new_vals
                lr = min(lr * 1.05, 0.5)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    break
            else:
                lr = max(lr * 0.7, 1e-6)
                if err > best_err * 1.5:
                    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                    err, vals = compute_error_vec(U, V, W)
        
        if best_err < 1e-8:
            return best_U, best_V, best_W
        return None

    # Try to improve the structured decomposition
    # First, try to reduce rank of the structured decomposition
    if struct_rank < schoolbook_rank:
        for r in range(struct_rank - 1, max(struct_rank - 20, 0), -1):
            if time.time() > deadline - 20:
                break
            found = False
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                result = gradient_descent(r, U_init=U_struct, V_init=V_struct, W_init=W_struct, max_iters=2000, seed=seed, lr_init=0.02)
                if result is not None:
                    U_list, V_list, W_list = result
                    if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                        best_U, best_V, best_W = U_list, V_list, W_list
                        best_rank = r
                        found = True
                        break
            if found:
                continue
    
    # Also try random initializations for even lower ranks
    for r in [best_rank - 1, best_rank - 2, best_rank - 5, best_rank - 10]:
        if r < 10 or time.time() > deadline - 15:
            break
        found = False
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            result = gradient_descent(r, max_iters=1500, seed=seed, lr_init=0.03)
            if result is not None:
                U_list, V_list, W_list = result
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = r
                    found = True
                    break
        if found:
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
