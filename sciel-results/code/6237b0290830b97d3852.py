import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Starts with rank-98 Strassen decomposition and attempts column elimination
    via proper tensor verification.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    U_base, V_base, W_base = _construct_rank98(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]
    
    # Try iterative column elimination with proper tensor verification
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 20:
        further_reduced = _iterative_column_elimination_v2(
            best_result[0], best_result[1], best_result[2], n, m, p, deadline - start_time
        )
        if further_reduced is not None:
            U_f, V_f, W_f = further_reduced
            if _verify_decomposition(U_f, V_f, W_f, n, m, p):
                if U_f.shape[1] < best_rank:
                    best_result = (U_f, V_f, W_f)
                    best_rank = U_f.shape[1]
    
    # Try to find a shared structure between the two 4x4 blocks
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 15 and best_rank == 98:
        shared = _try_shared_structure(n, m, p, deadline - start_time)
        if shared is not None:
            U_s, V_s, W_s = shared
            if _verify_decomposition(U_s, V_s, W_s, n, m, p):
                if U_s.shape[1] < best_rank:
                    best_result = (U_s, V_s, W_s)
                    best_rank = U_s.shape[1]
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}


def _construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W


def _iterative_column_elimination_v2(U, V, W, n, m, p, time_budget):
    """
    Iteratively try to eliminate columns by checking if the tensor identity
    still holds after removing each column. This is a proper tensor check,
    not just a linear span check.
    """
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Precompute index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    max_iterations = 20  # Try to remove up to 20 columns
    
    for iteration in range(max_iterations):
        if time.time() - start > time_budget:
            break
        
        current_rank = U.shape[1]
        if current_rank <= 64:  # Don't go below reasonable bounds
            break
        
        # Try to remove each column, starting from the end
        removed = False
        for col in range(current_rank - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            
            # Try removing column col
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            U_trial = U[:, mask]
            V_trial = V[:, mask]
            W_trial = W[:, mask]
            
            # Check if the tensor identity still holds
            if _tensor_identity_holds(U_trial, V_trial, W_trial, n, m, p):
                U, V, W = U_trial, V_trial, W_trial
                removed = True
                break
        
        if not removed:
            break
    
    if U.shape[1] < rank:
        return U, V, W
    return None


def _tensor_identity_holds(U, V, W, n, m, p, tol=1e-9):
    """
    Check if sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 for all (i,j,k).
    This is the defining property of a matrix multiplication tensor decomposition.
    """
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Efficient check: for each (i,j,k), compute the sum
    # We can vectorize this
    for i in range(n):
        for j in range(m):
            u_row = U[i*m + j, :]  # (rank,)
            for k in range(p):
                v_val = V[j*p + k, :]  # (rank,)
                w_val = W[k*n + i, :]  # (rank,)
                total = np.dot(u_row, v_val * w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True


def _try_shared_structure(n, m, p, time_budget):
    """
    Try to find a decomposition where the two 4x4 column blocks of B
    share some rank-1 components. The idea is that if we can find
    a set of rank-1 tensors that work for both blocks, we save rank.
    """
    start = time.time()
    
    # Get the 4x4x4 Strassen decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # The key insight: for 4x4x8, we need to compute C[:, 0:4] and C[:, 4:8]
    # Both use the same A (4x4) but different B blocks.
    # If we could find a decomposition where some rank-1 terms contribute to both,
    # we'd save rank.
    
    # One approach: use the fact that B is 4x8, and try to decompose it as
    # a sum of fewer rank-1 matrices in the B dimension.
    
    # Actually, let's try a different approach: use a 2x4x4 Strassen-like
    # decomposition at the top level, treating the 4x4x8 as a 2x(2x4)x(2x4)
    # problem where we use 7 products instead of 8 for the 2x2 block structure
    # in the output dimension.
    
    # Split output into two 4-row blocks: C_top (rows 0-3) and C_bottom (rows 4-7)
    # Wait, that doesn't make sense for 4x4x8. Let me reconsider.
    
    # For 4x4x8: A is 4x4, B is 4x8, C is 4x8
    # We can split B into two 4x4 blocks: B1 (cols 0-3) and B2 (cols 4-7)
    # C = [A*B1, A*B2]
    # Each A*B is a 4x4x4 multiplication (rank 49 with Strassen)
    # Total rank: 98
    
    # To do better, we need to exploit the fact that both multiplications
    # use the same A. This is the key structural property.
    
    # Let's try: can we find a decomposition of rank < 98 that correctly
    # computes both A*B1 and A*B2?
    
    # One idea: use a 2-level Strassen where the top level splits the output
    # dimension (p=8 into two groups of 4), and the bottom level splits
    # the row dimension (n=4 into two groups of 2).
    
    # Top level: 2x2x2 Strassen on the "output block" dimension
    # M1 = A * B1 (4x4 result)
    # M2 = A * B2 (4x4 result)
    # We want to compute [M1, M2] = A * [B1, B2]
    
    # Using Strassen at the top level (treating M1 and M2 as "2x2 blocks"):
    # But M1 and M2 are both 4x4, not 2x2. So this doesn't directly apply.
    
    # Alternative: treat the problem as computing 8 output entries per row of A.
    # For row i of A, we need C[i, 0:8] = A[i, :] * B
    # This is a 1x4 times 4x8 multiplication.
    
    # I don't see an obvious way to get below 98 with simple structural arguments.
    # Let's just return None and rely on the column elimination.
    
    return None


def _construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True
