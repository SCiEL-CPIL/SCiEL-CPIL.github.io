"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline, then attempt
    to find a lower-rank decomposition via structured local search starting from rank 33.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety (150s wall-clock limit)

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T (vectorized)."""
        rank = U.shape[1]
        # Vectorized: sum over rank dimension
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                col += 1
        
        return U, V, W

    def local_search_for_lower_rank(U33, V33, W33, target_rank, time_limit):
        """
        Try to find a rank-`target_rank` decomposition by starting from the rank-33
        solution, dropping one term, and doing gradient-based refinement with integer rounding.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline - 5:
            return None
        
        T_f = T.astype(np.float64)
        
        # Try dropping each of the 33 terms in turn
        for drop_idx in range(33):
            if time.time() > deadline - 5:
                break
            
            # Initialize with 32 terms (all except drop_idx)
            idxs = [i for i in range(33) if i != drop_idx]
            U = U33[:, idxs].astype(np.float64).copy()
            V = V33[:, idxs].astype(np.float64).copy()
            W = W33[:, idxs].astype(np.float64).copy()
            
            # Multiple restarts with different perturbations
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                
                # Add random perturbation to escape the exact rank-33 manifold
                scale = 0.1 * (restart + 1) * 0.5
                U_p = U + np.random.randn(*U.shape) * scale
                V_p = V + np.random.randn(*V.shape) * scale
                W_p = W + np.random.randn(*W.shape) * scale
                
                # Gradient descent with adaptive learning rate
                lr = 0.02
                for it in range(300):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute residual (vectorized)
                    residual = T_f - np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                    
                    # Check convergence
                    res_norm = np.linalg.norm(residual)
                    if res_norm < 1e-8:
                        # Try rounding to integers
                        U_int = np.round(U_p).astype(np.int64)
                        V_int = np.round(V_p).astype(np.int64)
                        W_int = np.round(W_p).astype(np.int64)
                        if verify_decomposition(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                        # Also try half-integer rounding
                        U_half = np.round(U_p * 2) / 2
                        V_half = np.round(V_p * 2) / 2
                        W_half = np.round(W_p * 2) / 2
                        # Check if half-integers give exact match
                        result_half = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half)
                        if np.allclose(result_half, T_f, atol=1e-10):
                            # Convert to int64 by doubling
                            U_i2 = (U_half * 2).astype(np.int64)
                            V_i2 = (V_half * 2).astype(np.int64)
                            W_i2 = (W_half * 2).astype(np.int64)
                            # Scale W by 1/4 to compensate? No, we need exact integers.
                            # Actually if entries are half-integers, multiply all by 2 and divide result by 8? 
                            # No - the decomposition must have integer/half-integer entries directly.
                            # Let's just check if the rounded version works
                            pass
                        break
                    
                    # Compute gradients (vectorized per term)
                    for r in range(target_rank):
                        grad_U = np.einsum('j,k,ijk->i', V_p[:, r], W_p[:, r], residual)
                        grad_V = np.einsum('i,k,ijk->j', U_p[:, r], W_p[:, r], residual)
                        grad_W = np.einsum('i,j,ijk->k', U_p[:, r], V_p[:, r], residual)
                        
                        U_p[:, r] -= lr * grad_U
                        V_p[:, r] -= lr * grad_V
                        W_p[:, r] -= lr * grad_W
                    
                    # Adapt learning rate
                    if it % 50 == 49:
                        lr *= 0.9
                    
                    # Periodically try rounding
                    if it % 25 == 24:
                        U_int = np.round(U_p).astype(np.int64)
                        V_int = np.round(V_p).astype(np.int64)
                        W_int = np.round(W_p).astype(np.int64)
                        if verify_decomposition(U_int, V_int, W_int):
                            return U_int, V_int, W_int
            
            # Final check after all restarts for this drop_idx
            residual = T_f - np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            if np.linalg.norm(residual) < 1e-8:
                U_int = np.round(U_p).astype(np.int64)
                V_int = np.round(V_p).astype(np.int64)
                W_int = np.round(W_p).astype(np.int64)
                if verify_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        
        return None

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p
    else:
        # Try to find a rank-32 decomposition
        if time.time() < deadline - 10:
            result = local_search_for_lower_rank(best_U, best_V, best_W, 32, 35)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = 32
        
        # If we found rank 32, try rank 31
        if best_rank == 32 and time.time() < deadline - 10:
            result = local_search_for_lower_rank(best_U, best_V, best_W, 31, 20)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = 31

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
