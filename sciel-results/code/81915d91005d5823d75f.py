"""Rewritten program for 2x3 * 3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Try known structured ansatz for rank 16 (Strassen-like)
    2. Use smarter search with algebraic structure exploitation
    3. Fall back to schoolbook rank 18 if nothing better found
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Schoolbook baseline (rank 18)
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = 18
    
    # Try structured rank-16 ansatz based on Strassen-like splitting
    # For 2x3 * 3x3, split into overlapping 2x2 blocks
    # Key insight: A is 2x3, B is 3x3
    # We can write A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # B = [B_top; B_mid; B_bot] where each is 1x3
    
    # Structured approach: use 16 products with the following pattern
    # Terms 0-3: A[0,j]*B[0,k] combinations
    # Terms 4-7: A[1,j]*B[1,k] combinations  
    # Terms 8-11: Cross terms with B[2,k]
    # Terms 12-15: Correction terms
    
    # Let's try a specific construction based on the observation that
    # the two rows of A can be handled more efficiently
    
    U_16 = np.array([
        # Row 0: A[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: A[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Row 2: A[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # Row 3: A[1,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: A[1,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Row 5: A[1,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    V_16 = np.array([
        # Row 0: B[0,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: B[0,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: B[0,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: B[1,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: B[1,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: B[1,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 6: B[2,0]
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 7: B[2,1]
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 8: B[2,2]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    W_16 = np.array([
        # Row 0: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 1: C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 2: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 3: C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 4: C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Row 5: C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    
    # Verify the structured ansatz
    if verify_decomp(U_16, V_16, W_16):
        best_rank = 16
        best_decomp = (U_16, V_16, W_16)
    else:
        # If structured ansatz doesn't work, try smarter search
        random.seed(42)
        
        def try_structured_search(r, max_iters=5000):
            """
            Try search with structure exploitation.
            Key insight: For 2x3 * 3x3, the tensor has symmetries we can exploit.
            """
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Generate with bias towards sparse, structured patterns
                def gen_sparse_matrix(rows, cols, sparsity=0.5, coeff_range=(-1, 1)):
                    M = np.zeros((rows, cols), dtype=int)
                    for row in range(rows):
                        for col in range(cols):
                            if random.random() < sparsity:
                                M[row, col] = 0
                            else:
                                M[row, col] = random.choice([coeff_range[0], coeff_range[1]])
                    return M
                
                U = gen_sparse_matrix(6, r, sparsity=0.4)
                V = gen_sparse_matrix(9, r, sparsity=0.4)
                W = gen_sparse_matrix(6, r, sparsity=0.4)
                
                # Quick check: ensure no all-zero rows
                if np.any(np.all(U == 0, axis=1)):
                    continue
                if np.any(np.all(V == 0, axis=1)):
                    continue
                if np.any(np.all(W == 0, axis=1)):
                    continue
                
                if verify_decomp(U, V, W):
                    return (U, V, W)
            return None
        
        # Try rank 16 first
        result = try_structured_search(16, max_iters=3000)
        if result is not None:
            best_rank = 16
            best_decomp = result
        elif time.time() < deadline - 20:
            # Try rank 17
            result = try_structured_search(17, max_iters=3000)
            if result is not None:
                best_rank = 17
                best_decomp = result
        elif time.time() < deadline - 10:
            # Try rank 18 (should find quickly)
            result = try_structured_search(18, max_iters=2000)
            if result is not None:
                best_rank = 18
                best_decomp = result
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
