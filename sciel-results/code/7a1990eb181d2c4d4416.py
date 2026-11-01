"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k,l] = 1 if i==k and j==l (for A[i,j]*B[j,k] -> C[i,k])
    # Actually for matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T_{i,j}^{k,l} where A has indices (i,j), B has (j,k), C has (i,k)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} * delta_{j,j} ... 
    # Actually it should equal 1 if the indices match for the multiplication, 0 otherwise
    # The tensor for matmul (n x m) * (m x p) -> (n x p):
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all valid i,j,k
    
    # Let's build the expected tensor and compare
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _strassen_2x2x6():
    """Return the rank-21 Strassen decomposition."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1

    return U, V, W

def _try_reduce_rank(deadline):
    """Try to find a rank-20 decomposition by perturbing the rank-21 solution."""
    n, m, p = 2, 2, 6
    U0, V0, W0 = _strassen_2x2x6()
    
    # Try to find a rank-20 solution by setting one column to zero and solving
    # This is a feasibility problem: find U,V,W with rank 20 such that the tensor matches
    
    # Strategy: Start from rank-21, try to merge two rank terms
    # If we can express one rank term as a linear combination of others in a way that preserves the bilinear structure...
    
    # Alternative: Use the fact that the tensor might have a lower rank due to the rectangular structure
    # Let's try a direct numerical search for rank-20
    
    # The matmul tensor for 2x2 * 2x6 has 4*12*12 = 576 equations (but many are trivially 0)
    # Non-trivial: for each i in [0,2), j in [0,2), k in [0,6): sum_r U[i*2+j,r]*V[j*6+k,r]*W[k*2+i,r] = 1
    # That's 2*2*6 = 24 equations
    
    # For rank 20: 20 variables per matrix, 4*20 + 12*20 + 12*20 = 560 unknowns, 24 equations
    # Very underdetermined, so rank 20 should be feasible if the structure allows
    
    # Let's try: take the rank-21 solution and try to eliminate one term
    # Set the last term (r=20) to be a combination that cancels out
    
    # Actually, let me try a different approach: 
    # Use 2 full Strassen blocks (rank 14) and handle the third 2x2 block with 6 terms
    # (which is possible since 2x2 matmul has rank at most 7, but maybe we can do 6 with shared structure)
    
    # Or: use a different decomposition for the 2x6 structure
    # Treat B as [B1, B2, B3] where each Bi is 2x2
    # C = A * [B1, B2, B3] = [A*B1, A*B2, A*B3]
    
    # The key question: can we share some Strassen terms across blocks?
    # In standard Strassen, the U factors depend only on A, so they can be shared!
    # But the V factors depend on B, and W factors depend on which output position
    
    # Actually in the current decomposition, each block has its own 7 terms because
    # V and W are different for each block. But U patterns repeat.
    
    # Let me try: can we use only 6 terms for one of the 2x2 blocks?
    # 2x2 matmul has known minimum rank 7 over reals (Strassen), so no.
    
    # What about using a different decomposition that isn't block-wise Strassen?
    # The total tensor might have rank < 21 due to cross-block interactions
    
    # Let's try a numerical optimization approach
    # Minimize the residual of the tensor equations with rank 20
    
    np.random.seed(42)
    best_rank = 21
    best_U, best_V, best_W = U0, V0, W0
    
    # Try rank 20
    for target_rank in [20]:
        if time.time() > deadline - 5:
            break
        
        # Initialize with rank-21 solution, zero out last column
        U = [row[:] for row in U0]
        V = [row[:] for row in V0]
        W = [row[:] for row in W0]
        
        # Zero out the last column (index target_rank)
        for i in range(len(U)):
            U[i][target_rank] = 0
        for i in range(len(V)):
            V[i][target_rank] = 0
        for i in range(len(W)):
            W[i][target_rank] = 0
        
        # Now try to adjust the remaining columns to compensate
        # Use gradient descent on the residual
        
        def compute_residual(U, V, W, n, m, p, rank):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                        res += (s - 1.0) ** 2
            return res
        
        # Simple iterative improvement
        lr = 0.01
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            res = compute_residual(U, V, W, n, m, p, target_rank)
            if res < 1e-12:
                break
            
            # Compute gradients numerically would be expensive; use a simpler approach
            # Try random perturbations and keep improvements
            improved = False
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random element to perturb
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = np.random.randint(len(U))
                    col = np.random.randint(target_rank)
                    old = U[idx][col]
                    U[idx][col] = old + 0.01 * np.random.randn()
                elif mat == 'V':
                    idx = np.random.randint(len(V))
                    col = np.random.randint(target_rank)
                    old = V[idx][col]
                    V[idx][col] = old + 0.01 * np.random.randn()
                else:
                    idx = np.random.randint(len(W))
                    col = np.random.randint(target_rank)
                    old = W[idx][col]
                    W[idx][col] = old + 0.01 * np.random.randn()
                
                new_res = compute_residual(U, V, W, n, m, p, target_rank)
                if new_res < res:
                    res = new_res
                    improved = True
                else:
                    if mat == 'U':
                        U[idx][col] = old
                    elif mat == 'V':
                        V[idx][col] = old
                    else:
                        W[idx][col] = old
            
            if not improved:
                # Try larger step
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        final_res = compute_residual(U, V, W, n, m, p, target_rank)
        if final_res < 1e-10:
            # Round to nearest half-integers
            U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded, target_rank
    
    return U0, V0, W0, 21

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    U, V, W, rank = _try_reduce_rank(deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
