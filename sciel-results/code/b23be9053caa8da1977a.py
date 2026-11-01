"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find minimal-rank exact tensor decomposition for 2x2 * 2x4 matrix multiplication."""
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    # Input dimensions: n*m = 4 (A), m*p = 8 (B), output: p*n = 8 (C)
    # Tensor T[a, b, c] = 1 if c = k*n+i and a = k*m+j and b = j*p+l and ... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # A is 2x2 (n=2 rows, m=2 cols), B is 2x4 (m=2 rows, p=4 cols)
    # C is 2x4 (n=2 rows, p=4 cols)
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # Indexing:
    # A index: i*m + j (i in 0..n-1, j in 0..m-1) -> 4 elements
    # B index: j*p + k (j in 0..m-1, k in 0..p-1) -> 8 elements  
    # C index: k*n + i (k in 0..p-1, i in 0..n-1) -> 8 elements (trace convention W[k*n+i, r])
    
    # The multiplication tensor T is a 3-tensor of shape (n*m, m*p, p*n)
    # T[a_idx, b_idx, c_idx] = 1 if the product A[a_idx] * B[b_idx] contributes to C[c_idx]
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1
    
    # We want rank-r decomposition: T = sum_{r=1}^{R} u_r * v_r * w_r
    # where u_r is (n*m)-vec, v_r is (m*p)-vec, w_r is (p*n)-vec
    
    # Strategy: Try rank 10, 11, 12, 13 with structured patterns
    # Use the fact that we can try various bilinear forms
    
    # First, let's try a direct algebraic approach
    # For 2x2 by 2x4, we can split B into two 2x2 blocks: B = [B1 | B2]
    # C = A * B = [A*B1 | A*B2]
    # Each A*B1 and A*B2 is 2x2 multiplication
    
    # Using Strassen for each 2x2 block gives 7+7=14
    # But we might share some computations between the two blocks
    
    # Let's try to find a rank-13 solution by sharing one term
    # Or try rank 12, 11, 10 with more aggressive sharing
    
    # Approach: Use numpy to verify and search
    # For a given set of rank-1 terms, verify they span the tensor
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W give exact decomposition."""
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.allclose(T_recon, T, atol=1e-10)
    
    def try_rank(rank):
        """Try to find a rank-r decomposition using structured search."""
        if time.time() > deadline:
            return None
        
        # For small ranks, use a more targeted approach
        # The key insight: for 2x2*2x4, we have specific structure
        
        # Let's try to construct using known bilinear forms
        # A = [[a, b], [c, d]] (indices 0,1,2,3)
        # B = [[e, f, g, h], [i, j, k, l]] (indices 0-7)
        # C = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
        # C indices (k*n+i): C[0,0]=ae+bi, C[1,0]=ce+di, C[0,1]=af+bj, C[1,1]=cf+dj,
        #                    C[0,2]=ag+bk, C[1,2]=cg+dk, C[0,3]=ah+bl, C[1,3]=ch+dl
        
        # Try various rank constructions
        # Rank 14 (current best): Strassen on each block
        # Try to find rank 13 by sharing a term between blocks
        
        # Strassen forms for A:
        # M1 = (a+d)(e+l)
        # M2 = (c+d)(e)  
        # M3 = (a+b)(l)
        # M4 = (d-b)(j+k)  -- wait, let me use standard Strassen
        # Actually standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a+b)(h)
        # p4 = (d-b)(g+h)
        # p5 = (a+c)(e+f)
        # p6 = (b-d)(f+h)
        # p7 = (a-c)(g+h)
        # Result: [[p1+p4-p5+p7, p3+p6], [p2+p4, p1-p2+p3+p5]]
        
        # For block 1 (cols 0,1 of B): B1 = [[e,f],[i,j]]
        # For block 2 (cols 2,3 of B): B2 = [[g,h],[k,l]]
        
        # Can we share terms? 
        # p1_1 = (a+d)(e+j), p1_2 = (a+d)(g+l) -- different
        # p2_1 = (c+d)(e), p2_2 = (c+d)(g) -- different
        # p3_1 = (a+b)(j), p3_2 = (a+b)(l) -- different
        # p4_1 = (d-b)(i+j), p4_2 = (d-b)(k+l) -- different
        # p5_1 = (a+c)(e+f), p5_2 = (a+c)(g+h) -- different
        # p6_1 = (b-d)(f+j), p6_2 = (b-d)(h+l) -- different
        # p7_1 = (a-c)(i+j), p7_2 = (a-c)(k+l) -- different
        
        # No obvious sharing. Let's try a different approach.
        
        # Use a linear algebra approach: find the minimum rank
        # by checking if we can express the tensor as sum of rank-1 terms
        
        # For rank r, we need to find U (4xr), V (8xr), W (8xr) such that
        # T = U @ diag(r) @ V^T ... no, it's T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # This is a non-convex optimization. Let's try a greedy/approximate approach.
        
        # Alternative: use the fact that the tensor has specific sparsity
        # T has exactly 8 non-zero entries (one for each (i,j,k) triple)
        # T[0,0,0]=1, T[0,1,2]=1, T[0,2,4]=1, T[0,3,6]=1
        # T[1,0,1]=1, T[1,1,3]=1, T[1,2,5]=1, T[1,3,7]=1
        # Wait, let me recheck:
        # a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # (i,j,k) = (0,0,0): a=0, b=0, c=0
        # (i,j,k) = (0,0,1): a=0, b=1, c=2
        # (i,j,k) = (0,0,2): a=0, b=2, c=4
        # (i,j,k) = (0,0,3): a=0, b=3, c=6
        # (i,j,k) = (0,1,0): a=1, b=4, c=1
        # (i,j,k) = (0,1,1): a=1, b=5, c=3
        # (i,j,k) = (0,1,2): a=1, b=6, c=5
        # (i,j,k) = (0,1,3): a=1, b=7, c=7
        # (i,j,k) = (1,0,0): a=2, b=0, c=0
        # (i,j,k) = (1,0,1): a=2, b=1, c=2
        # (i,j,k) = (1,0,2): a=2, b=2, c=4
        # (i,j,k) = (1,0,3): a=2, b=3, c=6
        # (i,j,k) = (1,1,0): a=3, b=4, c=1
        # (i,j,k) = (1,1,1): a=3, b=5, c=3
        # (i,j,k) = (1,1,2): a=3, b=6, c=5
        # (i,j,k) = (1,1,3): a=3, b=7, c=7
        
        # So T has 16 non-zero entries.
        
        # Let's try to use a known result: the bilinear complexity of 2x2*2x4
        # is known to be 13 (from literature on rectangular matrix multiplication)
        
        # Let me try to construct a rank-13 solution.
        # One approach: use 6 terms for one "half" and 7 for the other, with 1 shared.
        
        # Actually, let's try a numerical optimization approach for small ranks.
        # We'll parameterize U, V, W and minimize ||T - reconstructed||^2
        
        from scipy.optimize import minimize
        
        def objective(params, rank):
            U = params[:4*rank].reshape(4, rank)
            V = params[4*rank:4*rank+8*rank].reshape(8, rank)
            W = params[4*rank+8*rank:].reshape(8, rank)
            T_recon = np.zeros((4, 8, 8))
            for r in range(rank):
                T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            return np.sum((T_recon - T)**2)
        
        # Try rank 13 first
        for rank in [13, 12, 11, 10]:
            if time.time() > deadline:
                break
            
            # Try multiple random initializations
            best_obj = float('inf')
            best_params = None
            
            n_trials = 50 if rank >= 12 else 30
            for trial in range(n_trials):
                if time.time() > deadline:
                    break
                
                # Initialize with small random values
                np.random.seed(trial * 1000 + rank)
                init = np.random.randn(4*rank + 8*rank + 8*rank) * 0.1
                
                res = minimize(objective, init, args=(rank,), method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
                
                if best_obj < 1e-8:
                    break
            
            if best_obj < 1e-8:
                # Found a valid decomposition
                U = best_params[:4*rank].reshape(4, rank)
                V = best_params[4*rank:4*rank+8*rank].reshape(8, rank)
                W = best_params[4*rank+8*rank:].reshape(8, rank)
                
                # Convert to list format
                U_list = [[U[i][r] for r in range(rank)] for i in range(4)]
                V_list = [[V[i][r] for r in range(rank)] for i in range(8)]
                W_list = [[W[i][r] for r in range(rank)] for i in range(8)]
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no better solution found, return the rank-14 solution
        return _rank14_solution()
    
    def _rank14_solution():
        """Fallback rank-14 Strassen-based solution."""
        rank = 14
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(8)]
        W = [[0] * rank for _ in range(8)]
        
        # Strassen for block 1 (B cols 0,1)
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1
        
        # Strassen for block 2 (B cols 2,3)
        U[0][7] = 1; U[3][7] = 1
        U[2][8] = 1; U[3][8] = 1
        U[0][9] = 1
        U[3][10] = 1
        U[0][11] = 1; U[1][11] = 1
        U[2][12] = 1; U[0][12] = -1
        U[1][13] = 1; U[3][13] = -1
        
        # V for block 1
        V[0][0] = 1; V[5][0] = 1
        V[0][1] = 1
        V[1][2] = 1; V[5][2] = -1
        V[4][3] = 1; V[0][3] = -1
        V[5][4] = 1
        V[0][5] = 1; V[1][5] = 1
        V[4][6] = 1; V[5][6] = 1
        
        # V for block 2
        V[2][7] = 1; V[7][7] = 1
        V[2][8] = 1
        V[3][9] = 1; V[7][9] = -1
        V[6][10] = 1; V[2][10] = -1
        V[7][11] = 1
        V[2][12] = 1; V[3][12] = 1
        V[6][13] = 1; V[7][13] = 1
        
        # W
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
        W[1][1] = 1; W[1][3] = 1
        W[2][2] = 1; W[2][4] = 1
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
        W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
        W[5][8] = 1; W[5][10] = 1
        W[6][9] = 1; W[6][11] = 1
        W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
        
        return {"U": U, "V": V, "W": W}
    
    result = try_rank(14)
    if result is None:
        result = _rank14_solution()
    
    return result
# EVOLVE-BLOCK-END

