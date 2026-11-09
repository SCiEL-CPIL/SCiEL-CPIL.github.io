"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W, "rank": rank_sb}
    
    # Try to find lower rank via numerical search + exact verification
    # The multiplication tensor T[i,j,k] = 1 if output (i,k) depends on A[i,j]*B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
    # where delta is 1 when this triple is in the tensor
    
    # Build the tensor as a 3D array
    # T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    
    # Actually, the tensor for matrix mult is:
    # M(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the full tensor)
    # Wait no - the multiplication tensor is defined such that:
    # <T, A ⊗ B> = A*B
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # For rank-r decomposition: A*B = sum_c (U_c^T A) * (V_c^T B) * W_c
    # where U_c is vector of size n*m, V_c of size m*p, W_c of size p*n
    
    # Let me use a numerical approach: try ranks from low to high
    # Use random initialization and gradient-free optimization
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Define the target tensor
    # T[i, j, k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is equivalent to: for each c, we have three vectors
    # u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # such that the outer product u_c ⊗ v_c ⊗ w_c sums to T
    
    # Let's try a structured approach using known fast algorithms
    
    # For 3x3 * 3x6: split into two 3x3 * 3x3 problems
    # Each 3x3*3x3 has rank <= 23 (best known)
    # But we can try to find rank-23 decompositions numerically
    
    # Alternative: use the fact that 3x3 matrix mult can be done with rank 23
    # Known decomposition for 3x3x3x3 (from literature)
    
    # Let me try a numerical search for low rank
    # Strategy: use random initialization, then use alternating least squares
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] equals the multiplication tensor"""
        n, m, p = DIMENSIONS
        # Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U_mat[0])):
                        s += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_exact(U_mat, V_mat, W_mat):
        """Exact verification using fractions"""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(len(U_mat[0])):
                        s += Fraction(U_mat[i*m+j][c]) * Fraction(V_mat[j*p+k][c]) * Fraction(W_mat[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    # Try rank 23 for each 3x3 block approach
    # Actually, let me try a direct numerical search
    
    # For the full 3x3*3x6 problem, try various ranks
    for target_rank in range(20, min(rank_sb, 50)):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Alternating least squares: fix two, solve for third
        # For each c: U[:,c] * V[:,c] * W[:,c] should sum to T
        # Fix V and W, solve for U: U[i*m+j, c] = T[i,j,k] / (V[j*p+k,c] * W[k*n+i,c])
        # This is tricky because T is the same for all (i,j,k)
        
        # Actually, the condition is:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # Let's use a different formulation. Define:
        # For each (i,j): A_ij = sum_c U[i*m+j,c] * (V[:,c] ⊗ W[:,c]) restricted to j,k
        
        # This is getting complex. Let me try a simpler approach:
        # Use the known Strassen decomposition for 2x2 and extend
        
        # For 3x3, we can pad to 4x4 and use Strassen (rank 7 for 2x2, rank 19 for 4x4)
        # 3x3 can be embedded in 4x4, so rank <= 19 for 3x3*3x3
        # Then 3x3*3x6 = two 3x3*3x3, rank <= 38
        
        # But we can do better. Let me try rank 30-40 numerically.
        
        # Simpler approach: use the block structure
        # Split 3x6 into [3x3 | 3x3]
        # C1 = A * B1, C2 = A * B2 where B1, B2 are 3x3
        # Each can use rank-23 decomposition (if we can find one)
        
        # Let me try to find a rank-23 decomposition for 3x3*3x3 numerically
        # and verify it
        
        pass  # Skip for now, use structured approach
    
    # Structured approach: use 2x2 Strassen blocks
    # 3x3 can be partitioned as:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    
    # Use a 2x2 block structure with Strassen
    # Top-left 2x2 block of A times top-left 2x2 block of B uses Strassen (rank 7)
    # But this doesn't cover all entries efficiently
    
    # Let me just return the schoolbook for now and try a quick numerical search
    # for a slightly better rank
    
    # Quick numerical search: try rank 40 with random init and a few ALS steps
    for target_rank in [45, 42, 40, 38, 36]:
        if time.time() > deadline - 20:
            break
        
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(100 + target_rank + trial)
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # ALS: alternate fixing pairs
            for iteration in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Fix V, W; solve for U
                # For each (i,j): U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is: for each (i,j), we have a system in U[i*m+j, :]
                # Let M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then sum_c U[i*m+j,c] * M_ij[k,c] = 1 for all k
                # i.e., U[i*m+j, :] * M_ij^T = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        # Build M_ij: p x target_rank
                        M_ij = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                M_ij[k, c] = V_r[j*p+k, c] * W_r[k*n+i, c]
                        # Solve: U[i*m+j, :] @ M_ij.T = ones(p)
                        # U[i*m+j, :] = ones(p) @ pinv(M_ij.T)
                        try:
                            U_r[i*m+j, :] = np.linalg.lstsq(M_ij.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Fix U, W; solve for V
                for j in range(m):
                    for k in range(p):
                        M_jk = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                M_jk[i, c] = U_r[i*m+j, c] * W_r[k*n+i, c]
                        try:
                            V_r[j*p+k, :] = np.linalg.lstsq(M_jk.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Fix U, V; solve for W
                for k in range(p):
                    for i in range(n):
                        M_ki = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                M_ki[j, c] = U_r[i*m+j, c] * V_r[j*p+k, c]
                        try:
                            W_r[k*n+i, :] = np.linalg.lstsq(M_ki.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check residual
            residual = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                        residual += abs(s - 1)
            
            if residual < 1e-6:
                # Found a good numerical solution, try to make it exact
                # Round to nearest rational with small denominator
                U_exact = [[0]*target_rank for _ in range(n*m)]
                V_exact = [[0]*target_rank for _ in range(m*p)]
                W_exact = [[0]*target_rank for _ in range(p*n)]
                
                for idx in range(n*m):
                    for c in range(target_rank):
                        val = U_r[idx, c]
                        # Try to find a simple fraction
                        f = Fraction(val).limit_denominator(1000)
                        U_exact[idx][c] = float(f)
                
                for idx in range(m*p):
                    for c in range(target_rank):
                        val = V_r[idx, c]
                        f = Fraction(val).limit_denominator(1000)
                        V_exact[idx][c] = float(f)
                
                for idx in range(p*n):
                    for c in range(target_rank):
                        val = W_r[idx, c]
                        f = Fraction(val).limit_denominator(1000)
                        W_exact[idx][c] = float(f)
                
                # Verify
                if verify_decomposition(U_exact, V_exact, W_exact):
                    best = {"U": U_exact, "V": V_exact, "W": W_exact, "rank": target_rank}
                    break
                elif verify_exact(U_exact, V_exact, W_exact):
                    best = {"U": U_exact, "V": V_exact, "W": W_exact, "rank": target_rank}
                    break
    
    if time.time() > deadline - 5:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
