"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # leave 10s safety margin before 150s kill
    
    # Start with schoolbook (rank 72) as fallback
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook construction
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower-rank decomposition using ALS-style optimization
    # The matrix multiplication tensor T has T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 if (i,j,k) is a "multiplication" position, else 0
    
    # Actually the tensor for matrix mult is: T_{i,j,k} where the output C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition is: C[i,k] = sum_{j,r} A[i,j] * (B[j,k]) * ... 
    # More precisely, the bilinear map: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor in A⊗B⊗C space: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k with coefficient... 
    # Actually the multiplication tensor is T = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    # where E_{i,j} is the matrix unit.
    # So the tensor has 1s at positions (i,j,k) in the "triple index" sense.
    # In the factorization: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m} (indexed by (i,j)), v_r in R^{m*p} (indexed by (j,k)), w_r in R^{p*n} (indexed by (k,i))
    # and the constraint is: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1 for all i,j,k
    # Wait, that's not right either. Let me think more carefully.
    
    # The bilinear map is: (A, B) -> C where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # As a tensor: C[i,k] = sum_{j} A[i,j] B[j,k]
    # We want rank-r decomposition: C[i,k] = sum_r U[i,j,r] ... 
    
    # Standard formulation: The matrix multiplication tensor M_{n,m,p} can be decomposed as:
    # M = sum_{r=1}^R sum_{(i,j,k) in S_r} E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
    # where each term in the sum over r contributes a product of matrices.
    
    # Actually, let me use the direct formulation:
    # We need: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[k',i'] ... 
    
    # Let me re-derive. The output is C of size n×p (or p×n per the W convention).
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has indices (k*n+i, r) where k is in [0,p) and i is in [0,n).
    # This suggests C has shape p×n? Or the indexing is C[k,i]?
    
    # Looking at the schoolbook: W[k*n+i][column] = 1, and U[i*m+j], V[j*p+k]
    # So the tensor relation is: for each (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal 1
    # This means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # So we need to find rank-r decompositions where:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)
    
    # This is a 3-way tensor completion / low-rank factorization problem.
    # The "target" tensor T has T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes).
    # Wait no - the target is that the PRODUCT of the three vectors gives 1.
    
    # Actually this is exactly: find U, V, W such that for all i,j,k:
    # <U_row(i,j), V_row(j,k), W_row(k,i)> (trilinear product) = 1
    
    # Let me try a random search approach for lower rank.
    # Start with the schoolbook and try to merge terms or find a compressed representation.
    
    # Alternative: Try to solve the system directly for a given rank.
    # For rank r, we have r variables in each of U (n*m*r), V (m*p*r), W (p*n*r).
    # Constraint: for each (i,j,k): sum_{a=1}^r U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a] = 1
    # That's n*m*p = 72 scalar equations.
    # Total unknowns: r*(n*m + m*p + p*n) = r*(9 + 24 + 24) = 57r
    # For r >= 2, we have more unknowns than equations, so solutions may exist.
    # But the system is highly nonlinear (quadratic in each variable set).
    
    # Let's try an iterative approach: fix two factors, solve for the third.
    # Given V and W, for each (i,j,k): U[i*m+j, :] must satisfy
    # sum_a U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a] = 1
    # This is: U[i*m+j, :] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 (element-wise product then dot)
    # So U[i*m+j, :] is determined by: U[i*m+j, :] @ d_{j,k,i} = 1
    # where d_{j,k,i}[a] = V[j*p+k,a] * W[k*n+i,a]
    
    # This gives us n*m*p = 72 linear constraints on the n*m*r = 9r variables of U.
    # Similarly for V and W.
    
    # Let's try ALS:
    def try_als(max_rank, n_iters=100, n_restarts=5):
        """Try ALS optimization to find a valid decomposition."""
        best = None
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            r = max_rank
            # Random initialization with small values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                # This is a system: for fixed (i,j), we have p equations in r unknowns
                # A_{i,j} @ u = b where A_{i,j}[k,a] = V[j*p+k,a] * W[k*n+i,a], b_{i,j}[k] = 1
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        # Solve least squares
                        U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Update V: for each (j,k), V[j*p+k,:] @ (U[i*m+j,:] * W[k*n+i,:]) = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Update W: for each (k,i), W[k*n+i,:] @ (U[i*m+j,:] * V[j*p+k,:]) = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                if err < 1e-10:
                    # Found a solution
                    if best is None or r < best[0]:
                        best = (r, U.copy(), V.copy(), W.copy())
                    break
            
            if best is not None and best[0] < max_rank:
                break
        
        return best
    
    # Try ranks from low to high
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually rank >= max(n*m, m*p, p*n) is not a lower bound.
    # rank >= n*m*p / min(n*m, m*p, p*n) = 72/9 = 8 (very weak)
    # Better lower bound: rank >= max(n, m, p) * something
    # For 3×3×3, known rank is 23.
    # For 3×3×8, rank is at most 3*3*8=72 (schoolbook)
    # Could be lower. Let's try a few ranks.
    
    # First, let's try rank 27 (3x the 3x3x3 rank of 9... no)
    # Actually for 3x3x3, schoolbook is 27, best known is 23.
    # For 3x3x8, we can think of it as 8 copies of 3x3x1, or 2 copies of 3x3x3 plus one 3x3x2.
    # rank(3x3x8) <= rank(3x3x3) * 2 + rank(3x3x2) 
    # rank(3x3x2) <= 3*3*2 = 18, or better: rank(3x3x2) <= rank(3x3x3) = 23? No.
    # rank(3x3x2) <= 3*2*3 = 18 (schoolbook)
    # So rank(3x3x8) <= 2*23 + 18 = 64 (if we can use rank-23 for 3x3x3)
    # But we don't have the rank-23 coefficients hardcoded.
    
    # Let's just try to find any decomposition with rank < 72 using ALS.
    # Start with rank 36 (half of 72) and go up if needed.
    
    for trial_rank in [36, 45, 54, 63, 71]:
        if time.time() > deadline:
            break
        result = try_als(trial_rank, n_iters=50, n_restarts=3)
        if result is not None:
            r, U_sol, V_sol, W_sol = result
            # Convert to list of lists
            U_list = U_sol.tolist()
            V_list = V_sol.tolist()
            W_list = W_sol.tolist()
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_list[i*m+j][a] * V_list[j*p+k][a] * W_list[k*n+i][a] for a in range(r))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid and r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_list, V_list, W_list
                break  # Found a good one, stop
    
    # If ALS didn't find anything better, return schoolbook
    if best_U is None:
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
