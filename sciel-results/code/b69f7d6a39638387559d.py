"""Rewritten program: uses a random search over small-rank tensor decompositions
for the 3x3x7 matrix multiplication tensor, seeded with the schoolbook solution
and perturbed with random rank-1 updates, then verifies and returns the best valid
decomposition found."""

import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 63

    # Build the schoolbook tensor as a 3D array T[i, j, k] = 1 for all i,j,k
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # T_{i,j,k} such that C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T is a 3D tensor of shape (n, m, p) with T[i,j,k] = delta_{i,k} ... no wait.
    # Actually, the multiplication tensor T has entries T[i,j,k] such that
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i, j, k] = 1 if we're looking at the "identity" tensor.
    # More precisely, the tensor is T where T[i,j,k] = 1 for all valid i,j,k? No.
    # The multiplication tensor is defined such that <T, A ⊗ B> = <C, A*B>
    # In coordinates: T[i, j, k] where i indexes rows of A (and C), j indexes columns of A / rows of B,
    # k indexes columns of B (and C).
    # T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] appears in C[i,k], which is always true.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. The tensor T is such that:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Actually, I need to be more careful. The tensor decomposition expresses:
    # C[i,k] = sum_r U[i,j]... no.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no.
    # The tensor T has entries T[i, j, k] and the decomposition is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]? No, that's not standard.
    
    # Let me re-read the contract. The factor matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The decomposition must satisfy: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]? No, W is also indexed by r.
    
    # I think the convention is: the tensor T is decomposed as:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But that doesn't make sense dimensionally unless W is a scalar per r.
    
    # Let me look at the seed program more carefully. In the seed:
    # U[i*m + j][col] = 1
    # V[j*p + k][col] = 1  
    # W[k*n + i][col] = 1
    # And the claim is this decomposes the matrix multiplication tensor.
    
    # For schoolbook, T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    # With the above, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 when r corresponds to (i,j,k) and 0 otherwise.
    # So the sum is 1, which equals T[i,j,k] = 1. Yes!
    
    # So the tensor T[i,j,k] = 1 for all i,j,k, and we need:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a rank-r decomposition of the all-ones tensor of shape (n*m, m*p, p*n) but with a specific index mapping.
    # Actually, it's the tensor of shape (n, m, p) with all entries 1, viewed in the (nm, mp, pn) space.
    
    # Let's verify: for the schoolbook, each (i,j,k) gets its own rank-1 term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # Then for a given (i,j,k), the sum over r is 1 (only the r corresponding to (i,j,k) contributes).
    # For a different (i',j',k'), if (i',j',k') != (i,j,k), then no single r has all three non-zero... 
    # Wait, actually each r corresponds to a unique (i,j,k) triple, so for (i',j',k') != (i,j,k), 
    # the term with r=(i,j,k) has U[i'*m+j', r] = 0 unless i'=i and j'=j.
    # So yes, the sum is 1 for all (i,j,k). Good.
    
    # Now, to find a lower rank decomposition, we can use random search.
    # Start with the schoolbook and try to reduce rank by random linear combinations.
    
    # Strategy: Try to find a rank-r decomposition for r < 63.
    # We'll do a random search: initialize U, V, W with random values, then use gradient-free optimization
    # or just random perturbations and check validity.
    
    # Actually, for small tensors, we can try to find a decomposition directly.
    # The tensor is 9 x 21 x 21 (n*m=9, m*p=21, p*n=21).
    # We want rank r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r where u_r is in R^9, v_r in R^21, w_r in R^21.
    
    # Let's try a random search for rank 50, 55, 60, etc.
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        """Verify that the decomposition is valid."""
        n, m, p = n, m, p
        R = U.shape[1]
        # Build the tensor from the decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_random_decomposition(n, m, p, R, n_trials=200, seed=42):
        """Try to find a rank-R decomposition using random initialization and refinement."""
        rng = np.random.default_rng(seed)
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(n_trials):
            # Random initialization
            U = rng.standard_normal((nm, R))
            V = rng.standard_normal((mp, R))
            W = rng.standard_normal((pn, R))
            
            # Simple iterative refinement: alternate between fixing two matrices and solving for the third
            # This is a basic tensor ring / CP decomposition approach
            for it in range(100):
                # Fix V and W, solve for U
                # U[i*m+j, r] should satisfy: for all (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
                # This is a linear system for U given V, W
                # For each (i,j), we have: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is: U[i*m+j, :] @ diag(V[j*p+k, r] * W[k*n+i, r]) ... not quite.
                # Let's form a matrix M_{i,j} where M_{i,j}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then U[i*m+j, :] @ M_{i,j}.T = ones(p)
                # So U[i*m+j, :] = ones(p) @ M_{i,j}.T^{-1} if M is square and invertible
                # But M is p x R, so we need least squares
                
                for i in range(n):
                    for j in range(m):
                        # Build M of shape (p, R)
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve U[i*m+j, :] @ M.T = ones(p)
                        # i.e., M @ U[i*m+j, :].T = ones(p)
                        # Use least squares
                        try:
                            U[i*m+j, :], residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except np.linalg.LinAlgError:
                            pass
                
                # Similarly for V and W
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        except np.linalg.LinAlgError:
                            pass
                
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        except np.linalg.LinAlgError:
                            pass
                
                # Check convergence
                if verify(U, V, W, n, m, p, tol=1e-4):
                    return U, V, W
        
        return None
    
    # Try ranks from 55 down to 40
    best_U, best_V, best_W = None, None, None
    best_rank = rank_school
    
    for R in [55, 50, 45, 40]:
        result = try_random_decomposition(n, m, p, R, n_trials=50, seed=42+R)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
    
    # If no better decomposition found, return schoolbook
    if best_U is None:
        rank = rank_school
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
