"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 120  # 120s budget
    
    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    best_rank = rank_schoolbook
    
    # Known lower bound for 2x3 by 3x7:
    # The rank of matrix multiplication tensor [n,m,p] is at least max(n*m, m*p, p*n)
    # but more sophisticated bounds exist. For small cases, we can try to find
    # decompositions below schoolbook.
    
    # Strategy: Try to find a decomposition of rank r < 42 using a randomized
    # algebraic approach. We'll use the fact that if a rank-r decomposition exists,
    # then the tensor can be written as sum of r rank-1 tensors.
    
    # For matrix multiplication, the tensor T has entries T[i,j,k] = delta_{i,k} (when 
    # appropriate indexing). Let's set up the tensor properly.
    
    # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # has T[i, j, k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B 
    # contribute to C[i,k]. In our indexing:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for all (i,j,k), 
    #   sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... 
    # Wait, let me re-read the convention.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k*n + i, r) where k is the column of B (0..p-1) and i is the row of A (0..n-1).
    # The tensor entry for (i, j, k) should be 1 (since A[i,j]*B[j,k] contributes to C[i,k]).
    
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # No, that's not right either. Let me think again.
    
    # Standard tensor decomposition for matrix multiplication:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{i,k} in some sense... no.
    
    # Actually, the matrix multiplication tensor is defined such that:
    # For inputs X (size n*m), Y (size m*p), the output Z (size n*p) is given by:
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    
    # In tensor form, this is: sum_{i,j,k} T[i,j,k] X[i,j] Y[j,k] = Z[i,k]
    # So T[i,j,k] is 1 if the indices match appropriately, but actually:
    # Z[i,k] = sum_j X[i,j] Y[j,k], so the tensor has T[i,j,k] = delta_{i,k}... 
    # No, that's not right. Let me be more careful.
    
    # The bilinear map is: (X, Y) -> Z where Z[i,k] = sum_j X[i,j] Y[j,k]
    # This is a trilinear form: sum_{i,j,k} X[i,j] Y[j,k] Z^*[i,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) where the indices 
    # are consistent, but actually T is defined on the product space.
    
    # Let me use the standard formulation. The matrix multiplication tensor 
    # <n,m,p> has rank r if and only if there exist matrices U (n*m x r), 
    # V (m*p x r), W (n*p x r) such that for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[i*p+k, c]... 
    
    # Hmm, I'm getting confused. Let me look at the schoolbook solution again.
    # In the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # The product U[i*m+j, c] * V[j*p+k, c] should equal the tensor entry.
    
    # For the schoolbook, for a fixed (i,j,k), there's exactly one column c where 
    # U[i*m+j, c]=1 and V[j*p+k, c]=1, and that column has W[k*n+i, c]=1.
    # So the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes).
    
    # Wait, that can't be right because then the tensor would just be all ones.
    
    # Let me reconsider. The matrix multiplication tensor is such that:
    # T[i, j, k] = 1 if and only if the indices are "compatible" in the sense that
    # A[i,j] and B[j,k] multiply to give a contribution to C[i,k].
    
    # Actually, I think the correct interpretation is:
    # The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 for all i,j,k (since every pair (i,j) and (j,k) contributes).
    
    # But then the decomposition would be:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]... no, W is a separate factor.
    
    # I think the correct formulation is that the tensor is:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n.
    # And the entry T[i*m+j, j*p+k, k*n+i] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # For matrix multiplication, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    
    # For the schoolbook, r = n*m*p and it's trivial. We want to find smaller r.
    
    # Let's try a randomized approach: pick random U and V, then solve for W.
    # But W appears multiplied, so this is nonlinear.
    
    # Alternative: use the fact that for rank r, we can write:
    # For each (i,k), the vector (V[j*p+k, :] for j) dotted with (U[i*m+j, :] for j) 
    # should give W[k*n+i, :].
    
    # Actually, let's try a different approach. We'll use the following:
    # Assume a rank-r decomposition exists. We can try to find it by:
    # 1. Randomly initialize U and V
    # 2. For each (i,k), compute the required W[k*n+i, :] such that
    #    sum_j U[i*m+j, :] * V[j*p+k, :] = W[k*n+i, :] (element-wise)
    #    But this doesn't work because W is shared across j.
    
    # Let me think differently. The condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to: for all i, k:
    # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = m (since there are m terms in j, each contributing 1)
    
    # Wait, no. For fixed i and k, summing over j:
    # sum_j T[i*m+j, j*p+k, k*n+i] = m (since each T entry is 1)
    # And sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = m
    
    # This is getting complicated. Let me just try to use a numerical optimization approach
    # to find a low-rank decomposition, then round to rational numbers.
    
    # Actually, for such small dimensions, let's try to use the following known result:
    # The rank of the 2x3 by 3x7 matrix multiplication tensor is known or can be bounded.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Upper bound: n*m*p = 42
    
    # Let's try to find a decomposition of rank 21 or higher but less than 42.
    
    # I'll use a simple approach: try rank r from 21 to 41, and for each, try to find
    # a decomposition using random initialization and gradient descent.
    
    # Given the time constraint, let's be strategic. We'll try a few ranks.
    
    def try_rank_decomposition(r, max_iterations=1000, tol=1e-10):
        """Try to find a rank-r decomposition using optimization."""
        if r < 21:  # Lower bound
            return None
        
        np.random.seed(42)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # We want to minimize the error: for all i,j,k, 
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
        
        # This is a non-convex optimization problem. Let's use a simple approach:
        # alternate between fixing two factors and solving for the third.
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Fix U and V, solve for W
            # For each (i,k), we have:
            # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = m
            # This is linear in W if we define:
            # A_{(i,k), c} = sum_j U[i*m+j, c] * V[j*p+k, c]
            # Then sum_c A_{(i,k), c} * W[k*n+i, c] = m
            
            A = np.zeros((n * p, r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        s = 0.0
                        for j in range(m):
                            s += U[i * m + j, c] * V[j * p + k, c]
                        A[i * p + k, c] = s
            
            # Now solve A @ W.T = m * ones(n*p, 1)
            # W has shape (p*n, r), so W.T has shape (r, p*n)
            # We need A @ W.T = m * ones(n*p)
            # This is underdetermined if r < n*p, which it is (r <= 41, n*p = 14)
            # Wait, A is (n*p, r) = (14, r), and we want A @ W.T = m * ones(14)
            # W.T is (r, p*n) = (r, 14)
            # So A @ W.T is (14, 14), but we want it to be m * ones(14, 1)?
            
            # I think I made an error. Let me reconsider.
            
            # The condition is: for all i, j, k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is n*m*p = 42 equations.
            # Unknowns: r*(n*m + m*p + p*n) = r*(6 + 21 + 14) = 41*r
            
            # For r=21, we have 42 equations and 861 unknowns, so it's underdetermined.
            
            # Let's set up the full system.
            # For each (i,j,k), we have one equation:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is nonlinear in U, V, W. Let's use a different approach.
            
            # I'll use the following: fix U and V, and for each (i,k), compute
            # the vector b_{i,k} where b_{i,k, c} = sum_j U[i*m+j, c] * V[j*p+k, c]
            # Then we need: sum_c b_{i,k, c} * W[k*n+i, c] = m
            # This is a linear system in W for each (i,k), but W is shared.
            
            # Actually, for fixed (i,k), the equation is:
            # b_{i,k} dot W[k*n+i, :] = m
            # This is one equation in r unknowns (the components of W[k*n+i, :]).
            
            # We have n*p = 14 such equations, each in r unknowns, but the unknowns 
            # are different for different (i,k) since W[k*n+i, :] is different.
            # So we have 14 independent linear equations, each in r unknowns, with 
            # different unknowns for each equation. This is trivially solvable!
            
            # For each (i,k), we can choose W[k*n+i, :] to satisfy 
            # b_{i,k} dot W[k*n+i, :] = m, provided b_{i,k} is not zero.
            
            # So the optimization reduces to: find U and V such that for all (i,k),
            # b_{i,k} = sum_j U[i*m+j, :] * V[j*p+k, :] (element-wise product summed over j)
            # is not zero, and then set W accordingly.
            
            # But we also need the original condition to hold for all (i,j,k), not just 
            # summed over j. Let me re-examine.
            
            # The original condition is for each (i,j,k):
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # If we fix U and V, and define b_{i,k, c} = sum_j U[i*m+j, c] * V[j*p+k, c],
            # then summing over j gives:
            # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = m
            # which is b_{i,k} dot W[k*n+i, :] = m
            
            # But the individual conditions for each j are not automatically satisfied.
            # We need: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is a stronger condition. Let's define:
            # a_{i,j,k, c} = U[i*m+j, c] * V[j*p+k, c]
            # Then we need: for all i,j,k: sum_c a_{i,j,k, c} * W[k*n+i, c] = 1
            
            # For fixed (i,k), this is a system of m equations in r unknowns:
            # For j=0,...,m-1: sum_c a_{i,j,k, c} * W[k*n+i, c] = 1
            
            # This is a linear system: A_{i,k} @ W[k*n+i, :]^T = ones(m)
            # where A_{i,k} is an m x r matrix with A_{i,k, j, c} = a_{i,j,k, c}
            
            # For this to have a solution, we need ones(m) to be in the column space of A_{i,k}.
            # Since r >= m (for r >= 3), this is generically true.
            
            # So the algorithm is:
            # 1. Randomly initialize U and V
            # 2. For each (i,k), solve A_{i,k} @ W[k*n+i, :]^T = ones(m)
            # 3. Check if the original condition is satisfied
            # 4. If not, adjust U and V
            
            # Let's implement this.
            
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.zeros((p * n, r))
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 5:
                    break
                
                # For each (i,k), solve for W[k*n+i, :]
                valid = True
                for i in range(n):
                    for k in range(p):
                        # Build A_{i,k}
                        A_ik = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        
                        # Solve A_ik @ w = ones(m)
                        try:
                            w, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                            W[k * n + i, :] = w
                            if len(residuals) > 0 and residuals[0] > 1e-8:
                                valid = False
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    # Perturb U and V
                    U += np.random.randn(n * m, r) * 0.01
                    V += np.random.randn(m * p, r) * 0.01
                    continue
                
                # Check the original condition
                max_error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            max_error = max(max_error, abs(s - 1.0))
                
                if max_error < tol:
                    # Found a solution! Convert to fractions
                    U_frac = [[Fraction(U[i, c]).limit_denominator(1000) for c in range(r)] for i in range(n * m)]
                    V_frac = [[Fraction(V[i, c]).limit_denominator(1000) for c in range(r)] for i in range(m * p)]
                    W_frac = [[Fraction(W[i, c]).limit_denominator(1000) for c in range(r)] for i in range(p * n)]
                    
                    # Verify with fractions
                    max_error_frac = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for c in range(r):
                                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                                max_error_frac = max(max_error_frac, abs(float(s - 1)))
                    
                    if max_error_frac < 1e-6:
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
                
                # Perturb U and V for next iteration
                U += np.random.randn(n * m, r) * 0.01
                V += np.random.randn(m * p, r) * 0.01
        
        return None
    
    # Try ranks from 21 to 41
    for r in range(21, min(42, rank_schoolbook)):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_decomposition(r, max_iterations=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
