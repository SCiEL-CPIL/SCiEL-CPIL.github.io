"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

The tensor T for multiplying an n×m matrix A by an m×p matrix B to get C = A*B
is defined by the bilinear map. A rank-R decomposition T = sum_r u_r ⊗ v_r ⊗ w_r
means:
  C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]

For this to equal C[i,k] = sum_j A[i,j]*B[j,k], we need:
  sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
  for all i, j, j', k.
"""

import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _verify_exact(U, V, W, n, m, p, tol=1e-8):
    """Verify the exact tensor condition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[jp*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - expected) > tol:
                        return False
    return True

def _verify_decomposition(U, V, W, n, m, p, num_tests=10):
    """Verify by testing random matrices."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    u_val = 0.0
                    for j in range(m):
                        u_val += U_arr[i*m+j, r] * A[i,j]
                    v_val = 0.0
                    for j in range(m):
                        v_val += V_arr[j*p+k, r] * B[j,k]
                    s += u_val * v_val * W_arr[k*n+i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-7):
            return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _alternating_least_squares(n, m, p, target_rank, max_iters=500, deadline=None, seed=42):
    """
    Alternating least squares for tensor decomposition.
    
    We want to find U (n*m x R), V (m*p x R), W (p*n x R) such that:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
    
    Fix W, solve for U and V via linear systems.
    Fix U and V, solve for W.
    Iterate.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Fix W, solve for U
        # For each (i, j): we need sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        # This is a linear system in U. For fixed i, j, the constraint is:
        #   For all j', k: sum_r U[i*m+j, r] * (V[j'*p+k, r] * W[k*n+i, r]) = delta_{j,j'}
        # Let a_{j',k,r} = V[j'*p+k, r] * W[k*n+i, r]
        # Then sum_r U[i*m+j, r] * a_{j',k,r} = delta_{j,j'}
        
        # This gives m*p equations for m unknowns (one per j) but actually for each (i,j)
        # we have m*p equations. We can solve this as a least squares problem.
        
        # Actually, let's think of it differently. For fixed i:
        # We need for all j, j', k:
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # Let's build the system. For each i, we have n*m*p constraints on n*m variables (U[i*m:(i+1)*m, :]).
        # Actually U[i*m+j, :] is a row of length target_rank.
        
        # Let's use a different approach: vectorize.
        # For fixed i, define M_i as a matrix where:
        #   M_i[(j',k), (j,r)] = V[j'*p+k, r] * W[k*n+i, r]
        # Then M_i @ U[i*m:(i+1)*m, :].T should give the right side.
        
        # The right side for (j',k) is: sum_j delta_{j,j'} * ... 
        # Wait, the constraint is: for each (j', k), sum_j sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j',j'}? No.
        
        # Let me re-derive. C[i,k] = sum_j A[i,j] B[j,k]
        # Using the decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j' V[j'*p+k,r] B[j',k]) W[k*n+i,r]
        # = sum_r sum_j sum_j' U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r] A[i,j] B[j',k]
        
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_r U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r] = delta_{j,j'} for all i, j, j', k.
        
        # So for fixed i, and for each (j', k), we need:
        #   sum_j sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = (1 if j'=j else 0)
        #   But wait, the RHS depends on j. Let me re-read.
        
        # The condition is: for all i, j, j', k:
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # So for fixed i, j', k, and varying j:
        #   For each j: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # This means: for fixed i, j', k, the vector (sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r])_{j=0}^{m-1}
        # should equal e_{j'} (the j'-th standard basis vector).
        
        # Let's define for fixed i:
        #   For each r, let vec_r be the vector with entries (U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r]) for all j, j', k.
        #   This is getting complicated. Let's use a matrix formulation.
        
        # For fixed i, define A_i as a matrix of shape (m*p, m*target_rank) where:
        #   A_i[(j',k), (j,r)] = V[j'*p+k, r] * W[k*n+i, r]
        # Then A_i @ vec(U[i*m:(i+1)*m, :].T) should give a vector of length m*p where
        #   entry (j',k) = sum_j sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r]
        # And we want this to equal: for each (j',k), the value is 1 if there exists j such that j=j' (always true for the j'=j term)...
        
        # Wait, I'm confusing myself. Let me just directly solve.
        
        # For fixed i, we have variables U[i*m+j, r] for j in [0,m), r in [0,R).
        # That's m*R variables.
        # Constraints: for each j', k:
        #   sum_j sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = ?
        
        # The RHS: we need sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} for each j.
        # So for fixed j', k, the constraint for a specific j is:
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # So for fixed i, j', k, we have m constraints (one per j):
        #   For j=0: sum_r U[i*m+0, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{0,j'}
        #   For j=1: sum_r U[i*m+1, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{1,j'}
        #   ...
        
        # This is m*R variables with m*p*m = m^2*p constraints. For m=3, p=4, that's 36 constraints on 3*R variables.
        # If R >= 12, we have more variables than constraints, so it's underdetermined.
        
        # Let's build the system for fixed i.
        # Variables: x_{j,r} = U[i*m+j, r] for j in [0,m), r in [0,R). Total m*R variables.
        # Constraints: for each (j', k, j):
        #   sum_r x_{j,r} * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # Number of constraints: m * p * m = m^2 * p
        # Number of variables: m * R
        
        # Build the matrix A of shape (m^2*p, m*R)
        # Row index: (j', k, j) -> j' * (p*m) + k * m + j
        # Col index: (j, r) -> j * R + r
        
        A_mat = np.zeros((m * m * p, m * target_rank))
        b_vec = np.zeros(m * m * p)
        
        for jp in range(m):
            for k in range(p):
                for j in range(m):
                    row = jp * (p * m) + k * m + j
                    for r in range(target_rank):
                        col = j * target_rank + r
                        A_mat[row, col] = V[jp * p + k, r] * W[k * n + i, r]
                    b_vec[row] = 1.0 if j == jp else 0.0
        
        # Solve least squares
        U_new_i, residuals, rank_l, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Reshape back
        U[i*m:(i+1)*m, :] = U_new_i.reshape(m, target_rank)
        
        # Now fix U, solve for V
        # For fixed j, k: we need sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} for all i, j'.
        # Variables: V[j'*p+k, r] for j' in [0,m), r in [0,R). Total m*R variables.
        # Constraints: for each (i, j, j'):
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # Wait, k is fixed in the V index. Let me re-think.
        # V has shape (m*p, R). The index is j'*p+k.
        # For fixed k, we have variables V[j'*p+k, r] for j' in [0,m), r in [0,R). That's m*R variables.
        # Constraints: for each i, j, j':
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # Number of constraints: n * m * m
        # Number of variables: m * R
        
        for k in range(p):
            A_mat_v = np.zeros((n * m * m, m * target_rank))
            b_vec_v = np.zeros(n * m * m)
            
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        row = i * (m * m) + j * m + jp
                        for r in range(target_rank):
                            col = jp * target_rank + r
                            A_mat_v[row, col] = U[i * m + j, r] * W[k * n + i, r]
                        b_vec_v[row] = 1.0 if j == jp else 0.0
            
            V_new_k, residuals, rank_l, sv = np.linalg.lstsq(A_mat_v, b_vec_v, rcond=None)
            V[k::p, :] = V_new_k.reshape(m, target_rank)
        
        # Now fix U and V, solve for W
        # For fixed k, i: we need sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} for all j, j'.
        # Variables: W[k*n+i, r] for r in [0,R). Total R variables.
        # Constraints: for each j, j':
        #   sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
        
        # Number of constraints: m * m
        # Number of variables: R
        
        for i in range(n):
            for k in range(p):
                A_mat_w = np.zeros((m * m, target_rank))
                b_vec_w = np.zeros(m * m)
                
                for j in range(m):
                    for jp in range(m):
                        row = j * m + jp
                        for r in range(target_rank):
                            A_mat_w[row, r] = U[i * m + j, r] * V[jp * p + k, r]
                        b_vec_w[row] = 1.0 if j == jp else 0.0
                
                W_new, residuals, rank_l, sv = np.linalg.lstsq(A_mat_w, b_vec_w, rcond=None)
                W[k * n + i, :] = W_new
        
        # Check convergence
        if it % 50 == 0:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_exact(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist()
    
    # Final check
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    if _verify_exact(U_list, V_list, W_list, n, m, p, tol=1e-6):
        return U_list, V_list, W_list
    
    return None

def _try_rank(n, m, p, target_rank, deadline=None):
    """Try to find a rank-target_rank decomposition using alternating least squares."""
    # Try multiple random initializations
    for seed in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        result = _alternating_least_squares(n, m, p, target_rank, max_iters=300, deadline=deadline, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify_exact(U, V, W, n, m, p, tol=1e-7):
                return U, V, W
    
    return None

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    start_time = time.time()
    deadline = start_time + 120  # Leave safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # The lower bound is max(n*m, m*p, n*p) = max(9, 12, 12) = 12
    # Try ranks from 12 up to 35
    for target_rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
