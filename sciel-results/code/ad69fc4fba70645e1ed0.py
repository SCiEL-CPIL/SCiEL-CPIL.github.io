"""Find rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Index: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k] for the bilinear form
    
    # Actually, the condition is:
    # For the bilinear map (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This must equal sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] (coefficient of A[i,j]*B[j,k] in C[i,k]) must be 1
    # And the decomposition gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Wait, that's not right either. Let me reconsider.
    
    # The bilinear map is: f(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r decomposition means: f(A,B)[i,k] = sum_{c=1}^r (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # No, that's not standard either.
    
    # Standard tensor decomposition of matrix multiplication:
    # The multiplication tensor M has entries M[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k]
    # Actually, M is a 3-tensor with indices (i,j,k) where i is row of A/C, j is column of A/row of B, k is column of B/C
    # M[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    
    # Rank-r decomposition: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Using flat indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # And for "wrong" combinations, the sum should be 0. But wait, the tensor M only has 1s at valid (i,j,k) positions.
    
    # Actually, the multiplication tensor is: T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k]
    # For matrix multiplication, every (i,j,k) triple has T[i,j,k] = 1 because A[i,j]*B[j,k] always contributes to C[i,k].
    # So T is the all-ones tensor of shape (n, m, p)!
    
    # No wait, that's not right. The tensor represents the bilinear map. Let me use the standard formulation.
    
    # The matrix multiplication tensor T ∈ R^{(n*p) × (n*m) × (m*p)} where
    # T[i*k, i'*m+j', j'*p+k'] = delta[i,i'] * delta[k,k'] * delta[j,j']
    # Hmm, this is getting confusing. Let me use the direct approach.
    
    # We want to find U, V, W such that for all matrices A (n×m) and B (m×p):
    # C = A @ B = sum_c (U^T A_flattened) ⊗ (V^T B_flattened) * W
    # where the reconstruction is: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    # This must equal sum_j A[i,j] * B[j,k]
    
    # Expanding: sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_{j'} V[j'*p+k, c] * B[j',k])
    # = sum_c W[k*n+i, c] * sum_j sum_{j'} U[i*m+j, c] * V[j'*p+k, c] * A[i,j] * B[j',k]
    
    # For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta[j,j'] * delta[i,i'] * delta[k,k']
    # Wait, no. We need the coefficient of A[i,j] * B[j',k] to be delta[j,j'] (since j=j' for the product to contribute to C[i,k]).
    
    # So: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta[j, j'] for all i, j, j', k
    
    # This is the key constraint. Let me verify with schoolbook:
    # U[i*m+j, c] = 1 if c corresponds to (i,j), else 0
    # V[j*p+k, c] = 1 if c corresponds to (j,k), else 0  
    # W[k*n+i, c] = 1 if c corresponds to (i,k), else 0
    # For schoolbook, c = i*m*p + j*p + k
    # Then sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = 1 only when c matches all three, i.e., j=j'
    # So it gives delta[j,j']. Correct!
    
    # So the constraint is: for all i, j, j', k:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta[j, j']
    
    # This is a system of equations. Let me set it up.
    
    # Number of constraints: n * m * m * p = 3 * 3 * 3 * 4 = 108
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (9 + 12 + 12) = rank * 33
    
    # For rank=36 (schoolbook), we have 36*33 = 1188 unknowns, 108 constraints. Very underdetermined.
    # We want the smallest rank where a solution exists with rational entries.
    
    # Let me try to find solutions for decreasing rank.
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, generate schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = make_schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try to find lower rank solutions using optimization
    # We'll use a randomized approach: initialize with random values, optimize, then round
    
    def check_solution(U, V, W):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(U.shape[1]):
                            val += W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    def try_rank(r, num_trials=20, max_iter=2000):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            np.random.seed(42 + trial)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Simple gradient descent on the residual
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                # residual = sum_c W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c] - delta[j,jp]
                                residual = 0.0
                                for c in range(r):
                                    residual += W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c]
                                residual -= (1.0 if j == jp else 0.0)
                                
                                for c in range(r):
                                    dU[i*m+j, c] += residual * W[k*n+i, c] * V[jp*p+k, c]
                                    dV[jp*p+k, c] += residual * W[k*n+i, c] * U[i*m+j, c]
                                    dW[k*n+i, c] += residual * U[i*m+j, c] * V[jp*p+k, c]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 100 == 0:
                    # Check if close to solution
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for c in range(r):
                                        val += W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c]
                                    val -= (1.0 if j == jp else 0.0)
                                    max_err = max(max_err, abs(val))
                    if max_err < 1e-6:
                        # Try to round to rationals
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if check_solution(U_r, V_r, W_r):
                            return (U_r, V_r, W_r, r)
        
        return None
    
    # Try ranks from low to high
    for r in range(20, rank_sb + 1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, num_trials=10, max_iter=500)
        if result is not None:
            U, V, W, rank = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
