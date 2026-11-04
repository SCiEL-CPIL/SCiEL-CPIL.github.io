"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, n_tests=10):
    """Verify that the decomposition correctly computes C = A*B."""
    np.random.seed(42)
    for _ in range(n_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_reconstructed = np.zeros((n, p))
        for r in range(U.shape[1]):
            a_vals = np.zeros(n)
            b_vals = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    a_vals[i] += U[i * m + j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    b_vals[k] += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_reconstructed[i, k] += W[k * n + i, r] * a_vals[i] * b_vals[k]
        if not np.allclose(C_reconstructed, C_expected, atol=1e-8):
            return False
    return True

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, time_budget=30):
    """
    Use Alternating Least Squares to find a rank-r decomposition.
    Returns (U, V, W) or None if failed.
    """
    start = time.time()
    
    # Target: for all i,k,i',j',k':
    # sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j'*p+k', r] = delta(i',i) * delta(k',k)
    # 
    # This is equivalent to: the 3-way tensor T with T[i*m+j, j*p+k, k*n+i] 
    # defined by the consistency condition must be decomposed.
    #
    # Actually, let's reformulate. We need:
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r W[k*n+i,r] * (sum_{i',j'} U[i*m+j,r]*A[i',j']) * (sum_{j',k'} V[j*p+k,r]*B[j',k'])
    #
    # For this to hold for all A,B, we need:
    # sum_r W[k*n+i,r] * U[i'*m+j',r] * V[j'*p+k',r] = 1 if i'=i and k'=k, else 0
    #
    # Let's define the target tensor. The constraint is on the 3-way tensor with
    # indices (a, b, c) where a=i'*m+j', b=j'*p+k', c=k'*n+i.
    # But a,b,c are not independent. Let's enumerate all valid triples.
    
    # Build the target tensor as a list of (a, b, c, value) tuples
    # where value = 1 if i'=i and k'=k, else 0
    # But we need to be careful: for a given (a,b,c), we can extract:
    # i' = a//m, j' = a%m
    # j'_b = b//p, k' = b%p
    # k'_c = c//n, i = c%n
    # The constraint is: sum_r U[a,r]*V[b,r]*W[c,r] = 1 if (i'==i and k'==k) and (j'==j'_b), else 0
    # Wait, we also need j'==j'_b for consistency? Actually no, the sum is over r,
    # and U[a,r] and V[b,r] are defined for any a,b, but the physical meaning requires
    # j' from a to equal j' from b. However, the mathematical condition must hold for
    # ALL a,b,c, not just consistent ones.
    
    # Let me re-derive. We need:
    # For all i,k: C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,k] = sum_r W[k*n+i,r] * M_r
    # where M_r = (sum_{i',j'} U[i*m+j,r] * A[i',j']) * (sum_{j',k'} V[j*p+k,r] * B[j',k'])
    #
    # Expanding M_r:
    # M_r = sum_{i',j',j',k'} U[i*m+j,r] * V[j*p+k,r] * A[i',j'] * B[j',k']
    #
    # So C[i,k] = sum_r sum_{i',j',j',k'} W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] * A[i',j'] * B[j',k']
    #
    # We need this to equal sum_j A[i,j]*B[j,k] for all A,B.
    # So the coefficient of A[i',j']*B[j',k'] in C[i,k] must be:
    # 1 if i'=i and k'=k (and j'=j for some j, but actually j' is free)
    # Wait, A[i',j']*B[j',k'] contributes to C[i,k] iff i'=i and k'=k, regardless of j'.
    # But in the sum for C[i,k], j is summed over, so A[i,j]*B[j,k] for each j.
    # So the coefficient of A[i',j']*B[j',k'] in C[i,k] is 1 if i'=i and k'=k, else 0.
    # (j' can be anything; if i'=i and k'=k, then A[i',j']*B[j',k'] = A[i,j']*B[j',k] which is
    #  one of the terms in the sum for C[i,k] only if j'=j for the specific j, but since j is
    #  summed, it's included for any j'.)
    
    # So the condition is:
    # sum_r W[k*n+i,r] * U[i'*m+j',r] * V[j'*p+k',r] = 1 if i'==i and k'==k, else 0
    # for all i,k,i',j',k'.
    
    # Note: j' appears in both U and V indices but is not constrained to be equal between them.
    # This is a 3-way tensor decomposition problem.
    
    # Let's build the target. The number of constraints is n*p*n*m*p = 3*7*3*3*7 = 1323.
    # But many are redundant. Let's just solve the system directly.
    
    # Actually, for ALS, we need to minimize:
    # sum_{i,k,i',j',k'} (sum_r W[k*n+i,r]*U[i'*m+j',r]*V[j'*p+k',r] - target[i,k,i',j',k'])^2
    # where target = 1 if i'==i and k'==k, else 0.
    
    # This is expensive. Let's use a different approach: use the fact that the condition
    # must hold for all A,B, which gives us a bilinear constraint.
    
    # Alternative: use random A,B tests and minimize ||C_reconstructed - C_expected||^2.
    
    # Initialize
    np.random.seed(0)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Update U: fix V, W
        # For each column r, U[:,r] should satisfy:
        # sum_{i,k,i',j',k'} W[k*n+i,r] * V[j'*p+k',r] * U[i'*m+j',r] * target[i,k,i',j',k'] 
        # But this is coupled. Let's use gradient descent on the loss.
        
        # Loss = sum_{tests} ||C_rec - C_exp||^2
        # dL/dU = 2 * sum_{tests} (C_rec - C_exp)^T * dC_rec/dU
        
        # This is getting complex. Let's use a simpler approach: 
        # Project onto the constraint using least squares.
        
        # For fixed V, W, the optimal U minimizes:
        # sum_{i,k,i',j',k'} (sum_r W[k*n+i,r]*U[i'*m+j',r]*V[j'*p+k',r] - target)^2
        #
        # This is a linear system in U. Let's set it up.
        
        # Actually, let's just do gradient descent with a small step size.
        
        # Compute gradient
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_exp = A_test @ B_test
        C_rec = np.zeros((n, p))
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        a_vals = np.zeros((n, rank))
        b_vals = np.zeros((p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    a_vals[i, r] += U[i * m + j, r] * A_test[i, j]
            for k in range(p):
                for j in range(m):
                    b_vals[k, r] += V[j * p + k, r] * B_test[j, k]
            for i in range(n):
                for k in range(p):
                    C_rec[i, k] += W[k * n + i, r] * a_vals[i, r] * b_vals[k, r]
        
        err = C_rec - C_exp
        
        # Gradient with respect to U
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    # dC_rec[i,k]/dU[i*m+j,r] = W[k*n+i,r] * b_vals[k,r] for each k
                    # dL/dU[i*m+j,r] = 2 * sum_k err[i,k] * W[k*n+i,r] * b_vals[k,r]
                    for k in range(p):
                        dU[i * m + j, r] += 2 * err[i, k] * W[k * n + i, r] * b_vals[k, r]
        
        # Gradient with respect to V
        for r in range(rank):
            for j in range(m):
                for k in range(p):
                    # dC_rec[i,k]/dV[j*p+k,r] = W[k*n+i,r] * a_vals[i,r] for each i
                    # dL/dV[j*p+k,r] = 2 * sum_i err[i,k] * W[k*n+i,r] * a_vals[i,r]
                    for i in range(n):
                        dV[j * p + k, r] += 2 * err[i, k] * W[k * n + i, r] * a_vals[i, r]
        
        # Gradient with respect to W
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    # dC_rec[i,k]/dW[k*n+i,r] = a_vals[i,r] * b_vals[k,r]
                    # dL/dW[k*n+i,r] = 2 * err[i,k] * a_vals[i,r] * b_vals[k,r]
                    dW[k * n + i, r] = 2 * err[i, k] * a_vals[i, r] * b_vals[k, r]
        
        # Update
        lr = 0.01
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        err_val = np.sum(err**2)
        if err_val < best_err:
            best_err = err_val
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err_val < tol:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin from 150s limit
    
    # Schoolbook solution: rank = n * m * p = 3 * 3 * 7 = 63
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    for try_rank in range(21, 63):
        if time.time() > deadline:
            break
        
        result = _als_decompose(n, m, p, try_rank, max_iter=100, time_budget=5)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify_decomposition(U_try, V_try, W_try, n, m, p, n_tests=5):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = try_rank
                break  # Found a valid decomposition, can stop
    
    # Round to half-integers if possible
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    if best_rank < 63:
        # Try rounding
        U_r = round_to_half(best_U)
        V_r = round_to_half(best_V)
        W_r = round_to_half(best_W)
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, n_tests=5):
            best_U, best_V, best_W = U_r, V_r, W_r
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
