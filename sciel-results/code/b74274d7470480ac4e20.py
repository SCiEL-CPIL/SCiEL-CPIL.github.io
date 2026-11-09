import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS

def _validate(U, V, W):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n, m, p = N, M, P
    rank = U.shape[1]
    
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this is a valid matrix mult term, else 0
    # Actually for matrix multiplication C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes)
    # Wait no. The tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if the indices are valid, representing the bilinear form
    # Actually the standard tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not right either.
    
    # The matrix multiplication tensor is defined as:
    # For C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all valid i,j,k
    # And the decomposition means: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k
    
    # Wait, that would mean the tensor is all-ones, which has rank 1.
    # That can't be right.
    
    # Let me reconsider. The tensor for matrix multiplication n×m × m×p is:
    # T: (R^n ⊗ R^m) ⊗ (R^m ⊗ R^p) -> R^{n×p}
    # In index notation: T_{(i,j),(j',k)} = δ_{j,j'}
    # So T[i,j,j',k] = 1 if j==j', else 0
    
    # For our factorization with U (n*m × rank), V (m*p × rank), W (p*n × rank):
    # The output is indexed by (i,k) for C[i,k]
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal δ_{j,j'} summed over j,j'
    # Wait, I'm confusing myself.
    
    # Let me just check: for each i in [0,n), k in [0,p), the sum over j of A[i,j]*B[j,k]
    # should be recoverable. The tensor entry for the bilinear form is:
    # For each triple (i, j, k), the coefficient is 1 (since A[i,j]*B[j,k] is a term)
    # And the sum over j gives C[i,k].
    
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify with schoolbook:
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # So for term r = i*m+j: U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1*1*1 = 1
    # And all other terms are 0. So sum = 1. Correct.
    
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's compute the error
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err

def _build_schoolbook():
    """Build the schoolbook decomposition with rank n*m*p."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    
    return U, V, W

def _try_reduce_rank(U0, V0, W0, target_rank, max_iters=500, tol=1e-10, deadline=None):
    """Try to find a rank-target_rank decomposition starting from a rank-1 init."""
    n, m, p = N, M, P
    nm, mp, pn = n*m, m*p, p*n
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(3):
        if deadline and time.time() > deadline:
            break
            
        # Initialize with random vectors
        U = np.random.randn(nm, target_rank) * 0.1
        V = np.random.randn(mp, target_rank) * 0.1
        W = np.random.randn(pn, target_rank) * 0.1
        
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
                
            # Update U
            for r in range(target_rank):
                # U[:, r] should satisfy: sum_r U[:, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed r, we want U[:, r] to be as close as possible
                # This is a linear least squares problem
                # Let's build the constraint matrix
                # For each (i,j,k), we have: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is complex. Let's use a simpler approach.
                
                pass
            
            # Simpler: use alternating minimization
            # Fix V and W, solve for U
            # The equation is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in U.
            
            # Build the system: A_U * vec(U) = b
            # where A_U has one row per (i,j,k) triple
            num_triples = n * m * p
            A_U = np.zeros((num_triples, nm * target_rank))
            b = np.ones(num_triples)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            A_U[idx, (i*m+j)*target_rank + r] = V[j*p+k, r] * W[k*n+i, r]
                        idx += 1
            
            # Solve least squares
            try:
                sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U = sol.reshape(nm, target_rank)
            except:
                pass
            
            # Fix U and W, solve for V
            A_V = np.zeros((num_triples, mp * target_rank))
            b = np.ones(num_triples)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            A_V[idx, (j*p+k)*target_rank + r] = U[i*m+j, r] * W[k*n+i, r]
                        idx += 1
            
            try:
                sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = sol.reshape(mp, target_rank)
            except:
                pass
            
            # Fix U and V, solve for W
            A_W = np.zeros((num_triples, pn * target_rank))
            b = np.ones(num_triples)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            A_W[idx, (k*n+i)*target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
                        idx += 1
            
            try:
                sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W = sol.reshape(pn, target_rank)
            except:
                pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = max(err, abs(val - 1.0))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
    
    if best_U is not None and best_err < tol:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook()
    best_rank = N * M * P
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to reduce rank
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But actually the lower bound for matrix multiplication rank is more complex.
    # For n×m × m×p, the rank is at least max(n*m, m*p, p*n) / something...
    # Actually, the rank must be at least the maximum of the dimensions of the spaces.
    # U is nm × rank, so rank >= nm = 15
    # V is mp × rank, so rank >= mp = 25
    # W is pn × rank, so rank >= pn = 15
    # So rank >= 25.
    
    # Let's try ranks from 25 upward
    for target_rank in range(25, best_rank + 1):
        if time.time() > deadline:
            break
        
        U, V, W, err = _try_reduce_rank(U_school, V_school, W_school, target_rank, 
                                         max_iters=100, tol=1e-10, deadline=deadline)
        
        if U is not None:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition, stop
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
