"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta[i,k] (for A[i,j]*B[j,k])."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0 if i == k else 0.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Build tensor from decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Actually, the standard decomposition is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, let me reconsider. The tensor T[i,j,k] such that (A*B)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # T[i,j,k] = 1 if j is the connecting index, which is always true in standard matmul.
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    # Let me re-read the task. "Matrix multiplication 3,4,8"
    # Multiply 3x4 by 4x8. Output is 3x8.
    # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 (it's the identity in the j dimension)
    # But that's just the standard matmul, which has rank n*m*p for the trivial decomposition.
    
    # Actually, for matrix multiplication A (n x m) * B (m x p) -> C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_c U[i,c] V[c,k] where U depends on A, V depends on B
    # But the tensor decomposition form is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that doesn't seem right. Let me think again.
    
    # The standard rank decomposition of the matmul tensor:
    # The tensor T has entries T[i,j,k] = delta_{j,j} = 1? No.
    # T[i,j,k] represents the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # For standard matmul, T[i,j,k] = 1 for all i,j,k? That would give C[i,k] = sum_j A[i,j]B[j,k] which is correct.
    # But that's not a tensor decomposition in the usual sense.
    
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # This suggests the decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # Hmm, that doesn't look standard either.
    
    # Actually, I think the convention is:
    # Define intermediate vectors:
    # a_r = sum_j U[i*m+j, r] * A[i,j]  (for fixed i, r)
    # b_r = sum_j V[j*p+k, r] * B[j,k]  (for fixed k, r)
    # Then C[i,k] = sum_r a_r * b_r * W[k*n+i, r]
    # But a_r depends on i and r, b_r depends on k and r.
    # So C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k])
    
    # For this to equal sum_j A[i,j] B[j,k] for all A, B:
    # We need: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j' V[j'*p+k,r] B[j',k]) = sum_j A[i,j] B[j,k]
    
    # Let me expand: sum_r W[k*n+i,r] * sum_j U[i*m+j,r] A[i,j] * sum_j' V[j'*p+k,r] B[j',k]
    # = sum_j sum_j' A[i,j] B[j',k] * sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r]
    
    # For this to equal sum_j A[i,j] B[j,k] (note j=j' in the target), we need:
    # sum_r W[k*n+i,r] U[i*m+j,r] V[j'*p+k,r] = delta_{j,j'} for all i,k,j,j'
    
    # So the condition is: for all i, k, j, j':
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
    
    # This is the tensor T[i,j,j',k] = delta_{j,j'} which factors through the r dimension.
    
    # Let me verify with the schoolbook:
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k) triple, else 0
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k) triple, else 0
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k) triple, else 0
    # c indexes (i,j,k) triples.
    
    # sum_r U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r]
    # = sum_{(i',j',k')} [U[i*m+j, (i',j',k')]] [V[j'*p+k, (i',j',k')]] [W[k*n+i, (i',j',k')]]
    # U[i*m+j, (i',j',k')] = 1 if i'=i and j'=j
    # V[j'*p+k, (i',j',k')] = 1 if j'=j' (the j' in V's index) and k'=k
    # Wait, V is indexed by [j*p+k, c] where c=(i',j'',k''). V[j*p+k, c] = 1 if j=j'' and k=k''.
    # So V[j'*p+k, (i',j'',k'')] = 1 if j'=j'' and k=k''.
    # W[k*n+i, (i',j'',k'')] = 1 if k=k'' and i=i''.
    
    # So the product is 1 iff: i'=i, j'=j (from U), j'=j'', k'=k (from V), k'=k, i'=i (from W)
    # So i'=i, j'=j, j''=j, k'=k. And the sum is over all c=(i',j'',k'').
    # The only c that satisfies all: i'=i, j''=j, k'=k. And we also need j'=j'' which is j=j, always true.
    # So sum = 1 if j=j' (since j'=j from U constraint and j''=j from V constraint, and we need j'=j'').
    # Wait, I'm confusing myself.
    
    # Let me redo: The sum is over r (which indexes c=(i',j'',k'')).
    # U[i*m+j, r] is nonzero only when r corresponds to (i,j,k') for some k', i.e., i'=i, j''=j.
    # V[j'*p+k, r] is nonzero only when r corresponds to (i',j',k) for some i', i.e., j''=j', k'=k.
    # W[k*n+i, r] is nonzero only when r corresponds to (i,j'',k) for some j'', i.e., i'=i, k'=k.
    
    # So all three are nonzero only when: i'=i, j''=j, j''=j', k'=k.
    # This requires j=j'. If j=j', then there's exactly one r: (i,j,k), and the product is 1.
    # If j!=j', no r satisfies all, so sum is 0.
    # Therefore sum_r ... = delta_{j,j'}. Correct!
    
    # So the verification condition is:
    # For all i, k, j, j': sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'}
    
    # Let's implement this check.
    
    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    T_target[i, j, jp, k] = 1.0 if j == jp else 0.0
    
    # Compute T_decomposed[i, j, jp, k] = sum_r U[i*m+j, r] V[jp*p+k, r] W[k*n+i, r]
    T_dec = np.zeros((n, m, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        T_dec[i, j, jp, k] += U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
    
    if not np.allclose(T_dec, T_target, atol=tol):
        return False
    
    return True

def _try_rank(rank, n, m, p, time_limit=30.0):
    """Try to find a decomposition of given rank."""
    start_time = time.time()
    
    # Initialize with random values
    np.random.seed(rank)  # Deterministic for reproducibility
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Gradient descent to minimize ||T_dec - T_target||^2
    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    T_target[i, j, jp, k] = 1.0 if j == jp else 0.0
    
    lr = 0.01
    for step in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        # T_dec[i,j,jp,k] = sum_r U[i*m+j,r] V[jp*p+k,r] W[k*n+i,r]
        # dL/dU[i*m+j,r] = sum_{jp,k} (T_dec[i,j,jp,k] - T_target[i,j,jp,k]) * V[jp*p+k,r] * W[k*n+i,r]
        # dL/dV[jp*p+k,r] = sum_{i,j} (T_dec[i,j,jp,k] - T_target[i,j,jp,k]) * U[i*m+j,r] * W[k*n+i,r]
        # dL/dW[k*n+i,r] = sum_{i,j,jp} (T_dec[i,j,jp,k] - T_target[i,j,jp,k]) * U[i*m+j,r] * V[jp*p+k,r]
        
        # Compute T_dec
        T_dec = np.zeros((n, m, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            T_dec[i, j, jp, k] += U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
        
        residual = T_dec - T_target
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            dU[i*m+j, r] += residual[i, j, jp, k] * V[jp*p+k, r] * W[k*n+i, r]
                            dV[jp*p+k, r] += residual[i, j, jp, k] * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += residual[i, j, jp, k] * U[i*m+j, r] * V[jp*p+k, r]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if np.max(np.abs(residual)) < 1e-8:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    # Final check
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    # But for matmul, the known bounds are more complex.
    # Let's try ranks from a reasonable lower bound up to schoolbook.
    
    lower_bound = max(n*m, m*p, n*p)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:  # Safety margin
            break
        
        result = _try_rank(rank, n, m, p, time_limit=30.0)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            # If we found a valid decomposition, we can stop early if rank is close to lower bound
            if rank <= lower_bound + 2:
                break
    
    return best_result
# EVOLVE-BLOCK-END
