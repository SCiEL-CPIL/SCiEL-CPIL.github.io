"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]
DEADLINE = 120.0  # safety margin for evaluator kill at 150s

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _check_decomposition(U, V, W, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k,r] = delta_{i,r} if we're doing (n x m) @ (m x p)
    Actually the standard decomposition:
    (A @ B)[i,k] = sum_j A[i,j] B[j,k]
    
    Tensor form: T[i, j, k] = delta_{j, j} ... no.
    
    Let me reconsider. The matrix multiplication tensor for (n x m) @ (m x p) -> (n x p):
    T[i, j, k, l] = 1 if j==k and l==i... 
    
    Actually, the standard way:
    (A @ B)[i, k] = sum_j A[i, j] * B[j, k]
    
    So the tensor is: M[i, j, k] where we sum over j:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The bilinear form: C[i,k] = sum_r U[i, r] * V[r, k]... no.
    
    Let me think about this more carefully with the given interface:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    The trace convention W[k*n+i, r] suggests:
    The output C[i, k] (i-th row of first, k-th col of second) is computed as:
    C[i, k] = sum_r U[i*m + j, r] * ... 
    
    Actually, looking at the schoolbook:
    U[i*m+j, col] = 1  (index by row i and column j of A)
    V[j*p+k, col] = 1  (index by row j and column k of B)  
    W[k*n+i, col] = 1  (index by column k and row i of C, using trace convention)
    
    So the decomposition is:
    C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_r U[i*m+j, r] * V[j*p+k', r] * W[k*n+i, r] = delta_{k,k'} for each i,j
    
    More precisely, the tensor identity is:
    For all i, j, k, k': sum_r U[i*m+j, r] * V[j*p+k', r] * W[k*n+i, r] = delta_{k,k'}
    
    Wait, that's not right either. Let me re-derive.
    
    The matrix product C = A @ B where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In tensor decomposition form with rank r:
    C[i, k] = sum_s U[i, s] * V[k, s] ... no, that's for a specific structure.
    
    With the given shapes:
    U has n*m rows (indexed by (i,j) pairs), V has m*p rows (indexed by (j,k) pairs), W has p*n rows (indexed by (k,i) pairs).
    
    The bilinear algorithm:
    C[i,k] = sum_s sum_j U[(i,j), s] * B[j,k] ... 
    
    I think the correct interpretation is:
    The multiplication is expressed as:
    C[i,k] = sum_s (sum_j U[(i,j), s] * A[i,j]) * (sum_j V[(j,k), s] * B[j,k]) * W[(k,i), s]
    
    No wait, that doesn't work dimensionally. Let me look at the schoolbook again:
    
    For each (i,j,k) triple, we have one column s:
    U[i*m+j, s] = 1
    V[j*p+k, s] = 1  
    W[k*n+i, s] = 1
    
    So: C[i,k] = sum_s sum_j U[i*m+j, s] * A[i,j] * V[j*p+k, s] * B[j,k] * W[k*n+i, s]
    
    Hmm, but that has A and B mixed in. 
    
    Actually I think the standard formulation is:
    The tensor T[i, j, k] = delta_{j, j} is not quite right.
    
    Let me think of it as: we want to compute the bilinear map (A, B) -> C where
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition means we can write:
    C[i,k] = sum_{s=1}^r f_s(A) * g_s(B) * h_s(C)
    
    where f_s(A) is a linear functional of A, g_s(B) is a linear functional of B, and h_s is the coefficient for C[i,k].
    
    So: C[i,k] = sum_s h_s[i,k] * f_s(A) * g_s(B)
    
    where f_s(A) = sum_{i,j} U[i*m+j, s] * A[i,j]
    g_s(B) = sum_{j,k} V[j*p+k, s] * B[j,k]
    h_s[i,k] = W[k*n+i, s]
    
    For this to equal the matrix product:
    sum_s W[k*n+i, s] * (sum_{i',j} U[i'*m+j, s] A[i',j]) * (sum_{j',k'} V[j'*p+k', s] B[j',k'])
    = sum_j A[i,j] B[j,k]
    
    This must hold for all A, B. So the coefficient of A[i',j'] B[j'',k''] must match:
    sum_s W[k*n+i, s] U[i'*m+j', s] V[j''*p+k'', s] = delta_{i,i'} delta_{j,j'} delta_{j'',k'} delta_{k,k''}
    
    Wait, more carefully: the LHS is a function of (i,k) and the RHS is sum_j A[i,j]B[j,k].
    The coefficient of A[a,b] B[c,d] in the LHS (as a function of i,k) is:
    sum_s W[k*n+i, s] U[a*m+b, s] V[c*p+d, s]
    
    And in the RHS, the coefficient of A[a,b] B[c,d] is delta_{a,i} delta_{b,c} delta_{d,k}.
    
    So we need: for all a, b, c, d:
    sum_s W[k*n+i, s] U[a*m+b, s] V[c*p+d, s] = delta_{a,i} delta_{b,c} delta_{d,k}
    
    This is the condition to check.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Check the tensor identity
    # For random samples of (a,b,c,d), verify the identity
    # Full check would be n*m*m*p = 2*3*3*7 = 126 combinations
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    # Compute LHS: sum_s W[k*n+i, s] U[a*m+b, s] V[c*p+d, s] for each (i,k)
                    # This should equal delta_{a,i} delta_{b,c} delta_{d,k}
                    if b != c:
                        # Should be all zeros
                        for i in range(n):
                            for k in range(p):
                                val = np.dot(W[k*n+i, :], U[a*m+b, :] * V[c*p+d, :])
                                if abs(val) > tol:
                                    return False
                    else:
                        # Should be delta_{a,i} delta_{d,k}
                        for i in range(n):
                            for k in range(p):
                                expected = 1.0 if (a == i and d == k) else 0.0
                                val = np.dot(W[k*n+i, :], U[a*m+b, :] * V[c*p+d, :])
                                if abs(val - expected) > tol:
                                    return False
    return True


def _try_rank_decomposition(target_rank, max_iters=5000, seed=None):
    """Try to find a rank-r decomposition using random initialization and gradient descent."""
    n, m, p = DIMENSIONS
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    r = target_rank
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute gradients
        # Loss = sum over all (a,b,c,d,i,k) of (sum_s W[k*n+i,s] U[a*m+b,s] V[c*p+d,s] - delta_{a,i}delta_{b,c}delta_{d,k})^2
        # This is expensive to compute fully, so use a subset
        
        # Sample a subset of constraints
        num_samples = min(50, n * m * m * p)
        samples = []
        for _ in range(num_samples):
            a = random.randint(0, n-1)
            b = random.randint(0, m-1)
            c = random.randint(0, m-1)
            d = random.randint(0, p-1)
            samples.append((a, b, c, d))
        
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for (a, b, c, d) in samples:
            expected = 1.0 if (b == c) else 0.0
            # For this (a,b,c,d), the constraint is on W[k*n+i] for all i,k
            # sum_s W[k*n+i, s] U[a*m+b, s] V[c*p+d, s] = delta_{a,i} * delta_{b,c} * delta_{d,k}
            
            if b == c:
                for i in range(n):
                    for k in range(p):
                        target = 1.0 if (a == i and d == k) else 0.0
                        idx_w = k * n + i
                        # val = dot(W[idx_w], U[a*m+b] * V[c*p+d])
                        prod = U[a*m+b, :] * V[c*p+d, :]
                        val = np.dot(W[idx_w, :], prod)
                        err = val - target
                        loss += err * err
                        # dW[idx_w] += 2 * err * prod
                        dW[idx_w, :] += 2 * err * prod
                        # dU[a*m+b] += 2 * err * W[idx_w] * V[c*p+d]
                        dU[a*m+b, :] += 2 * err * W[idx_w, :] * V[c*p+d, :]
                        # dV[c*p+d] += 2 * err * W[idx_w] * U[a*m+b]
                        dV[c*p+d, :] += 2 * err * W[idx_w, :] * U[a*m+b, :]
            else:
                for i in range(n):
                    for k in range(p):
                        idx_w = k * n + i
                        prod = U[a*m+b, :] * V[c*p+d, :]
                        val = np.dot(W[idx_w, :], prod)
                        err = val
                        loss += err * err
                        dW[idx_w, :] += 2 * err * prod
                        dU[a*m+b, :] += 2 * err * W[idx_w, :] * V[c*p+d, :]
                        dV[c*p+d, :] += 2 * err * W[idx_w, :] * U[a*m+b, :]
        
        loss /= num_samples
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.9
        
        if it % 500 == 0:
            # Try to snap to half-integers
            pass
    
    # Final check
    U, V, W = best_U, best_V, best_W
    if _check_decomposition(U, V, W, tol=1e-8):
        return U, V, W, True
    
    # Try snapping to nearest half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    if _check_decomposition(U_snap, V_snap, W_snap, tol=1e-8):
        return U_snap, V_snap, W_snap, True
    
    return None, None, None, False


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # First, get the schoolbook solution as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also rank >= n*m*p / max(n*m, m*p, p*n) ... not a standard bound
    # For 2x3x7, let's try ranks from 21 upward
    
    best_result = (U_sb, V_sb, W_sb, schoolbook_rank)
    
    # Try a few ranks
    for target_rank in range(21, min(35, schoolbook_rank + 1)):
        if time.time() > DEADLINE - 10:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > DEADLINE - 10:
                break
            
            seed = random.randint(0, 1000000)
            U, V, W, success = _try_rank_decomposition(target_rank, max_iters=2000, seed=seed)
            
            if success:
                best_result = (U, V, W, target_rank)
                found = True
                break
        
        if found:
            # Try even lower ranks
            continue
        else:
            # If we couldn't find this rank, stop trying lower
            # Actually, let's continue trying a few more
            break
    
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
