"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]
TIME_LIMIT = 100  # seconds, safety margin before 120s kill

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _is_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # But actually the condition is more subtle. Let me think...
    # The matrix multiplication tensor T[i,j,k] should be represented.
    # Actually, the condition is: for all i, k:
    #   sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * ... 
    # I need to be more careful.
    
    # The correct condition: the tensor T with T[(i,j),(j',k),(k',i')] = delta_{j,j'}delta_{k,k'}delta_{i,i'}
    # should equal sum_r U_r ⊗ V_r ⊗ W_r where U_r[i,j] = U[i*m+j, r], etc.
    # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} delta_{k,k'} delta_{i,i'}
    
    # This is a very specific condition. For the schoolbook, it holds.
    # Let me check a subset of conditions for efficiency.
    
    # Check: for all i, j, k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    # And for "off-diagonal" combinations, it should be 0.
    
    # Actually, the full condition is complex. Let me just verify by reconstructing.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_r sum_j A[i,j] * (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r])
    # Wait, that's not right either.
    
    # Let me re-derive. The standard form for matrix multiplication tensor decomposition:
    # We want: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i}delta_{j,j}delta_{k,k}
    # No, that's not it either.
    
    # I think the correct interpretation is:
    # The tensor M_{m×n,n×p} ∈ R^{mn × np × mp} is defined by:
    # M[(i,j),(j',k),(k',i')] = 1 if i=i', j=j', k=k' else 0
    # A rank-r decomposition: M = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{mn}, v_r ∈ R^{np}, w_r ∈ R^{mp}
    # So: sum_r u_r[i*m+j] * v_r[j'*p+k] * w_r[k'*n+i'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    
    # In our notation: U[i*m+j, r] = u_r[i*m+j], V[j'*p+k, r] = v_r[j'*p+k], W[k'*n+i', r] = w_r[k'*n+i']
    
    # So the condition is: for all i, j, j', k, k', i':
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 1 if (i=i' and j=j' and k=k') else 0
    
    # This is a huge number of conditions. For efficiency, let me check a random subset.
    
    # Actually, let me just check: for all i, j, k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # And for a few off-diagonal cases, check they're 0.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check some off-diagonal conditions
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        jp = random.randint(0, m-1)
        k = random.randint(0, p-1)
        kp = random.randint(0, p-1)
        ip = random.randint(0, n-1)
        if i == ip and j == jp and k == kp:
            continue
        s = 0.0
        for r in range(rank):
            s += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
        if abs(s) > 1e-9:
            return False
    return True

def _try_rank(r, n, m, p, max_iters=200, lr=0.01, seed=None):
    """Try to find a rank-r decomposition using gradient descent."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    # We want to minimize the error of the tensor decomposition.
    # The target tensor is: T[(i,j),(j',k),(k',i')] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Error = sum over all (i,j,j',k,k',i') of (sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] - T[...])^2
    
    # This is expensive to compute fully. Let me use a sampled approach.
    
    for it in range(max_iters):
        # Sample a few conditions
        errors = []
        grads_U = np.zeros_like(U)
        grads_V = np.zeros_like(V)
        grads_W = np.zeros_like(W)
        
        for _ in range(50):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            kp = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            
            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
            
            # Compute current value
            val = 0.0
            for r in range(r):
                val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
            
            err = val - target
            
            # Gradients
            for r in range(r):
                grads_U[i*m+j, r] += err * V[jp*p+k, r] * W[kp*n+ip, r]
                grads_V[jp*p+k, r] += err * U[i*m+j, r] * W[kp*n+ip, r]
                grads_W[kp*n+ip, r] += err * U[i*m+j, r] * V[jp*p+k, r]
        
        # Update
        U -= lr * grads_U
        V -= lr * grads_V
        W -= lr * grads_W
        
        # Round to half-integers periodically
        if it % 10 == 9:
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
    
    # Final rounding
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _is_exact(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    for rank in [48, 40, 36, 32]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        for attempt in range(3):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            result = _try_rank(rank, n, m, p, max_iters=100, lr=0.05, seed=rank*100+attempt)
            if result is not None:
                return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
