import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_list(mat):
    return mat.tolist()

def _check_exact(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _gradient_descent_search(n, m, p, target_rank, seed, max_iter=1000, lr=0.01):
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    # Target tensor T[i,j,k] = 1 if (i,j,k) is a valid index, else 0?
    # No, T is the multiplication tensor.
    # T[i,j,k] is 1 if the term E_ij * E_jk * E_ki exists.
    # In our convention, T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, the tensor is T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}.
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    
    for it in range(max_iter):
        # Compute predicted tensor
        # P[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is expensive to do in pure python loops.
        # Use numpy broadcasting?
        # U3: (n, m, r)
        # V3: (m, p, r)
        # W3: (p, n, r)
        # We need sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        
        # Reshape
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)
        
        # Compute P
        # P[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a 4th order contraction.
        # We can do it in steps.
        # Temp[j,k,c] = sum_i? No.
        # Let's just loop i, j, k for small n,m,p.
        # n=3, m=5, p=6. 90 entries.
        # r=85.
        # 90 * 85 = 7650 ops per iteration.
        # 1000 iterations = 7.6M ops. Fast enough.
        
        P = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_row = U3[i, j]
                for k in range(p):
                    # sum_c u_row[c] * V3[j, k, c] * W3[k, i, c]
                    P[i, j, k] = np.dot(u_row, V3[j, k] * W3[k, i])
        
        # Target T is all ones
        T = np.ones((n, m, p))
        
        # Error
        E = P - T
        loss = np.sum(E**2)
        
        if loss < 1e-10:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Convert to object for exact check
            U_obj = U_rounded.astype(object)
            V_obj = V_rounded.astype(object)
            W_obj = W_rounded.astype(object)
            
            if _check_exact(U_obj, V_obj, W_obj, n, m, p):
                return U_obj, V_obj, W_obj
            else:
                return None

        # Compute Gradients
        # dL/dU[i*m+j, c] = 2 * sum_{k} E[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # dL/dV[j*p+k, c] = 2 * sum_{i} E[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
        # dL/dW[k*n+i, c] = 2 * sum_{j} E[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = E[i, j, k]
                    if e == 0: continue
                    for c in range(target_rank):
                        dU[i * m + j, c] += 2 * e * V[j * p + k, c] * W[k * n + i, c]
                        dV[j * p + k, c] += 2 * e * U[i * m + j, c] * W[k * n + i, c]
                        dW[k * n + i, c] += 2 * e * U[i * m + j, c] * V[j * p + k, c]
                        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Optional: restart or adapt lr
        if it % 100 == 0:
            lr *= 0.9

    return None

def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS

    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    # Try gradient descent for ranks 89, 88, 87
    for target_rank in [89, 88, 87]:
        if time.time() > deadline:
            break
        result = _gradient_descent_search(n, m, p, target_rank, seed=42, max_iter=500, lr=0.05)
        if result is not None:
            best = result
            best_rank = target_rank
            break

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
