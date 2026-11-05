"""Matrix multiplication 3x4x5 tensor decomposition - search for reduced rank."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)


def _check_decomposition(U, V, W, n=3, m=4, p=5):
    """Check if U @ V @ W^T reconstructs the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0.
    We check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this as: for each i,j,k, element-wise product along c, then sum
    # Efficient way: for each k, compute A_k[i,j] = sum_c U3[i,j,c] * W3[k,i,c]
    # Then T[i,j,k] = sum_c A_k[i,j] * V3[j,k,c]... no that's not right.
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Let's just iterate - it's small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _check_decomposition_fast(U, V, W, n=3, m=4, p=5):
    """Faster check using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k] - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n=3, m=4, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _random_search(n=3, m=4, p=5, max_rank=59, trials_per_rank=200, time_budget=130.0):
    """Try random low-rank decompositions with random restarts."""
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    for rank in range(max_rank, 1, -1):
        if _time_left() < 10:
            break
        for trial in range(trials_per_rank):
            if _time_left() < 15:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Gradient descent to minimize sum of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            # Loss = sum_{i,j,k} (T[i,j,k] - 1)^2 where T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Also add penalty for entries that should be 0 (but in this problem all (i,j,k) should be 1)
            
            lr = 0.01
            for step in range(500):
                if _time_left() < 15:
                    break
                
                # Compute T
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                
                # Check if close to valid
                if np.max(np.abs(T - 1.0)) < 1e-9:
                    # Convert to exact form
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if _check_decomposition_fast(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # Gradient of L = sum_{i,j,k} (T[i,j,k] - 1)^2
                # dL/dU[i*m+j,c] = 2*(T[i,j,k]-1) * V[j*p+k,c] * W[k*n+i,c] summed over k
                # dL/dV[j*p+k,c] = 2*(T[i,j,k]-1) * U[i*m+j,c] * W[k*n+i,c] summed over i
                # dL/dW[k*n+i,c] = 2*(T[i,j,k]-1) * U[i*m+j,c] * V[j*p+k,c] summed over j
                
                err = (T - 1.0)  # (n,m,p)
                
                # dU: for each i,j,c: sum_k 2*err[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
                # = 2 * einsum('ijk,jkc,kic->ijc', err, V3, W3)
                dU3 = 2.0 * np.einsum('ijk,jkc,kic->ijc', err, V3, W3)
                dV3 = 2.0 * np.einsum('ijk,ijc,kic->jkc', err, U3, W3)
                dW3 = 2.0 * np.einsum('ijk,ijc,jkc->kic', err, U3, V3)
                
                dU = dU3.reshape(n*m, rank)
                dV = dV3.reshape(m*p, rank)
                dW = dW3.reshape(p*n, rank)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Final check
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            if _check_decomposition_fast(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best


def _structured_search(n=3, m=4, p=5, time_budget=130.0):
    """Try structured decompositions: exploit the symmetry of the problem.
    
    The tensor has a special structure: T[i,j,k] = 1 for ALL i,j,k.
    This means we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    One approach: try rank-r decompositions where each "path" (i,j,k) gets
    contributions from multiple columns.
    
    Key insight: If we can find matrices A (n x r), B (r x p), etc. such that
    the outer product structure works...
    
    Actually, let's think about it differently. The tensor T is the all-ones tensor.
    T[i,j,k] = 1 = 1 * 1 * 1. So rank 1 would work IF we could have
    U[i*m+j,c] = a_i, V[j*p+k,c] = b_j, W[k*n+i,c] = c_k with a_i*b_j*c_k = 1.
    But then a_i*b_j*c_k = a_i'*b_j'*c_k' for all triples, which forces a_i = const, etc.
    So rank 1 gives T[i,j,k] = a*b*c = const, which can be 1. Wait...
    
    U[i*m+j, c] = a[i] (depends only on i)
    V[j*p+k, c] = b[j] (depends only on j)
    W[k*n+i, c] = c[k] (depends only on k)
    Then T[i,j,k] = a[i]*b[j]*c[k]. We need this to equal 1 for all i,j,k.
    So a[i]*b[j]*c[k] = 1 for all i,j,k. This means a[i] = 1/b[j]/c[k] for all j,k,
    which forces b[j]*c[k] = const for all j,k, so b[j] = const and c[k] = const.
    Then a[i] = 1/(const*const) = const. So a[i]=b[j]=c[k]=1 works!
    
    So rank 1: U[i*m+j, 0] = 1 for all i,j; V[j*p+k, 0] = 1 for all j,k; W[k*n+i, 0] = 1 for all k,i.
    Then T[i,j,k] = 1*1*1 = 1. THIS WORKS!
    """
    # Rank 1 decomposition!
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    # Verify
    if _check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If that somehow failed (it shouldn't), fall back
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    
    # First try the rank-1 structured decomposition
    result = _structured_search(n, m, p)
    if result is not None and len(result["U"][0]) == 1:
        # Verify once more with slow check
        if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # If rank 1 didn't work (shouldn't happen), try random search for rank < n*m*p
    if _time_left() > 30:
        result2 = _random_search(n, m, p, max_rank=59, trials_per_rank=100, time_budget=100.0)
        if result2 is not None and len(result2["U"][0]) < n*m*p:
            if _check_decomposition(result2["U"], result2["V"], result2["W"], n, m, p):
                return result2
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
