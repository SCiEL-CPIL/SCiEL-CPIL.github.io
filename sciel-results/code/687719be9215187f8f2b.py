"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """
    Verify if the decomposition exactly multiplies matrices.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Convention: W index is k*n + i.
    T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    We want T[i, j, k] = 1 if j is the "middle" index? No.
    Matrix multiplication C = A * B where A is n x m, B is m x p.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor is T[i, j, k] which is 1 if the indices match the multiplication structure.
    Specifically, the tensor for multiplication is T[i, j, k] = delta_{j, j}? No.
    The tensor is T_{i, j, k} such that sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}.
    Wait, the standard tensor is T[i, j, k] = 1 if the indices correspond to the term A[i,j]B[j,k].
    So T[i, j, k] = 1 for all i, j, k? No, that would be the "all ones" tensor.
    The multiplication tensor has entries T[i, j, k] = 1 if j is the shared index.
    Actually, the tensor is T[i, j, k] = 1 if we are summing over j.
    For fixed i, k, the term is A[i, j] B[j, k].
    So the tensor T has non-zero entries at (i, j, k) for all valid i, j, k?
    No, the tensor represents the bilinear form.
    T[i, j, k] = 1.
    Let's check the schoolbook.
    U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1.
    Sum_c U[i*m + j, c] V[j*p + k, c] W[k*n + i, c].
    This product is 1 only if there is a c such that U and V and W are 1 at that c.
    In schoolbook, c corresponds to (i, j, k).
    U[i*m + j, c] is 1 if c corresponds to (i', j', k') with i'=i, j'=j.
    V[j*p + k, c] is 1 if c corresponds to (i', j', k') with j'=j, k'=k.
    W[k*n + i, c] is 1 if c corresponds to (i', j', k') with i'=i, k'=k.
    So for a specific (i, j, k), the term is 1 if c matches (i, j, k).
    So the sum is 1 for each (i, j, k).
    Thus T[i, j, k] = 1 for all i, j, k.
    Wait, this implies the tensor is the "all ones" tensor?
    No, the tensor for matrix multiplication is NOT the all ones tensor.
    The tensor T is such that C = (T, A, B).
    C[i, k] = sum_{j} A[i, j] B[j, k].
    This can be written as sum_{i, j, k} T[i, j, k] A[i, j] B[j, k] delta_{i, i'} delta_{k, k'}?
    The tensor T is defined on indices i (row of A), j (col of A / row of B), k (col of B).
    The value T[i, j, k] is 1.
    So yes, the tensor for matrix multiplication is the tensor with all entries 1?
    No! That would mean C[i,k] = sum_{j} 1 * A[i,j] * B[j,k].
    Which is correct.
    So the target tensor T[i, j, k] = 1 for all i, j, k.
    
    However, the decomposition uses indices:
    U is indexed by (i, j).
    V is indexed by (j, k).
    W is indexed by (k, i).
    
    The product is sum_c U[i, j, c] V[j, k, c] W[k, i, c].
    This must equal T[i, j, k] = 1.
    
    So we are looking for a decomposition of the tensor T[i, j, k] = 1.
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != r or len(V[0]) != r or len(W[0]) != r:
        return False
    
    # Target tensor is 1 everywhere.
    # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should be 1.
    
    # For speed, we can use numpy.
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # We need to check for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # This is equivalent to checking if the tensor defined by the decomposition
    # is the all-ones tensor.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    
    # We can iterate over i, j, k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u = U[i*m + j]
                # v = V[j*p + k]
                # w = W[k*n + i]
                # s = sum(u[c]*v[c]*w[c] for c in range(r))
                # Using numpy dot product is faster if we reshape?
                # Let's just do the dot product of u and (v * w)
                # v_w = v * w
                # s = np.dot(u, v_w)
                u = U_np[i * m + j]
                v = V_np[j * p + k]
                w = W_np[k * n + i]
                s = np.dot(u, v * w)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _try_random_rank(target_rank, timeout=5):
    """
    Attempt to find a random rank decomposition.
    This is very unlikely to work for exact integer/half-integer solutions,
    but we try a few random integer matrices and verify.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    for _ in range(1000):
        if time.time() - start_time > timeout:
            break
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        W = np.random.randint(-2, 3, size=(p * n, target_rank))
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def construct():
    deadline = time.time() + 110  # 10s margin for 120s limit
    
    # 1. Try schoolbook (guaranteed)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # 2. Try to find lower rank
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    # Strassen-like algorithms might get close, but for 3x5x5, it's likely still high.
    # Random search for integer solutions is extremely unlikely to succeed unless the rank is very close to the solution space density.
    # We will try a few ranks below 75.
    
    for r in range(74, 25, -1):
        if time.time() > deadline:
            break
        # Only try a few if we have time
        timeout = 5 if time.time() < deadline - 20 else 2
        res = _try_random_rank(r, timeout=timeout)
        if res:
            return res
            
    return best
# EVOLVE-BLOCK-END
