"""Matrix multiplication 3x5x7 - rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Sum over rank
    # Result[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    # Check if all entries are close to 1
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = Fraction(1)
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c])
                    v_val = Fraction(V[j*p+k][c])
                    w_val = Fraction(W[k*n+i][c])
                    val += u_val * v_val * w_val
                if val != target:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    # The tensor for matrix multiplication C = AB is T_{ijk} = 1 for all valid i,j,k.
    # This is because C_{ik} = sum_j A_{ij} B_{jk}.
    # The rank of the all-ones tensor is 1.
    # Decomposition: U_{ij,1}=1, V_{jk,1}=1, W_{ki,1}=1.
    
    # Attempt Rank 1
    def try_rank_1():
        rank = 1
        U = [[1.0] for _ in range(nm)]
        V = [[1.0] for _ in range(mp)]
        W = [[1.0] for _ in range(pn)]
        
        if _verify_np(U, V, W, n, m, p):
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        return None

    result = try_rank_1()
    if result:
        return result

    # Fallback to schoolbook if Rank 1 fails (should not happen)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    return schoolbook()
