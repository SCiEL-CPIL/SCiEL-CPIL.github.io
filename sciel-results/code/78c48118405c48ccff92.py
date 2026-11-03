"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-6

def _try_round(U, V, W, n, m, p):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _strassen_decompose(n, m, p):
    """
    Recursive Strassen-style decomposition.
    Splits matrices into 2x2 blocks.
    If n=1, m=1, p=1 -> rank 1
    If n=1, m=1, p=2 -> rank 2 (schoolbook)
    If n=1, m=2, p=1 -> rank 2 (schoolbook)
    If n=2, m=1, p=1 -> rank 2 (schoolbook)
    Otherwise, use 7 multiplications.
    """
    # Base cases: small dimensions where Strassen doesn't help or isn't defined
    if n == 1 and m == 1 and p == 1:
        # C[0,0] = A[0,0] * B[0,0]
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W, 1
        
    if n == 1 and m == 1 and p == 2:
        # C[0,0] = A[0,0]B[0,0], C[0,1] = A[0,0]B[0,1]
        # Rank 2
        U = [[1, 1]]
        V = [[1, 0], [0, 1]]
        W = [[1, 0], [0, 1]]
        return U, V, W, 2

    if n == 1 and m == 2 and p == 1:
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # Rank 2
        U = [[1, 1]]
        V = [[1, 0], [0, 1]]
        W = [[1, 1]]
        return U, V, W, 2

    if n == 2 and m == 1 and p == 1:
        # C[0,0] = A[0,0]B[0,0], C[1,0] = A[1,0]B[0,0]
        # Rank 2
        U = [[1, 0], [0, 1]]
        V = [[1], [1]]
        W = [[1, 1]]
        return U, V, W, 2

    # Check if dimensions are even
    if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
        # Fallback to schoolbook for odd dimensions (or could pad, but let's keep it simple)
        # Actually, we can pad to next even number, but for now, if not divisible by 2, use schoolbook
        # However, for 2,4,7: 7 is odd.
        # We need to handle odd dimensions.
        # Strategy: If any dimension is odd, we can't split evenly.
        # For 2,4,7: n=2, m=4, p=7.
        # We can split n=2 into 1+1, m=4 into 2+2, but p=7 into 3+4?
        # Strassen requires square blocks.
        # Let's try to split into (n1, n2) where n1+n2=n, etc.
        # Standard Strassen: n1=n2=n/2, m1=m2=m/2, p1=p2=p/2.
        # If not divisible by 2, we can't use standard Strassen directly.
        
        # Let's try a general split: n1 = n//2, n2 = n - n1, etc.
        # But the 7 multiplications rely on the structure.
        # For now, if not all even, use schoolbook.
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
        return U, V, W, rank

    n1 = n // 2
    n2 = n - n1
    m1 = m // 2
    m2 = m - m1
    p1 = p // 2
    p2 = p - p1
    
    # 7 sub-problems
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to construct U, V, W for these 7 sub-problems combined.
    # This is complex because the sub-problems have different dimensions.
    # M1: (n1+m2) x (m1+m2) x (p1+p2) ... wait, A11 is n1 x m1, A22 is n2 x m2.
    # Strassen works for square matrices. For rectangular, it's more complex.
    # Given the time, let's stick to a simpler approach if Strassen is too hard to implement correctly for rectangular.
    
    # Let's try a different approach: Winograd-like for 1x4x7?
    # Or just use the schoolbook but try to find a lower rank via ALS if possible.
    
    # Actually, the prompt suggests "row-blocked" decomposition.
    # Let's implement the "row-blocked" idea properly.
    
    # For now, return schoolbook to be safe.
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
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Try Strassen (if it works)
    try:
        U_s, V_s, W_s, rank_s = _strassen_decompose(n, m, p)
        if _verify(U_s, V_s, W_s, n, m, p):
            # Try to round
            rounded = _try_round(U_s, V_s, W_s, n, m, p)
            if rounded:
                return {"U": rounded[0], "V": rounded[1], "W": rounded[2]}
            else:
                return {"U": U_s, "V": V_s, "W": W_s}
    except:
        pass

    # 2. Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # 3. Try ALS for lower ranks
    # Lower bound: max(n*m, m*p, p*n) = 28
    # Try ranks from 28 to 55
    
    for rank in range(28, rank_school):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            rng = np.random.default_rng(rank * 100 + trial)
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            
            T = np.ones((n, m, p), dtype=np.float64)
            
            # ALS iterations
            for it in range(100):
                if time.time() - _START_TIME > _TIME_LIMIT - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((rank, p))
                        for k in range(p):
                            for c in range(rank):
                                A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((rank, n))
                        for i in range(n):
                            for c in range(rank):
                                A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((rank, m))
                        for j in range(m):
                            for c in range(rank):
                                A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i,j,k] = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                res = np.linalg.norm(recon - T)
                if res < 1e-7:
                    break
            
            # Try to round
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            rounded = _try_round(U_list, V_list, W_list, n, m, p)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                # Found a valid decomposition at this rank, can stop
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
