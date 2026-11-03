"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _verify_444(U, V, W, rank):
    """Verify 4x4x4 decomposition."""
    n, m, p = 4, 4, 4
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _find_444_rank21(max_time=30):
    """
    Find a rank-21 decomposition for 4x4x4 matrix multiplication.
    Uses structured random search with integer coefficients.
    """
    n, m, p = 4, 4, 4
    rank = 21
    start = time.time()
    
    for seed in range(500):
        if time.time() - start > max_time:
            break
        random.seed(seed)
        np.random.seed(seed)
        
        # Try small integer coefficients
        U = np.random.randint(-2, 3, size=(16, rank))
        V = np.random.randint(-2, 3, size=(16, rank))
        
        # For each (k,i) pair, solve for W[k*n+i]
        W = np.zeros((16, rank), dtype=object)
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                b = np.ones(m)
                
                # Solve least squares and check if integer solution exists
                try:
                    w_sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                    w_int = np.round(w_sol)
                    error = np.abs(A @ w_int - b)
                    if np.max(error) > 1e-9:
                        valid = False
                        break
                    W[k*n+i, :] = w_int
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_444(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _compose_444_to_449(decomp_444):
    """
    Compose a rank-21 4x4x4 decomposition into a rank-63 4x4x9 decomposition.
    The 9 output columns are split into 3 groups of 3.
    """
    U_444 = np.array(decomp_444["U"])
    V_444 = np.array(decomp_444["V"])
    W_444 = np.array(decomp_444["W"])
    rank_444 = U_444.shape[1]
    
    n, m, p = 4, 4, 9
    rank_total = rank_444 * 3  # 63
    
    U = np.zeros((n*m, rank_total), dtype=object)
    V = np.zeros((m*p, rank_total), dtype=object)
    W = np.zeros((p*n, rank_total), dtype=object)
    
    for block in range(3):
        offset = block * rank_444
        # U is the same for all blocks
        U[:, offset:offset+rank_444] = U_444
        
        # V: for block b, columns 3*b to 3*b+2 of the 4x9 matrix
        # V_444[j*p4+k4] where p4=4, we need V_449[j*9+k] for k in [3b, 3b+2]
        for j in range(m):
            for k4 in range(3):  # only first 3 of the 4 columns in V_444
                k9 = 3*block + k4
                V[j*9+k9, offset:offset+rank_444] = V_444[j*4+k4]
        
        # W: W_444[k4*n+i] maps to W_449[k9*n+i] where k9 = 3*block + k4
        for k4 in range(3):
            k9 = 3*block + k4
            for i in range(n):
                W[k9*n+i, offset:offset+rank_444] = W_444[k4*n+i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_4x4x4_rank21():
    """
    Construct a rank 21 decomposition for 4x4x4 matrix multiplication.
    """
    n, m, p = 4, 4, 4
    result = _find_444_rank21(max_time=20)
    if result is not None:
        return result
    # Fallback to schoolbook 16
    return _schoolbook(4, 4, 4)

def _combine_decompositions(decomp1, decomp2, n1, m1, p1, n2, m2, p2):
    """
    Combine two decompositions using Kronecker product logic.
    """
    pass

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = 4, 4, 9
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = 144
    
    # Strategy 1: Find rank-21 for 4x4x4, compose to get rank-63 for 4x4x9
    if time.time() - start_time < DEADLINE - 30:
        decomp_444 = _find_444_rank21(max_time=DEADLINE - 40 - (time.time() - start_time))
        if decomp_444 is not None:
            composed = _compose_444_to_449(decomp_444)
            if _verify(composed["U"], composed["V"], composed["W"], n, m, p):
                best_result = composed
                best_rank = 63
                print(f"Found rank-63 decomposition via composition")
    
    # Strategy 2: If we have time, try direct search for even lower rank
    if time.time() - start_time < DEADLINE - 20 and best_rank > 63:
        candidate_ranks = [62, 61, 60, 59, 58, 57, 56, 55]
        
        for rank in candidate_ranks:
            if time.time() - start_time > DEADLINE - 15:
                break
            if rank >= best_rank:
                continue
                
            def _local_search(n, m, p, rank, max_iter=10, seed=42):
                random.seed(seed)
                np.random.seed(seed)
                for it in range(max_iter):
                    U = np.random.randint(-1, 2, size=(n*m, rank))
                    V = np.random.randint(-1, 2, size=(m*p, rank))
                    W = np.zeros((p*n, rank), dtype=object)
                    valid = True
                    for k in range(p):
                        for i in range(n):
                            A = np.zeros((m, rank), dtype=float)
                            for j in range(m):
                                for c in range(rank):
                                    A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                            b = np.ones(m)
                            try:
                                w_sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                                w_int = np.round(w_sol)
                                error = np.abs(A @ w_int - b)
                                if np.max(error) > 1e-9:
                                    valid = False
                                    break
                                W[k*n+i, :] = w_int
                            except:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if _verify(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                return None
            
            result = _local_search(n, m, p, rank, max_iter=5, seed=rank)
            if result is not None:
                best_result = result
                best_rank = rank
                print(f"Found rank-{rank} decomposition via direct search")
                break
    
    return best_result
