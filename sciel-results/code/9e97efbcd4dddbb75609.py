"""Search for rank-minimal exact tensor decomposition of 2x4x7 matrix multiplication."""
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n=2, m=4, p=7):
    """Check if U, V, W give exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2_terms():
    """Return 7 rank-1 terms for 2x2x2 matrix multiplication tensor.
    Each term is (u_dict, v_dict, w_dict) where:
    u_dict maps (i,j) -> coeff, v_dict maps (j,k) -> coeff, w_dict maps (k,i) -> coeff
    """
    # Verified Strassen decomposition for 2x2x2
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # Sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
    
    terms = [
        # Term 1: (a00+a11)(b00+b11) -> c00+=, c11+=
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # Term 2: (a10+a11)b00 -> c10+=, c11-=
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # Term 3: a00(b01-b11) -> c01+=, c11+=
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # Term 4: a11(b10-b00) -> c00+=, c10+=
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # Term 5: (a00+a01)b11 -> c00-=, c01+=
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # Term 6: (a10-a00)(b00+b10) -> c11+=
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # Term 7: (a01-a11)(b01+b11) -> c00+=
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    terms = _strassen_2x2x2_terms()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for u_d, v_d, w_d in terms:
                    total += u_d.get((i,j), 0) * v_d.get((j,k), 0) * w_d.get((k,i), 0)
                if total != 1:
                    print(f"Strassen verification failed at ({i},{j},{k}): got {total}")
                    return False
    return True

def _construct_block_decomposition(n=2, m=4, p=7):
    """Construct block-structured decomposition using Strassen for 2x2x2 blocks."""
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, 6)]
    
    U = []
    V = []
    W = []
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for j_start, j_end in j_blocks:
        for k_start, k_end in k_blocks:
            k_size = k_end - k_start + 1
            
            if k_size == 2:
                for (u_dict, v_dict, w_dict) in strassen_terms:
                    u_row = [0] * (n * m)
                    v_row = [0] * (m * p)
                    w_row = [0] * (p * n)
                    
                    for (li, lj), val in u_dict.items():
                        u_row[li * m + (j_start + lj)] = val
                    
                    for (lj, lk), val in v_dict.items():
                        v_row[(j_start + lj) * p + (k_start + lk)] = val
                    
                    for (lk, li), val in w_dict.items():
                        w_row[(k_start + lk) * n + li] = val
                    
                    U.append(u_row)
                    V.append(v_row)
                    W.append(w_row)
            else:
                # Schoolbook for 2x2x1
                for li in range(2):
                    for lj in range(2):
                        u_row = [0] * (n * m)
                        v_row = [0] * (m * p)
                        w_row = [0] * (p * n)
                        
                        u_row[li * m + (j_start + lj)] = 1
                        v_row[(j_start + lj) * p + k_start] = 1
                        w_row[k_start * n + li] = 1
                        
                        U.append(u_row)
                        V.append(v_row)
                        W.append(w_row)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Verify Strassen decomposition first
    if not _verify_strassen():
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Construct block decomposition
    U, V, W = _construct_block_decomposition(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
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
    return U, V, W
