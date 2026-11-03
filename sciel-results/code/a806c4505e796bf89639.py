import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (U_dict, V_dict, W_dict) with keys (i,j), (j,k), (k,i) in local 2x2 coords.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # M2 = (A10+A11)B00
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    # M3 = A00(B10-B00)
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    # M4 = A11(B01-B10)
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    # M5 = (A00+A01)B11
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _construct_2x3x9():
    n, m, p = 2, 3, 9
    # Rank calculation:
    # 4 pairs of columns * 7 Strassen terms = 28
    # 1 single column * 4 direct terms = 4
    # 1 row of A (index 2) * 9 columns * 2 rows of A = 18
    # Total = 50
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col_idx = 0
    
    # Part 1: Strassen for A[0:2, 0:2] x B[0:2, {k1, k2}] for 4 pairs
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            # U part: A[li, lj]
            for (li, lj), coeff in u_dict.items():
                U[li * m + lj, col_idx] = coeff
            
            # V part: B[lj, lk] -> B[lj, actual_k]
            for (lj, lk), coeff in v_dict.items():
                actual_k = k1 if lk == 0 else k2
                V[lj * p + actual_k, col_idx] = coeff
            
            # W part: C[li, lk] -> C[li, actual_k]
            for (li, lk), coeff in w_dict.items():
                actual_k = k1 if lk == 0 else k2
                W[actual_k * n + li, col_idx] = coeff
            
            col_idx += 1
            
    # Part 2: Direct multiplication for A[0:2, 0:2] x B[0:2, 8]
    # C[i, 8] = A[i, 0]B[0, 8] + A[i, 1]B[1, 8]
    for i in range(2):
        for j in range(2):
            U[i * m + j, col_idx] = 1
            V[j * p + 8, col_idx] = 1
            W[8 * n + i, col_idx] = 1
            col_idx += 1

    # Part 3: Direct multiplication for A[0:2, 2] x B[2, 0:9]
    # C[i, k] += A[i, 2] B[2, k]
    for i in range(2):
        for k in range(9):
            U[i * m + 2, col_idx] = 1
            V[2 * p + k, col_idx] = 1
            W[k * n + i, col_idx] = 1
            col_idx += 1
            
    return U.tolist(), V.tolist(), W.tolist()

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    # Convert to numpy arrays for fast einsum
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # T_computed[i*m+j, j*p+k, k*n+i] should be 1 if i,j,k are valid indices
    # T = sum_c U[:,c] V[:,c] W[:,c]
    # We can check a few random points or all points.
    # Since rank is small (50), we can reconstruct the full tensor.
    # Shape: (n*m, m*p, p*n)
    T_computed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    T_expected = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T_computed, T_expected, atol=1e-10)

def _rank1_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    try:
        U, V, W = _construct_2x3x9()
        
        # Quick verification
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            else:
                # If verification fails (should not happen), fall back
                U, V, W = _rank1_decomposition(n, m, p)
                return {"U": U, "V": V, "W": W}
        else:
            # If we ran out of time, return the constructed result hoping it's correct
            # (It is correct by construction)
            return {"U": U, "V": V, "W": W}
            
    except Exception:
        # Fallback to rank 1 if anything goes wrong
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
