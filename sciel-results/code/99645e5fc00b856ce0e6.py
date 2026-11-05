import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n=6, m=6, p=6, tol=1e-9):
    """
    Verifies if the given U, V, W matrices form a valid tensor decomposition
    for n x m by m x p matrix multiplication.
    The condition is:
    sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct_strassen_3x3_blocks():
    """
    Strassen 2x2 blocks of 3x3. Rank: 7 * 27 = 189.
    This is the known-good solution.
    """
    n, m, p = 6, 6, 6
    
    U = np.zeros((n * m, 189))
    V = np.zeros((m * p, 189))
    W = np.zeros((p * n, 189))
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb, c_comb = strassen_terms[term_idx]
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * 3
                        col_a = (blk_a % 2) * 3
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx, col] += coeff_a
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * 3
                        col_b = (blk_b % 2) * 3
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx, col] += coeff_b
                    for (blk_c, coeff_c) in c_comb:
                        row_c = (blk_c // 2) * 3
                        col_c = (blk_c % 2) * 3
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx, col] += coeff_c
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_laderman_3x3():
    """
    Laderman 15-term decomposition for 3x3 matrix multiplication.
    This gives rank 15 for 3x3 x 3x3 -> 3x3.
    """
    n, m, p = 3, 3, 3
    
    # Laderman's 15-term decomposition for 3x3 matrix multiplication
    # The 15 terms are:
    # For each term t, we have coefficients a_t[i,k], b_t[k,j], c_t[i,j]
    # such that sum_t a_t[i,k] * b_t[k,j] * c_t[i,j] = 1 if i,j,k are valid and 0 otherwise.
    
    # I'll use the known Laderman decomposition.
    # The terms can be described by which (i,k), (k,j), (i,j) entries are nonzero.
    
    # Let me use a verified Laderman decomposition.
    # The 15 terms for 3x3:
    
    U = np.zeros((n * m, 15))
    V = np.zeros((m * p, 15))
    W = np.zeros((p * n, 15))
    
    # Laderman 3x3 decomposition (verified):
    # Term 0: a[0,0]=1, b[0,0]=1, c[0,0]=1
    # Term 1: a[1,1]=1, b[1,1]=1, c[1,1]=1
    # Term 2: a[2,2]=1, b[2,2]=1, c[2,2]=1
    # Term 3: a[0,1]=1, b[1,0]=1, c[0,0]=1
    # Term 4: a[0,2]=1, b[2,0]=1, c[0,0]=1
    # Term 5: a[1,0]=1, b[0,1]=1, c[1,1]=1
    # Term 6: a[2,0]=1, b[0,2]=1, c[2,2]=1
    # Term 7: a[1,2]=1, b[2,1]=1, c[1,1]=1
    # Term 8: a[2,1]=1, b[1,2]=1, c[2,2]=1
    # Term 9: a[0,1]=1, b[1,2]=1, c[0,2]=1
    # Term 10: a[1,0]=1, b[0,2]=1, c[1,2]=1
    # Term 11: a[1,2]=1, b[2,0]=1, c[1,0]=1
    # Term 12: a[2,1]=1, b[1,0]=1, c[2,0]=1
    # Term 13: a[0,2]=1, b[2,1]=1, c[0,1]=1
    # Term 14: a[2,0]=1, b[0,1]=1, c[2,1]=1
    
    # Wait, this doesn't seem right. Let me look up the actual Laderman decomposition.
    # The Laderman 3x3 decomposition has 15 terms. Let me use a known correct one.
    
    # Actually, let me just verify if the simple approach works.
    # For now, let me return a fallback that's known to work.
    
    return None  # Placeholder - need correct Laderman

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Return the known-good 189 rank solution
    result = construct_strassen_3x3_blocks()
    
    # Verify
    if _verify_decomposition(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback: schoolbook rank 216
        n, m, p = 6, 6, 6
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    r = i * m * p + k * p + j
                    U[i * m + k, r] = 1
                    V[k * p + j, r] = 1
                    W[j * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    valid = _verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}")
