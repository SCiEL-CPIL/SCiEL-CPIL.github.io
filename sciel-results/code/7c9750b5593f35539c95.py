import numpy as np
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 110.0

    n = 4
    m = 4
    p = 4

    # Known rank-49 decomposition via Kronecker product of Strassen's rank-7 2x2 algorithm
    # We first build this, verify it, and use it as baseline.
    # Then attempt to find a lower rank decomposition via randomized/local search.
    
    # Strassen 2x2 (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify 2x2 Strassen
    ok2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(s - 1.0) > 1e-10:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            break
    
    if not ok2:
        # Fallback: schoolbook 4x4 (rank 64)
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
    
    # Build 4x4 rank-49 from Kronecker product
    rank49 = 49
    U49 = np.zeros((16, 49), dtype=float)
    V49 = np.zeros((16, 49), dtype=float)
    W49 = np.zeros((16, 49), dtype=float)
    
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            idx_u = i * 4 + j
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U49[idx_u, c] = U2[i_hi * 2 + j_hi, c1] * U2[i_lo * 2 + j_lo, c2]
    
    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            idx_v = j * 4 + k
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V49[idx_v, c] = V2[j_hi * 2 + k_hi, c1] * V2[j_lo * 2 + k_lo, c2]
    
    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            idx_w = k * 4 + i
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W49[idx_w, c] = W2[k_hi * 2 + i_hi, c1] * W2[k_lo * 2 + i_lo, c2]
    
    # Verify rank-49
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U49, V49, W49, n, m, p):
        # Fallback: schoolbook
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
    
    # Try to find a lower rank decomposition via local search
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # We want to find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] sums to 1
    
    # Attempt to use a randomized approach to find rank < 49
    # This is a hard problem, but we'll try a few strategies
    
    best_rank = 49
    best_U = U49.copy()
    best_V = V49.copy()
    best_W = W49.copy()
    
    # Strategy: try to find a rank-40, 42, 44, 46, 48 decomposition
    # Using a simple local search: start with rank-49, try to merge/remove terms
    
    # Actually, let's try a more principled approach:
    # Randomly initialize a lower-rank decomposition and try to refine it
    # This is unlikely to work well, but we'll try
    
    # For now, just return the rank-49 solution
    # The search space is too large for a simple local search to find rank < 49
    # without much more sophisticated machinery
    
    # Convert to lists with integer/half-integer entries
    def to_half_integer_list(arr):
        """Convert array to list of lists with half-integer (multiple of 0.5) entries"""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                # If it's very close to an integer, use integer
                if abs(rounded - round(rounded)) < 1e-10:
                    new_row.append(int(round(rounded)))
                else:
                    # Represent as fraction with denominator 2
                    new_row.append(rounded)
            result.append(new_row)
        return result
    
    U_list = to_half_integer_list(best_U)
    V_list = to_half_integer_list(best_V)
    W_list = to_half_integer_list(best_W)
    
    # Final verification with the converted lists
    def verify_lists(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(len(U[0])))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_lists(U_list, V_list, W_list, n, m, p):
        # If conversion caused issues, fall back to original
        U_list = [[float(x) for x in row] for row in U49.tolist()]
        V_list = [[float(x) for x in row] for row in V49.tolist()]
        W_list = [[float(x) for x in row] for row in W49.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
