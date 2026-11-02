"""
Tensor decomposition for 4x5 * 5x5 matrix multiplication.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks to reduce rank below schoolbook.
"""
import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank_school = n * m * p  # 100
    
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def build_schoolbook():
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1.0
                    V[j * p + k][c] = 1.0
                    W[k * n + i][c] = 1.0
        return U, V, W
    
    # Strassen's 7-term decomposition for 2x2 matrix multiplication
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each of the 7 terms, we have:
    # U[i,j] = coefficient of A[i,j]
    # V[j,k] = coefficient of B[j,k]
    # W[i,k] = coefficient for C[i,k]
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    # Term 2: M2 = (A10+A11)B00 -> contributes to C10 (+1)
    # U: A10=1, A11=1; V: B00=1; W: C10=1
    # Term 3: M3 = A00(B10-B11) -> contributes to C01 (+1)
    # U: A00=1; V: B10=1, B11=-1; W: C01=1
    # Term 4: M4 = A11(B01-B00) -> contributes to C10 (+1), C11 (-1)
    # U: A11=1; V: B01=1, B00=-1; W: C10=1, C11=-1
    # Term 5: M5 = (A00+A01)B11 -> contributes to C00 (-1), C01 (+1)
    # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
    # Term 6: M6 = (A10-A00)(B00+B01) -> contributes to C11 (+1)
    # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
    # Term 7: M7 = (A01-A11)(B10+B11) -> contributes to C00 (+1), C11 (+1)
    # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1, C11=1
    
    # Define the 7 terms as (U_factors, V_factors, W_factors)
    # U_factors: list of (local_i, local_j, coeff)
    # V_factors: list of (local_j, local_k, coeff)
    # W_factors: list of (local_i, local_k, coeff)
    
    strassen_terms = [
        # Term 1: M1
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        # Term 2: M2
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0)]),
        # Term 3: M3
        ([(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)], [(0,1,1.0)]),
        # Term 4: M4
        ([(1,1,1.0)], [(0,1,1.0),(0,0,-1.0)], [(1,0,1.0),(1,1,-1.0)]),
        # Term 5: M5
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        # Term 6: M6
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        # Term 7: M7
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
    ]
    
    def build_strassen_decomposition():
        """Build decomposition using Strassen for two disjoint 2x2 blocks."""
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        # Block 2: i in {2,3}, j in {2,3}, k in {2,3}
        
        # Total rank: 7 (block 1) + 7 (block 2) + 86 (rest) = 100? 
        # Wait, rest is 100 - 4 - 4 = 92 entries, so rank = 7 + 7 + 92 = 106? No.
        
        # Let me recount. Total entries = 4*5*5 = 100.
        # Block 1 covers (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}: 2*2*2 = 8 entries
        # Block 2 covers (i,j,k) where i in {2,3}, j in {2,3}, k in {2,3}: 2*2*2 = 8 entries
        # Rest: 100 - 8 - 8 = 84 entries
        # Rank = 7 + 7 + 84 = 98
        
        total_rank = 98
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        base1 = 0
        for t in range(7):
            u_factors, v_factors, w_factors = strassen_terms[t]
            for (li, lj, coeff) in u_factors:
                i, j = li, lj
                U[i * m + j, base1 + t] = coeff
            for (lj, lk, coeff) in v_factors:
                j, k = lj, lk
                V[j * p + k, base1 + t] = coeff
            for (li, lk, coeff) in w_factors:
                i, k = li, lk
                W[k * n + i, base1 + t] = coeff
        
        # Block 2: i in {2,3}, j in {2,3}, k in {2,3}
        base2 = 7
        for t in range(7):
            u_factors, v_factors, w_factors = strassen_terms[t]
            for (li, lj, coeff) in u_factors:
                i, j = li + 2, lj + 2
                U[i * m + j, base2 + t] = coeff
            for (lj, lk, coeff) in v_factors:
                j, k = lj + 2, lk + 2
                V[j * p + k, base2 + t] = coeff
            for (li, lk, coeff) in w_factors:
                i, k = li + 2, lk + 2
                W[k * n + i, base2 + t] = coeff
        
        # Rest: all other (i,j,k)
        col = 14
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip block 1 and block 2 entries
                    if i < 2 and j < 2 and k < 2:
                        continue
                    if i >= 2 and i < 4 and j >= 2 and j < 4 and k >= 2 and k < 4:
                        continue
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        assert col == total_rank, f"Column count mismatch: {col} vs {total_rank}"
        
        return U.tolist(), V.tolist(), W.tolist()
    
    def verify(U_list, V_list, W_list, rank):
        """Verify the decomposition is exact."""
        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # First, try the Strassen-based decomposition
    U_str, V_str, W_str = build_strassen_decomposition()
    
    if verify(U_str, V_str, W_str, 98):
        return {
            "U": U_str,
            "V": V_str,
            "W": W_str
        }
    
    # If Strassen verification fails, try schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    if verify(U_school, V_school, W_school, rank_school):
        return {
            "U": U_school,
            "V": V_school,
            "W": W_school
        }
    
    # Should not reach here, but return schoolbook anyway
    return {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }
