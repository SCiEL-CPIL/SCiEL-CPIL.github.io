import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 150  # seconds

    n, m, p = DIMENSIONS  # 4, 4, 9

    # Indices
    # U: i*m + j (16)
    # V: j*p + k (36)
    # W: k*n + i (36)
    
    # We'll use Strassen for 4x4 x 4x4 (rank 49) + schoolbook for 4x4 x 4x5 (rank 80) = 129
    # But let's try a better approach: use Strassen 4x4x4 with rank 49 for the first 4 columns
    # and schoolbook for the remaining 5 columns.
    # Actually, let's just implement Strassen properly for the full problem.
    
    # Simpler: use schoolbook for all 9 columns = 144, but we can do better.
    # Let's use Strassen for 4x4 x 4x4 (49 terms) and schoolbook for 4x4 x 4x5 (80 terms) = 129
    
    rank = 129
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    term_idx = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal term_idx
        if term_idx >= rank:
            return
        for idx, coeff in u_coeffs.items():
            if coeff != 0:
                U[idx][term_idx] += coeff
        for idx, coeff in v_coeffs.items():
            if coeff != 0:
                V[idx][term_idx] += coeff
        for idx, coeff in w_coeffs.items():
            if coeff != 0:
                W[idx][term_idx] += coeff
        term_idx += 1

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_lin_comb(d1, c1, d2, c2):
        res = {}
        for k, v in d1.items():
            res[k] = res.get(k, 0) + c1 * v
        for k, v in d2.items():
            res[k] = res.get(k, 0) + c2 * v
        # Remove zeros
        return {k: v for k, v in res.items() if v != 0}

    # Strassen 4x4 x 4x4 for columns 0-3
    # A blocks (2x2 each):
    A11 = {u_idx(0,0): 1, u_idx(0,1): 1, u_idx(1,0): 1, u_idx(1,1): 1}
    A12 = {u_idx(0,2): 1, u_idx(0,3): 1, u_idx(1,2): 1, u_idx(1,3): 1}
    A21 = {u_idx(2,0): 1, u_idx(2,1): 1, u_idx(3,0): 1, u_idx(3,1): 1}
    A22 = {u_idx(2,2): 1, u_idx(2,3): 1, u_idx(3,2): 1, u_idx(3,3): 1}
    
    # B blocks for columns 0-3 (2x2 each):
    B11 = {v_idx(0,0): 1, v_idx(0,1): 1, v_idx(1,0): 1, v_idx(1,1): 1}
    B12 = {v_idx(0,2): 1, v_idx(0,3): 1, v_idx(1,2): 1, v_idx(1,3): 1}
    B21 = {v_idx(2,0): 1, v_idx(2,1): 1, v_idx(3,0): 1, v_idx(3,1): 1}
    B22 = {v_idx(2,2): 1, v_idx(2,3): 1, v_idx(3,2): 1, v_idx(3,3): 1}
    
    # C blocks (2x2 each):
    # C11: i 0,1; k 0,1 -> w_idx(0,0)=0, w_idx(0,1)=1, w_idx(1,0)=4, w_idx(1,1)=5
    # C12: i 0,1; k 2,3 -> w_idx(2,0)=8, w_idx(2,1)=9, w_idx(3,0)=12, w_idx(3,1)=13
    # C21: i 2,3; k 0,1 -> w_idx(0,2)=2, w_idx(0,3)=3, w_idx(1,2)=6, w_idx(1,3)=7
    # C22: i 2,3; k 2,3 -> w_idx(2,2)=10, w_idx(2,3)=11, w_idx(3,2)=14, w_idx(3,3)=15
    
    C11_indices = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]  # [0,1,4,5]
    C12_indices = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]  # [8,9,12,13]
    C21_indices = [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]  # [2,3,6,7]
    C22_indices = [w_idx(2,2), w_idx(2,3), w_idx(3,2), w_idx(3,3)]  # [10,11,14,15]
    
    # Strassen 4x4 formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_k is a 2x2 matrix product. We decompose each into 7 Strassen 2x2 terms.
    # But that gives 7*7=49 terms for 4x4x4.
    
    # Let's implement a helper to add a 2x2 Strassen product
    def add_strassen_2x2(A_entries, B_entries, target_indices, sign):
        """
        A_entries: list of 4 dicts [A00, A01, A10, A11]
        B_entries: list of 4 dicts [B00, B01, B10, B11]
        target_indices: list of 4 w_idx values for the 2x2 C block [C00, C01, C10, C11]
        sign: multiplier for the whole product
        """
        nonlocal term_idx
        
        # Strassen 2x2:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        A00, A01, A10, A11 = A_entries
        B00, B01, B10, B11 = B_entries
        
        # P1
        u1 = add_lin_comb(A00, 1, A11, 1)
        v1 = add_lin_comb(B00, 1, B11, 1)
        w1 = {target_indices[0]: sign, target_indices[3]: sign}  # C00 and C11
        add_term(u1, v1, w1)
        
        # P2
        u2 = add_lin_comb(A10, 1, A11, 1)
        v2 = B00
        w2 = {target_indices[2]: sign, target_indices[3]: -sign}  # C10 and C11
        add_term(u2, v2, w2)
        
        # P3
        u3 = A00
        v3 = add_lin_comb(B01, 1, B11, -1)
        w3 = {target_indices[1]: sign}  # C01
        add_term(u3, v3, w3)
        
        # P4
        u4 = A11
        v4 = add_lin_comb(B10, 1, B00, -1)
        w4 = {target_indices[0]: sign, target_indices[2]: sign}  # C00 and C10
        add_term(u4, v4, w4)
        
        # P5
        u5 = add_lin_comb(A00, 1, A01, 1)
        v5 = B11
        w5 = {target_indices[0]: -sign, target_indices[1]: sign}  # C00 and C01
        add_term(u5, v5, w5)
        
        # P6
        u6 = add_lin_comb(A10, 1, A00, -1)
        v6 = add_lin_comb(B00, 1, B01, 1)
        w6 = {target_indices[3]: sign}  # C11
        add_term(u6, v6, w6)
        
        # P7
        u7 = add_lin_comb(A01, 1, A11, -1)
        v7 = add_lin_comb(B10, 1, B11, 1)
        w7 = {target_indices[0]: sign}  # C00
        add_term(u7, v7, w7)
    
    # Now we need to extract the 2x2 entries from the block linear forms.
    # For A11+A22:
    # Entry (0,0): A[0,0] + A[2,2] = u_idx(0,0) + u_idx(2,2)
    # Entry (0,1): A[0,1] + A[2,3] = u_idx(0,1) + u_idx(2,3)
    # Entry (1,0): A[1,0] + A[3,2] = u_idx(1,0) + u_idx(3,2)
    # Entry (1,1): A[1,1] + A[3,3] = u_idx(1,1) + u_idx(3,3)
    
    def get_A_entries(block1, block2, sign1=1, sign2=1):
        """Get 2x2 entries from sum of two A blocks"""
        # block1, block2 are dicts with global U indices
        # We need to map to local 2x2 indices
        # For A11: local (r,c) -> global u_idx(r, c)
        # For A12: local (r,c) -> global u_idx(r, c+2)
        # For A21: local (r,c) -> global u_idx(r+2, c)
        # For A22: local (r,c) -> global u_idx(r+2, c+2)
        
        entries = [{}, {}, {}, {}]  # [00, 01, 10, 11]
        for local_r in range(2):
            for local_c in range(2):
                local_idx = local_r * 2 + local_c
                # Map to global based on which block
                # This is tricky because we need to know the block type
                # Let's just build from scratch
        pass
    
    # Simpler: define the linear forms for each M_k's A' and B' explicitly
    
    # M1 = (A11+A22)(B11+B22)
    A_M1 = [
        {u_idx(0,0): 1, u_idx(2,2): 1},  # (0,0)
        {u_idx(0,1): 1, u_idx(2,3): 1},  # (0,1)
        {u_idx(1,0): 1, u_idx(3,2): 1},  # (1,0)
        {u_idx(1,1): 1, u_idx(3,3): 1}   # (1,1)
    ]
    B_M1 = [
        {v_idx(0,0): 1, v_idx(2,2): 1},  # (0,0)
        {v_idx(0,1): 1, v_idx(2,3): 1},  # (0,1)
        {v_idx(1,0): 1, v_idx(3,2): 1},  # (1,0)
        {v_idx(1,1): 1, v_idx(3,3): 1}   # (1,1)
    ]
    
    # M2 = (A21+A22)B11
    A_M2 = [
        {u_idx(2,0): 1, u_idx(2,2): 1},  # (0,0)
        {u_idx(2,1): 1, u_idx(2,3): 1},  # (0,1)
        {u_idx(3,0): 1, u_idx(3,2): 1},  # (1,0)
        {u_idx(3,1): 1, u_idx(3,3): 1}   # (1,1)
    ]
    B_M2 = [
        {v_idx(0,0): 1},  # (0,0)
        {v_idx(0,1): 1},  # (0,1)
        {v_idx(1,0): 1},  # (1,0)
        {v_idx(1,1): 1}   # (1,1)
    ]
    
    # M3 = A11(B12-B22)
    A_M3 = [
        {u_idx(0,0): 1},  # (0,0)
        {u_idx(0,1): 1},  # (0,1)
        {u_idx(1,0): 1},  # (1,0)
        {u_idx(1,1): 1}   # (1,1)
    ]
    B_M3 = [
        {v_idx(0,2): 1, v_idx(2,2): -1},  # (0,0)
        {v_idx(0,3): 1, v_idx(2,3): -1},  # (0,1)
        {v_idx(1,2): 1, v_idx(3,2): -1},  # (1,0)
        {v_idx(1,3): 1, v_idx(3,3): -1}   # (1,1)
    ]
    
    # M4 = A22(B21-B11)
    A_M4 = [
        {u_idx(2,2): 1},  # (0,0)
        {u_idx(2,3): 1},  # (0,1)
        {u_idx(3,2): 1},  # (1,0)
        {u_idx(3,3): 1}   # (1,1)
    ]
    B_M4 = [
        {v_idx(2,0): 1, v_idx(0,0): -1},  # (0,0)
        {v_idx(2,1): 1, v_idx(0,1): -1},  # (0,1)
        {v_idx(3,0): 1, v_idx(1,0): -1},  # (1,0)
        {v_idx(3,1): 1, v_idx(1,1): -1}   # (1,1)
    ]
    
    # M5 = (A11+A12)B22
    A_M5 = [
        {u_idx(0,0): 1, u_idx(0,2): 1},  # (0,0)
        {u_idx(0,1): 1, u_idx(0,3): 1},  # (0,1)
        {u_idx(1,0): 1, u_idx(1,2): 1},  # (1,0)
        {u_idx(1,1): 1, u_idx(1,3): 1}   # (1,1)
    ]
    B_M5 = [
        {v_idx(2,2): 1},  # (0,0)
        {v_idx(2,3): 1},  # (0,1)
        {v_idx(3,2): 1},  # (1,0)
        {v_idx(3,3): 1}   # (1,1)
    ]
    
    # M6 = (A21-A11)(B11+B12)
    A_M6 = [
        {u_idx(2,0): 1, u_idx(0,0): -1},  # (0,0)
        {u_idx(2,1): 1, u_idx(0,1): -1},  # (0,1)
        {u_idx(3,0): 1, u_idx(1,0): -1},  # (1,0)
        {u_idx(3,1): 1, u_idx(1,1): -1}   # (1,1)
    ]
    B_M6 = [
        {v_idx(0,0): 1, v_idx(0,2): 1},  # (0,0)
        {v_idx(0,1): 1, v_idx(0,3): 1},  # (0,1)
        {v_idx(1,0): 1, v_idx(1,2): 1},  # (1,0)
        {v_idx(1,1): 1, v_idx(1,3): 1}   # (1,1)
    ]
    
    # M7 = (A12-A22)(B21+B22)
    A_M7 = [
        {u_idx(0,2): 1, u_idx(2,2): -1},  # (0,0)
        {u_idx(0,3): 1, u_idx(2,3): -1},  # (0,1)
        {u_idx(1,2): 1, u_idx(3,2): -1},  # (1,0)
        {u_idx(1,3): 1, u_idx(3,3): -1}   # (1,1)
    ]
    B_M7 = [
        {v_idx(2,0): 1, v_idx(2,2): 1},  # (0,0)
        {v_idx(2,1): 1, v_idx(2,3): 1},  # (0,1)
        {v_idx(3,0): 1, v_idx(3,2): 1},  # (1,0)
        {v_idx(3,1): 1, v_idx(3,3): 1}   # (1,1)
    ]
    
    # Add M1 to C11 (sign +1) and C22 (sign +1)
    add_strassen_2x2(A_M1, B_M1, C11_indices, 1)
    add_strassen_2x2(A_M1, B_M1, C22_indices, 1)
    
    # Add M2 to C21 (sign +1) and C22 (sign -1)
    add_strassen_2x2(A_M2, B_M2, C21_indices, 1)
    add_strassen_2x2(A_M2, B_M2, C22_indices, -1)
    
    # Add M3 to C12 (sign +1) and C22 (sign +1)
    add_strassen_2x2(A_M3, B_M3, C12_indices, 1)
    add_strassen_2x2(A_M3, B_M3, C22_indices, 1)
    
    # Add M4 to C11 (sign +1) and C21 (sign +1)
    add_strassen_2x2(A_M4, B_M4, C11_indices, 1)
    add_strassen_2x2(A_M4, B_M4, C21_indices, 1)
    
    # Add M5 to C11 (sign -1) and C12 (sign +1)
    add_strassen_2x2(A_M5, B_M5, C11_indices, -1)
    add_strassen_2x2(A_M5, B_M5, C12_indices, 1)
    
    # Add M6 to C22 (sign +1)
    add_strassen_2x2(A_M6, B_M6, C22_indices, 1)
    
    # Add M7 to C11 (sign +1)
    add_strassen_2x2(A_M7, B_M7, C11_indices, 1)
    
    # Now add schoolbook for columns 4-8 (5 columns)
    # For each column k in 4..8, and each i,j:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    for k in range(4, 9):
        for i in range(4):
            for j in range(4):
                u = {u_idx(i, j): 1}
                v = {v_idx(j, k): 1}
                w = {w_idx(k, i): 1}
                add_term(u, v, w)
    
    # Verify term_idx
    # Should be 49 (Strassen) + 80 (schoolbook for 5 cols) = 129
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
