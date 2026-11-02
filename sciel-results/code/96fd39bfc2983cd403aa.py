import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Use a block decomposition strategy to reduce rank from 120 (schoolbook)
    # We can use Winograd/Strassen-like tricks on sub-blocks.
    # For 4x5x6, we can split into:
    # A: 4x5 -> [2x2, 2x2, 2x1] (rows 0:2, 2:4; cols 0:2, 2:4, 4:5)
    # B: 5x6 -> [2x3, 2x3, 1x3] (rows 0:2, 2:4, 4:5; cols 0:3, 3:6)
    # C: 4x6 -> [2x3, 2x3] (rows 0:2, 2:4; cols 0:3, 3:6)
    
    # This is getting complicated. Let's stick to a simpler, verified improvement.
    # Known best for 4x5x6 is likely lower than 120.
    # Schoolbook is 120.
    # If we use Strassen on the 4x4x4 part (rank 28 instead of 64) and schoolbook for the rest?
    # A(4x5) = [A1(4x4), A2(4x1)]
    # B(5x6) = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    
    # Cost of A1*B1 (4x4x4): Strassen rank 28 (vs 64)
    # Cost of A2*B2 (4x1x6): Schoolbook rank 4*1*6 = 24 (vs 24)
    # Total rank: 28 + 24 = 52.
    
    # Let's implement this.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1
        
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(j, k):
        return j * p + k
        
    def w_idx(i, k):
        return k * n + i

    # Part 1: A1 (4x4) * B1 (4x6) -> C1 (4x6)
    # We can further split B1 into B11 (4x3) and B12 (4x3).
    # A1 * B11 -> C11 (4x3)
    # A1 * B12 -> C12 (4x3)
    
    # For 4x4x3, we can use Strassen on 4x4x4 part? No, B is 4x3.
    # We can pad B to 4x4? Or just use schoolbook for 4x4x3 (48 ops) vs Strassen?
    # Strassen for 4x4x4 is 28. If we only need 3 cols, maybe we can save?
    # Actually, let's just use Strassen for the 4x4x4 block of B1 (cols 0-3) and schoolbook for col 4?
    # B1 is 4x6. Cols 0-5.
    # Let's split B1 into B1a (4x4) and B1b (4x2).
    # A1 * B1a (4x4x4) -> C1a (4x4). Use Strassen (rank 28).
    # A1 * B1b (4x4x2) -> C1b (4x2). Use Schoolbook (rank 32).
    # Total for Part 1: 28 + 32 = 60.
    
    # Wait, my previous estimate was 28 + 24 = 52.
    # A2 (4x1) * B2 (1x6) -> C2 (4x6). Rank 24.
    # So if Part 1 is 60, total is 84.
    # If Part 1 is 48 (schoolbook for 4x4x6), total is 48 + 24 = 72.
    # Schoolbook for 4x4x6 is 4*4*6 = 96? No, A1 is 4x4, B1 is 4x6.
    # Schoolbook rank for 4x4x6 is 4*4*6 = 96? No, it's sum over j.
    # Rank is n*m*p? No, rank is number of terms.
    # Schoolbook for (n x m) * (m x p) has rank n*m*p.
    # So A1(4x4) * B1(4x6) has rank 4*4*6 = 96? No.
    # The tensor is T[i,j,k] = sum_l A[i,l] B[l,k].
    # Here A is 4x4, B is 4x6.
    # Terms are indexed by i (0-3), k (0-5), and l (0-3).
    # Total terms: 4 * 6 * 4 = 96.
    # So schoolbook for Part 1 is 96.
    # Strassen for 4x4x4 part (l=0-3, k=0-3) is 28.
    # Remaining part: l=0-3, k=4-5. Rank 4*2*4 = 32.
    # Total Part 1: 28 + 32 = 60.
    # Part 2: A2(4x1) * B2(1x6). Rank 4*1*6 = 24.
    # Total: 60 + 24 = 84.
    
    # This is better than 120.
    # Can we do better?
    # Use Strassen for 4x4x6 directly?
    # There are algorithms for 4x4x6.
    # Or use 2x2x2 blocks.
    
    # Let's implement the 84 rank solution first as it's safer.
    # Actually, let's try to get closer to 79.
    
    # Re-evaluating:
    # A (4x5), B (5x6).
    # Split A into A1 (4x4) and A2 (4x1).
    # Split B into B1 (4x6) and B2 (1x6).
    # C = A1*B1 + A2*B2.
    
    # A1*B1: 4x4 * 4x6.
    # We can split B1 into B11 (4x3) and B12 (4x3).
    # A1*B11: 4x4 * 4x3. Rank 48 (schoolbook).
    # A1*B12: 4x4 * 4x3. Rank 48 (schoolbook).
    # Total 96.
    
    # Using Strassen for 4x4x4 part of B11 (cols 0-2)?
    # B11 is 4x3. We can treat it as 4x4 with last col zero?
    # Then rank 28. But we waste rank on the zero col?
    # The terms for the zero col will be zero in V, so they don't contribute to U or W?
    # No, the rank is the number of columns in U, V, W.
    # If V has zeros in a column, that column is still counted.
    # So we can't just pad.
    
    # Let's stick to the 84 rank implementation.
    # 1. A1 (4x4) * B1a (4x4) -> C1a (4x4). Strassen (28).
    # 2. A1 (4x4) * B1b (4x2) -> C1b (4x2). Schoolbook (32).
    # 3. A2 (4x1) * B2 (1x6) -> C2 (4x6). Schoolbook (24).
    # Total: 84.
    
    # Implement Strassen 4x4x4.
    # A1 is rows 0-3, cols 0-3.
    # B1a is rows 0-3, cols 0-3.
    # C1a is rows 0-3, cols 0-3.
    
    # Strassen 4x4x4 decomposes into 7 2x2x2 multiplications.
    # Each 2x2x2 has rank 7. Total 49?
    # No, Strassen 4x4x4 uses 7 multiplications of 2x2x2?
    # No, Strassen 2x2x2 uses 7 multiplications of 1x1x1 (scalars).
    # Strassen 4x4x4 uses 7 multiplications of 2x2x2.
    # Each 2x2x2 has rank 7 (if using Strassen) or 8 (schoolbook).
    # If we use Strassen recursively, rank is 7*7 = 49.
    # But we can use schoolbook for the 2x2x2 parts? Rank 7*8 = 56.
    # Or use the optimal 2x2x2 rank (7).
    # So 49 is possible for 4x4x4.
    # My previous estimate of 28 was wrong?
    # Winograd 4x4x4 is 28?
    # Yes, Winograd's algorithm for 4x4x4 has rank 28.
    # Strassen is 49.
    # So if we use Winograd, Part 1a is 28.
    # Part 1b is 32.
    # Part 2 is 24.
    # Total 84.
    
    # If we use Strassen (49), total is 49 + 32 + 24 = 105. Worse than schoolbook 120? No, 105 < 120.
    # But Winograd 28 is better.
    
    # Let's implement Winograd 4x4x4?
    # It's complex.
    # Let's just use Strassen 4x4x4 (rank 49) to be safe, or schoolbook for the whole thing?
    # Schoolbook 120.
    # Strassen 4x4x4 (49) + Schoolbook 32 + Schoolbook 24 = 105.
    # This is a valid improvement.
    
    # Let's implement the 105 rank solution.
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1
            
    def a_idx(i, j):
        return i * m + j
            
    def b_idx(j, k):
        return j * p + k
            
    def w_idx(i, k):
        return k * n + i
        
    # Strassen 2x2x2 helper
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        # A block at (a_r, a_c) size 2x2
        # B block at (b_r, b_c) size 2x2
        # C block at (c_r, c_c) size 2x2
        
        # 1. (A11+A22)(B11+B22)
        add_term(
            {a_idx(a_r, a_c): 1, a_idx(a_r+1, a_c+1): 1},
            {b_idx(b_r, b_c): 1, b_idx(b_r+1, b_c+1): 1},
            {w_idx(c_r, c_c): 1, w_idx(c_r+1, c_c+1): 1}
        )
        # 2. (A21+A22)B11
        add_term(
            {a_idx(a_r+1, a_c): 1, a_idx(a_r+1, a_c+1): 1},
            {b_idx(b_r, b_c): 1},
            {w_idx(c_r+1, c_c): 1, w_idx(c_r+1, c_c+1): 1}
        )
        # 3. A11(B12-B22)
        add_term(
            {a_idx(a_r, a_c): 1},
            {b_idx(b_r, b_c+1): 1, b_idx(b_r+1, b_c+1): -1},
            {w_idx(c_r, c_c+1): 1, w_idx(c_r+1, c_c+1): 1}
        )
        # 4. A22(B21-B11)
        add_term(
            {a_idx(a_r+1, a_c+1): 1},
            {b_idx(b_r+1, b_c): 1, b_idx(b_r, b_c): -1},
            {w_idx(c_r, c_c): 1, w_idx(c_r+1, c_c): 1}
        )
        # 5. (A11+A12)B22
        add_term(
            {a_idx(a_r, a_c): 1, a_idx(a_r, a_c+1): 1},
            {b_idx(b_r+1, b_c+1): 1},
            {w_idx(c_r, c_c): 1, w_idx(c_r, c_c+1): 1}
        )
        # 6. (A21-A11)(B11+B12)
        add_term(
            {a_idx(a_r+1, a_c): 1, a_idx(a_r, a_c): -1},
            {b_idx(b_r, b_c): 1, b_idx(b_r, b_c+1): 1},
            {w_idx(c_r+1, c_c+1): 1}
        )
        # 7. (A12-A22)(B21+B22)
        add_term(
            {a_idx(a_r, a_c+1): 1, a_idx(a_r+1, a_c+1): -1},
            {b_idx(b_r+1, b_c): 1, b_idx(b_r+1, b_c+1): 1},
            {w_idx(c_r, c_c): 1}
        )
        
    # Apply Strassen 4x4x4 to A1 (0:4, 0:4), B1a (0:4, 0:4), C1a (0:4, 0:4)
    # A1 blocks: (0,0), (0,2), (2,0), (2,2)
    # B1a blocks: (0,0), (0,2), (2,0), (2,2)
    # C1a blocks: (0,0), (0,2), (2,0), (2,2)
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is (0,0), A22 is (2,2)
    # This requires a combined Strassen call?
    # No, Strassen 4x4x4 defines 7 products of 2x2 matrices.
    # Each product is computed by a 2x2x2 multiplication.
    # So we call add_strassen_2x2 7 times?
    # No, we call a function that computes M1, M2, etc.
    # M1 = (A11+A22) * (B11+B22).
    # This is a 2x2 matrix multiplication.
    # We can compute it using Strassen 2x2x2.
    # But the inputs are sums of blocks.
    # We need to add terms that represent (A11+A22) * (B11+B22).
    # This expands to A11*B11 + A11*B22 + A22*B11 + A22*B22.
    # This is 4 terms of 2x2x2? No, 4 scalar multiplications?
    # No, A11 is 2x2.
    # So we need to add the rank-1 tensors for these 4 products.
    # Each product is a 2x2x2 tensor.
    # So we add 4 * 8 = 32 terms? No, we can use Strassen for each?
    # This is getting messy.
    
    # Let's just use schoolbook for everything to be safe and ensure validity.
    # Rank 120.
    # The current program returns 120.
    # If I can't find a valid lower rank, I should return 120.
    # But I want to improve.
    
    # Let's try a simple block decomposition:
    # Split A into 2x2 blocks.
    # A (4x5) -> [2x2, 2x2, 2x1; 2x2, 2x2, 2x1]
    # B (5x6) -> [2x3, 2x3; 2x3, 2x3; 1x3, 1x3]
    # C (4x6) -> [2x3, 2x3; 2x3, 2x3]
    
    # This is too complex to implement quickly and correctly.
    
    # I will return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Test
    res = construct()
    print("Rank:", len(res["U"][0]))
