import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    
    # Start with the known rank-48 split (schoolbook for 4x4*4x4 and 4x4*4x2)
    best_factors = _simple_split(4, 4, 6)
    best_rank = 48
    
    # Try to find a better decomposition using Strassen for the 4x4 part
    # 4x4 * 4x4 Strassen has rank 49, which is worse than schoolbook 16.
    # So Strassen for 4x4 is not helpful here.
    
    # However, we can try to optimize the 4x4 * 4x2 part.
    # The 4x4 * 4x2 part is essentially 4 independent 4x4 * 4x1 multiplications? No.
    # It is A (4x4) * B (4x2).
    # This is equivalent to computing 2 columns of C = A*B.
    # Each column is a 4x4 * 4x1 multiplication.
    # Rank of 4x4 * 4x1 is 16 (schoolbook).
    # So 2 columns -> rank 32.
    # This is the same as the schoolbook split.
    
    # Is there a rank-deficient structure?
    # For 4x4 * 4x2, the tensor is T_{i,j,k} = delta_{j,k} ? No.
    # The multiplication tensor for A(n,m) * B(m,p) -> C(n,p)
    # T_{i,j;k,l} = delta_{j,k} * delta_{i,l} ? No.
    # The tensor is T_{(i,j), (k,l), (r,s)} = delta_{j,k} delta_{i,r} ?
    # Let's stick to the rank-48 solution which is safe and correct.
    
    # Wait, can we do better than 48?
    # Schoolbook 4x4*4x6 is 96.
    # Split 4x4*4x4 (16) + 4x4*4x2 (32) = 48.
    # Is there a decomposition of 4x4*4x2 with rank < 32?
    # 4x4*4x2 is equivalent to 2 independent 4x4*4x1 multiplications?
    # No, A is shared.
    # C = A * B.
    # B = [b1, b2].
    # C = [A*b1, A*b2].
    # This is a bilinear map from A (16 vars) and B (8 vars) to C (8 vars).
    # The rank is at least max(16, 8, 8) = 16? No, rank is at least max(nm, mp, np) ?
    # Lower bound for rank of matrix multiplication is max(nm, mp, np).
    # Here nm=16, mp=8, np=8. So rank >= 16.
    # Schoolbook is 32.
    # Can we do better than 32 for 4x4*4x2?
    # This is a known problem.
    # For 2x2 * 2x2, rank is 7 (Strassen).
    # For 2x2 * 2x1, rank is 4 (schoolbook).
    # For 4x4 * 4x2, we can view it as:
    # Split A into 2x2 blocks: A00, A01, A10, A11.
    # Split B into 2x2 blocks (vertically): B_top (2x2), B_bot (2x2).
    # C_top = A00 B_top + A01 B_bot (2x2 * 2x2 + 2x2 * 2x2)
    # C_bot = A10 B_top + A11 B_bot (2x2 * 2x2 + 2x2 * 2x2)
    # Each term is 2x2 * 2x2.
    # We can use Strassen for each 2x2 * 2x2?
    # C_top = M1 + M2 where M1 = A00 B_top, M2 = A01 B_bot.
    # Rank of M1 is 7, M2 is 7. Total 14?
    # But we can combine them?
    # Actually, C_top is a 2x2 matrix.
    # We are computing 4 such 2x2 matrices (C00, C01, C10, C11 of the 4x2 result).
    # Wait, C is 4x2.
    # C = [ C00 C01 ]
    #     [ C10 C11 ]
    # where C00, C01, C10, C11 are 2x1 vectors? No, 2x1 blocks?
    # Let's be precise.
    # A is 4x4. B is 4x2.
    # A = [ A00 A01 ] (2x2 blocks)
    #     [ A10 A11 ]
    # B = [ B0 ] (2x2)
    #     [ B1 ] (2x2)
    # C = [ C0 ] (2x2)
    #     [ C1 ] (2x2)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # C0 is a 2x2 matrix.
    # C0 = A00 B0 + A01 B1.
    # This is the sum of two 2x2 matrix products.
    # We can compute A00 B0 with rank 7 (Strassen) and A01 B1 with rank 7.
    # Then add them. The addition is linear, so it doesn't add rank if we output the sum?
    # No, the tensor decomposition requires us to output the final C.
    # If we use separate rank-7 decompositions for A00B0 and A01B1, we get two intermediate results.
    # We need to sum them to get C0.
    # The decomposition is:
    # C0 = sum_{r=1}^7 U_r(A00) V_r(B0) W_r + sum_{s=1}^7 U_s(A01) V_s(B1) W_s
    # This means we have 14 terms for C0.
    # Similarly 14 terms for C1.
    # Total rank = 28.
    # 28 < 32.
    
    # So we can get rank 16 (for 4x4*4x4) + 28 (for 4x4*4x2) = 44.
    
    # Let's implement the rank 28 decomposition for 4x4*4x2.
    # And keep the rank 16 schoolbook for 4x4*4x4.
    # Total rank 44.
    
    try:
        factors_44_42 = _strassen_4x4_4x2()
        if factors_44_42 is not None:
            rank_44_42 = factors_44_42[0].shape[1]
            # 4x4*4x4 schoolbook is 16.
            # We need to combine these.
            # The 4x4*4x4 part uses B cols 0-3.
            # The 4x4*4x2 part uses B cols 4-5.
            # We can just concatenate the columns of U, V, W.
            
            U_left, V_left, W_left = _schoolbook_4x4_4x4()
            rank_left = U_left.shape[1]
            
            U_new = np.hstack([U_left, factors_44_42[0]])
            V_new = np.hstack([V_left, factors_44_42[1]])
            W_new = np.hstack([W_left, factors_44_42[2]])
            
            # Check validity?
            # We should verify that the decomposition is correct.
            # For 4x4*4x2, the indices are:
            # U: 16 rows (A indices)
            # V: 8 rows (B indices for cols 4,5) -> Wait, V in the full problem has 24 rows.
            # The factors from _strassen_4x4_4x2 should be shaped for the sub-problem.
            # Sub-problem: A (4x4), B_sub (4x2).
            # U_sub: (16, 28)
            # V_sub: (8, 28) -> B_sub indices 0..7 correspond to B[0,4], B[1,4], B[0,5], B[1,5], etc.
            # W_sub: (8, 28) -> C_sub indices 0..7 correspond to C[0,4], C[1,4], etc.
            
            # We need to embed these into the full U, V, W.
            # Full U: (16, total_rank). The rows are the same (A indices).
            # Full V: (24, total_rank). The rows for B_sub are 16..23 (j*p + k where j=0..3, k=4,5).
            # Wait, B index is j*p + k.
            # For B_sub (cols 4,5), the indices are:
            # j=0, k=4 -> 4
            # j=0, k=5 -> 5
            # j=1, k=4 -> 10
            # j=1, k=5 -> 11
            # j=2, k=4 -> 16
            # j=2, k=5 -> 17
            # j=3, k=4 -> 22
            # j=3, k=5 -> 23
            
            # The V_sub from _strassen_4x4_4x2 will have rows 0..7 corresponding to B_sub[0..7].
            # We need to map B_sub row r to full V row.
            # B_sub is 4x2. Row-major:
            # 0: B[0,4]
            # 1: B[0,5]
            # 2: B[1,4]
            # 3: B[1,5]
            # 4: B[2,4]
            # 5: B[2,5]
            # 6: B[3,4]
            # 7: B[3,5]
            
            # Full V indices:
            # B[0,4] -> 0*6 + 4 = 4
            # B[0,5] -> 0*6 + 5 = 5
            # B[1,4] -> 1*6 + 4 = 10
            # B[1,5] -> 1*6 + 5 = 11
            # B[2,4] -> 2*6 + 4 = 16
            # B[2,5] -> 2*6 + 5 = 17
            # B[3,4] -> 3*6 + 4 = 22
            # B[3,5] -> 3*6 + 5 = 23
            
            # Similarly for W_sub.
            # W_sub is 8x28. Rows 0..7 correspond to C_sub[0..7].
            # C_sub is 4x2. Row-major:
            # 0: C[0,4]
            # 1: C[0,5]
            # 2: C[1,4]
            # 3: C[1,5]
            # 4: C[2,4]
            # 5: C[2,5]
            # 6: C[3,4]
            # 7: C[3,5]
            
            # Full W indices:
            # C[0,4] -> 4*4 + 0 = 16
            # C[0,5] -> 5*4 + 0 = 20
            # C[1,4] -> 4*4 + 1 = 17
            # C[1,5] -> 5*4 + 1 = 21
            # C[2,4] -> 4*4 + 2 = 18
            # C[2,5] -> 5*4 + 2 = 22
            # C[3,4] -> 4*4 + 3 = 19
            # C[3,5] -> 5*4 + 3 = 23
            
            # Note: The W index is k*n + i.
            # C[i,k] -> k*n + i.
            
            # Let's build the full factors.
            total_rank = rank_left + rank_44_42
            U_full = np.zeros((16, total_rank))
            V_full = np.zeros((24, total_rank))
            W_full = np.zeros((24, total_rank))
            
            # Copy left part
            U_full[:, :rank_left] = U_left
            V_full[:, :rank_left] = V_left
            W_full[:, :rank_left] = W_left
            
            # Copy right part
            U_full[:, rank_left:] = factors_44_42[0]
            
            # Map V_sub to V_full
            V_sub = factors_44_42[1]
            v_map = [4, 5, 10, 11, 16, 17, 22, 23]
            for r_sub, r_full in enumerate(v_map):
                V_full[r_full, rank_left:] = V_sub[r_sub, :]
                
            # Map W_sub to W_full
            W_sub = factors_44_42[2]
            w_map = [16, 20, 17, 21, 18, 22, 19, 23]
            for r_sub, r_full in enumerate(w_map):
                W_full[r_full, rank_left:] = W_sub[r_sub, :]
                
            # Verify
            if _verify_decomposition(U_full, V_full, W_full, 4, 4, 6):
                best_factors = (U_full, V_full, W_full)
                best_rank = total_rank
    except Exception as e:
        pass
        
    return {
        "U": [list(map(float, row)) for row in best_factors[0]],
        "V": [list(map(float, row)) for row in best_factors[1]],
        "W": [list(map(float, row)) for row in best_factors[2]]
    }

def _schoolbook_4x4_4x4():
    n, m, p = 4, 4, 4
    rank = 16
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4 + j, col] = 1.0
                V[j*4 + k, col] = 1.0
                W[k*4 + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_4x4_4x2():
    # Decomposition for A(4x4) * B(4x2) -> C(4x2)
    # Rank 28
    # U: (16, 28)
    # V: (8, 28)
    # W: (8, 28)
    
    # Strassen 2x2 factors
    U_s, V_s, W_s = _strassen2x2()
    
    U = np.zeros((16, 28))
    V = np.zeros((8, 28))
    W = np.zeros((8, 28))
    
    # C0 = A00 B0 + A01 B1
    # A00: rows 0-3, cols 0-3 of A. U indices: 0,1,4,5 (row-major 2x2)
    # A01: rows 0-3, cols 4-7 of A. U indices: 2,3,6,7
    # B0: rows 0-3 of B_sub. V indices: 0,1,4,5
    # B1: rows 4-7 of B_sub. V indices: 2,3,6,7
    # C0: rows 0-3 of C_sub. W indices: 0,1,4,5
    
    # Strassen steps for A00 B0:
    # M1 = (A00+A11)(B00+B11)
    # In our notation, A00 is the top-left 2x2 block of A.
    # Let's denote the 2x2 blocks of A as:
    # A = [ A00 A01 ]
    #     [ A10 A11 ]
    # And B_sub = [ B0 ]
    #             [ B1 ]
    
    # We want to compute A00 B0.
    # This is a 2x2 * 2x2 multiplication.
    # We use the Strassen factors U_s, V_s, W_s.
    # U_s rows correspond to A00 indices: 0,1,4,5 (in full A)
    # V_s rows correspond to B0 indices: 0,1,4,5 (in full B_sub)
    # W_s rows correspond to C0 indices: 0,1,4,5 (in full C_sub)
    
    # We need to map the Strassen indices to the full indices.
    # Strassen U_s is 4x7. Rows 0,1,2,3 correspond to A00[0,0], A00[0,1], A00[1,0], A00[1,1].
    # Full A indices:
    # A00[0,0] -> 0
    # A00[0,1] -> 1
    # A00[1,0] -> 4
    # A00[1,1] -> 5
    
    # Strassen V_s is 4x7. Rows 0,1,2,3 correspond to B0[0,0], B0[0,1], B0[1,0], B0[1,1].
    # Full B_sub indices:
    # B0[0,0] -> 0
    # B0[0,1] -> 1
    # B0[1,0] -> 4
    # B0[1,1] -> 5
    
    # Strassen W_s is 4x7. Rows 0,1,2,3 correspond to C0[0,0], C0[0,1], C0[1,0], C0[1,1].
    # Full C_sub indices:
    # C0[0,0] -> 0
    # C0[0,1] -> 1
    # C0[1,0] -> 4
    # C0[1,1] -> 5
    
    # We will fill in the U, V, W matrices for the 4 Strassen terms:
    # 1. A00 B0
    # 2. A01 B1
    # 3. A10 B0
    # 4. A11 B1
    
    # Term 1: A00 B0
    # U rows: 0,1,4,5. V rows: 0,1,4,5. W rows: 0,1,4,5.
    U[0, 0:7] = U_s[0, :]
    U[1, 0:7] = U_s[1, :]
    U[4, 0:7] = U_s[2, :]
    U[5, 0:7] = U_s[3, :]
    
    V[0, 0:7] = V_s[0, :]
    V[1, 0:7] = V_s[1, :]
    V[4, 0:7] = V_s[2, :]
    V[5, 0:7] = V_s[3, :]
    
    W[0, 0:7] = W_s[0, :]
    W[1, 0:7] = W_s[1, :]
    W[4, 0:7] = W_s[2, :]
    W[5, 0:7] = W_s[3, :]
    
    # Term 2: A01 B1
    # A01 indices: 2,3,6,7. B1 indices: 2,3,6,7. C0 indices: 0,1,4,5.
    U[2, 7:14] = U_s[0, :]
    U[3, 7:14] = U_s[1, :]
    U[6, 7:14] = U_s[2, :]
    U[7, 7:14] = U_s[3, :]
    
    V[2, 7:14] = V_s[0, :]
    V[3, 7:14] = V_s[1, :]
    V[6, 7:14] = V_s[2, :]
    V[7, 7:14] = V_s[3, :]
    
    W[0, 7:14] += W_s[0, :]
    W[1, 7:14] += W_s[1, :]
    W[4, 7:14] += W_s[2, :]
    W[5, 7:14] += W_s[3, :]
    
    # Term 3: A10 B0
    # A10 indices: 8,9,12,13. B0 indices: 0,1,4,5. C1 indices: 2,3,6,7.
    U[8, 14:21] = U_s[0, :]
    U[9, 14:21] = U_s[1, :]
    U[12, 14:21] = U_s[2, :]
    U[13, 14:21] = U_s[3, :]
    
    V[0, 14:21] += V_s[0, :]
    V[1, 14:21] += V_s[1, :]
    V[4, 14:21] += V_s[2, :]
    V[5, 14:21] += V_s[3, :]
    
    W[2, 14:21] = W_s[0, :]
    W[3, 14:21] = W_s[1, :]
    W[6, 14:21] = W_s[2, :]
    W[7, 14:21] = W_s[3, :]
    
    # Term 4: A11 B1
    # A11 indices: 10,11,14,15. B1 indices: 2,3,6,7. C1 indices: 2,3,6,7.
    U[10, 21:28] = U_s[0, :]
    U[11, 21:28] = U_s[1, :]
    U[14, 21:28] = U_s[2, :]
    U[15, 21:28] = U_s[3, :]
    
    V[2, 21:28] += V_s[0, :]
    V[3, 21:28] += V_s[1, :]
    V[6, 21:28] += V_s[2, :]
    V[7, 21:28] += V_s[3, :]
    
    W[2, 21:28] += W_s[0, :]
    W[3, 21:28] += W_s[1, :]
    W[6, 21:28] += W_s[2, :]
    W[7, 21:28] += W_s[3, :]
    
    return U, V, W

def _strassen2x2():
    U = np.array([
        [1, 0, 0, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1], # B10
        [1, 0, -1, -1, 1, 0, 1] # B11
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, -1, 0, 1, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, 0, 0, -1, -1, 0, 0] # C11
    ], dtype=float)
    
    return U, V, W

def _simple_split(n, m, p):
    rank = 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
            for k in range(4, p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    # Check if U * V * W reconstructs the tensor
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We check a few random points.
    for _ in range(10):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        val = 0.0
        for r in range(U.shape[1]):
            val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            
        # Expected value is 1.0
        if abs(val - 1.0) > 1e-6:
            return False
            
    # Check off-diagonals
    for _ in range(10):
        i1 = random.randint(0, n-1)
        i2 = random.randint(0, n-1)
        if i1 == i2: continue
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        val = 0.0
        for r in range(U.shape[1]):
            val += U[i1*m + j, r] * V[j*p + k, r] * W[k*n + i2, r]
            
        if abs(val) > 1e-6:
            return False
            
    return True
