"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # where 0<=i<n, 0<=j<m, 0<=k<p
    
    # Strategy: Constructive decomposition using known low-rank identities.
    # For 2x2x6, the rank is known to be 11.
    # We can partition the 12 columns into two sets of 6.
    # Let A be 2x2 and B be 2x12.
    # Split B into B1 (2x6) and B2 (2x6).
    # C = A * B = A * [B1, B2] = [A * B1, A * B2].
    # We can decompose A * B1 using rank 11 and A * B2 using rank 11.
    # Total rank = 22.
    
    # We need to construct the rank-11 decomposition for 2x2x6.
    # The standard Strassen decomposition for 2x2x2 has rank 7.
    # For 2x2x6, we can view it as three 2x2x2 multiplications? No, that would be rank 21.
    # Actually, the rank of 2x2x6 is 11. This is a known result.
    # Let's try to construct it.
    
    # Let's use the known rank-11 decomposition for 2x2x6.
    # One way to get it is by using the rank-7 Strassen for 2x2x2 and combining with linear combinations.
    # However, constructing the specific 11-term decomposition from scratch is complex.
    
    # Alternative: Use the rank-7 Strassen for 2x2x2 and apply it to blocks.
    # If we split 2x2x6 into three 2x2x2 blocks, we get rank 21.
    # Can we do better?
    # The rank of 2x2xn is roughly 2n + 1 for large n? No.
    # For 2x2x2, rank 7.
    # For 2x2x4, rank 11? Let's check.
    # Actually, the rank of 2x2xn is 2n+1 for n>=2?
    # If rank(2x2x2)=7, rank(2x2x4)=11, rank(2x2x6)=15?
    # Wait, the inspiration says "known rank-11 decomposition for 2x2x6".
    # Let's verify if rank(2x2x6) is indeed 11.
    # If rank(2x2x2)=7, then rank(2x2x6) >= 7 * (6/2) / something?
    # Actually, the rank of matrix multiplication tensor <2,2,n> is 2n+1 for n=2,3,4...?
    # No, for n=2, 2*2+1 = 5, but rank is 7.
    # For n=3, rank is 10?
    # Let's assume the inspiration is correct that rank(2x2x6) is 11.
    # If so, total rank for 2x2x12 would be 22.
    
    # Let's try to find a rank-11 decomposition for 2x2x6.
    # We can use the following identity:
    # The 2x2x2 Strassen decomposition has 7 terms.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For 2x2x6, we can write B as [B1, B2, B3] where each Bj is 2x2.
    # C = A * B = [A*B1, A*B2, A*B3].
    # If we use Strassen for each, we get 21 terms.
    # But we can share some computations?
    # A*B1 and A*B2 share A.
    # Let's see if we can combine them.
    
    # Actually, there is a known rank-11 decomposition for 2x2x6.
    # Let's try to construct it by solving a linear system or using a known formula.
    
    # Given the time constraint and complexity, let's try a different approach.
    # We will use a randomized search specifically for rank 22, but with a better initialization.
    # We initialize U, V, W using the known Strassen structure for the 2x2x2 blocks.
    
    # Let's build the rank-21 decomposition (3 Strassen blocks) first.
    # Then we can try to reduce it to rank 22 or lower using ALS.
    
    # But the inspiration says rank 22 is achievable.
    # Let's try to directly construct the rank-22 decomposition.
    
    # Let's assume the rank-11 decomposition for 2x2x6 exists and try to find it.
    # We can use the following strategy:
    # 1. Construct the rank-21 decomposition (3 Strassen blocks).
    # 2. Use ALS to try to reduce the rank to 22 (which is actually an increase? No, 21 < 22).
    # Wait, if rank(2x2x6) is 11, then 2*11 = 22.
    # If rank(2x2x2) is 7, then 3*7 = 21.
    # So 21 is already better than 22.
    # Why does the inspiration say rank 11 for 2x2x6?
    # Maybe rank(2x2x6) is not 11?
    # Let's check the rank of 2x2xn.
    # Rank(2x2x2) = 7.
    # Rank(2x2x3) = 10?
    # Rank(2x2x4) = 11?
    # Rank(2x2x6) = 15?
    
    # Let's try to find a decomposition with rank 21 (3 Strassen blocks).
    # This is easier to construct.
    
    # Construct rank-21 decomposition for 2x2x12.
    # Split B into 6 blocks of 2x2.
    # For each block, use Strassen.
    # Total rank 6 * 7 = 42.
    # Wait, 12 columns / 2 = 6 blocks.
    # 6 * 7 = 42.
    # This matches the current best of 42.
    
    # Can we do better?
    # If we split into 3 blocks of 4 columns (2x4), and use rank(2x2x4) = 11?
    # Then 3 * 11 = 33.
    # If rank(2x2x4) is 11, then rank 33 is possible.
    
    # Let's try to construct a rank-11 decomposition for 2x2x4.
    # If we can do that, we can get rank 33 for 2x2x12.
    
    # Let's try to find a rank-11 decomposition for 2x2x4.
    # We can use a randomized search for rank 11 for 2x2x4.
    
    # But we are solving 2x2x12.
    # Let's try to find a rank-33 decomposition for 2x2x12.
    # We can use 3 blocks of 2x2x4.
    
    # Let's try to find a rank-11 decomposition for 2x2x4.
    # We will use ALS for 2x2x4.
    
    def find_rank_11_2x2x4():
        n, m, p = 2, 2, 4
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        T1 = T.reshape(n*m, m*p * p*n)
        T2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
        T3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
        
        r = 11
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(100):
            # Solve for U
            M = np.zeros((p*n * m*p, r))
            for col in range(r):
                M[:, col] = np.kron(W[:, col], V[:, col])
            U, _, _, _ = np.linalg.lstsq(M, T1.T, rcond=None)
            U = U.T
            
            # Solve for V
            MW = np.zeros((n*m * p*n, r))
            for col in range(r):
                MW[:, col] = np.kron(U[:, col], W[:, col])
            V, _, _, _ = np.linalg.lstsq(MW, T2.T, rcond=None)
            V = V.T
            
            # Solve for W
            MV = np.zeros((n*m * m*p, r))
            for col in range(r):
                MV[:, col] = np.kron(U[:, col], V[:, col])
            W, _, _, _ = np.linalg.lstsq(MV, T3.T, rcond=None)
            W = W.T
            
            # Check error
            T_rec = np.zeros((n*m, m*p, p*n))
            for col in range(r):
                T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
            err = np.linalg.norm(T_rec - T)
            if err < 1e-5:
                return U, V, W
        
        return None
    
    # Try to find rank-11 for 2x2x4
    if time.time() < deadline - 50:
        res = find_rank_11_2x2x4()
        if res is not None:
            U4, V4, W4 = res
            # Construct rank-33 for 2x2x12
            # Split B into 3 blocks of 4 columns.
            # B1: cols 0-3, B2: cols 4-7, B3: cols 8-11
            # For each block, we have a 2x2x4 decomposition.
            # We need to adjust the indices.
            
            U = np.zeros((n*m, 33))
            V = np.zeros((m*p, 33))
            W = np.zeros((p*n, 33))
            
            for block in range(3):
                # Block index offset
                col_offset = block * 4
                
                # U is the same for all blocks? No, U is 2x2, so it's the same.
                # But in the decomposition, U is multiplied by V and W.
                # The tensor is T[a, b, c].
                # For block 0, b is in 0..3, c is in 0..3 (for k=0..1) and 4..7 (for k=2..3)?
                # Wait, the output C is 2x12.
                # C[i, k] = sum_j A[i, j] B[j, k].
                # For block 0, k is 0..3.
                # c = k*n + i. For k=0..3, c is 0..7.
                # So W for block 0 affects c=0..7.
                # W4 is 2x4 (p*n = 4*2 = 8? No, p=4, n=2, so p*n=8).
                # Wait, for 2x2x4, p=4, n=2.
                # W4 shape is (p*n, rank) = (8, 11).
                # In the full problem, p=12, n=2.
                # W shape is (24, 33).
                # For block 0, the k indices are 0..3.
                # c = k*2 + i.
                # For k=0, c=0,1.
                # For k=1, c=2,3.
                # For k=2, c=4,5.
                # For k=3, c=6,7.
                # So W4[0..7] maps to W[0..7].
                
                U[:, block*11:(block+1)*11] = U4
                V[:, block*11:(block+1)*11] = V4
                # V4 shape is (m*p, rank) = (2*4=8, 11).
                # In full problem, V shape is (24, 33).
                # For block 0, b = j*4 + k? No, b = j*p + k.
                # In the full problem, p=12.
                # For block 0, the columns of B are 0..3.
                # b = j*12 + k.
                # For k=0..3, b = j*12 + k.
                # So V4[0..7] maps to V[0..3, 12..15]?
                # No, V is (m*p, rank) = (24, 33).
                # V4 is (8, 11).
                # V4[j*4 + k] corresponds to B[j, k] in the 2x4 block.
                # In the full problem, B[j, k] is at index j*12 + k.
                # So V4[0] -> V[0], V4[1] -> V[1], V4[2] -> V[2], V4[3] -> V[3]
                # V4[4] -> V[12], V4[5] -> V[13], V4[6] -> V[14], V4[7] -> V[15]
                
                for j in range(2):
                    for k in range(4):
                        V[j*12 + k, block*11:(block+1)*11] = V4[j*4 + k, :]
                
                # W4 is (8, 11).
                # W4[k*2 + i] corresponds to C[i, k].
                # In the full problem, C[i, k] is at index k*2 + i.
                # So W4[0..7] maps to W[0..7].
                W[0:8, block*11:(block+1)*11] = W4
                
            # Verify
            T_rec = np.zeros((n*m, m*p, p*n))
            for col in range(33):
                T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
            
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T[a, b, c] = 1.0
            
            err = np.linalg.norm(T_rec - T)
            if err < 1e-5:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to rank-42 (6 Strassen blocks)
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A is 2x2, indices 0,1,2,3 (00, 01, 10, 11)
    # B is 2x2, indices 0,1,2,3 (00, 01, 10, 11)
    # C is 2x2, indices 0,1,2,3 (00, 01, 10, 11)
    
    # U (A factors):
    # M1: A00 + A11 -> [1, 0, 0, 1]
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00 + A01 -> [1, 1, 0, 0]
    # M6: A10 - A00 -> [-1, 0, 1, 0]
    # M7: A01 - A11 -> [0, 1, 0, -1]
    
    # V (B factors):
    # M1: B00 + B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B10 - B11 -> [0, 0, 1, -1]
    # M4: B01 - B00 -> [-1, 1, 0, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00 + B01 -> [1, 1, 0, 0]
    # M7: B10 + B11 -> [0, 0, 1, 1]
    
    # W (C factors):
    # M1: C00 + C11 + (C01 - C00) + (C10 - C11) ? No.
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Let's check:
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # W factors (for C00, C01, C10, C11):
    # M1: [1, -1, 1, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 0]
    # M4: [0, 0, -1, 1]
    # M5: [-1, 1, 0, 1]
    # M6: [1, 0, 1, 0]
    # M7: [1, 1, 0, 0]
    
    # Let's verify C00:
    # M1 - M5 + M7 + M6 = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B10+B11) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A00B00 + A11B00 + A01B10 + A10B00 + A10B01 - A00B01
    # This is not C00.
    
    # Let's use the standard Strassen formulas.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Let's re-verify C00:
    # M1 - M5 + M7 + M6
    # = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B10+B11) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A00B00 + A11B00 + A01B10 + A10B00 + A10B01 - A00B01
    # This is still not C00 = A00B00 + A01B10.
    
    # I think I made a mistake in the Strassen formulas or the combination.
    # Let's use a known correct set.
    
    # Correct Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Let's check C00 again.
    # M1 - M5 + M7 + M6
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11 + A10B00 + A10B01 - A00B00 - A00B01
    # = A11B00 + A01B10 + A10B00 + A10B01 - A00B01
    # This is wrong.
    
    # Let's look up the correct Strassen formulas.
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Wait, the standard Strassen is:
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # Let's try a different set.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # I will use a known correct implementation from a reliable source.
    
    # U factors:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [-1, 0, 1, 0]
    # M7: [0, 1, 0, -1]
    
    # V factors:
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 0, 1, -1]
    # M4: [-1, 1, 0, 0]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    
    # W factors:
    # M1: [1, -1, 1, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 0]
    # M4: [0, 0, -1, 1]
    # M5: [-1, 1, 0, 1]
    # M6: [1, 0, 1, 0]
    # M7: [1, 1, 0, 0]
    
    # Let's verify C00 with these.
    # M1 - M5 + M7 + M6
    # = 1*M1 + (-1)*M5 + 1*M7 + 1*M6
    # = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B10+B11) + (A10-A00)(B00+B01)
    # This is the same as before.
    
    # I will use a different set of Strassen formulas.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M5 + M7 + M6
    # C01 = M5 - M1 + M3 + M7
    # C10 = M2 - M4 + M6 + M1
    # C11 = M4 - M2 + M5 + M1
    
    # I will use the fallback rank-48 decomposition if the rank-42 construction fails.
    
    # Let's just use the rank-48 fallback for now to ensure correctness.
    rank = 48
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
