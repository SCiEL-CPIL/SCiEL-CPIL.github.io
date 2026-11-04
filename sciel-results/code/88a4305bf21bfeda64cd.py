# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 110 based on embedding Strassen 4x4 and schoolbook for the rest.
    # Blocks:
    # 1. 4x4 core (Strassen, rank 49)
    # 2. 4x4 cross (A_1..4,5 * B_5,1..4) (rank 16)
    # 3. 4x1 col (A_1..4,1..4 * B_1..4,5) (rank 16)
    # 4. 4x1 corner (A_1..4,5 * B_5,5) (rank 4)
    # 5. 1x4 row (A_5,1..4 * B_1..4,1..4) (rank 16)
    # 6. 1x4 corner (A_5,5 * B_5,1..4) (rank 4)
    # 7. 1x1 core (A_5,1..4 * B_1..4,5) (rank 4)
    # 8. 1x1 corner (A_5,5 * B_5,5) (rank 1)
    # Total: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110

    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return U, V, W

    # Helper to create Strassen 2x2 decomposition (rank 7)
    def strassen_2x2():
        # Standard Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices:
        # A: 00, 01, 10, 11 -> 0,1,2,3
        # B: 00, 01, 10, 11 -> 0,1,2,3
        # C: 00, 01, 10, 11 -> 0,1,2,3
        
        # M1
        U[0,0] += 1; U[3,0] += 1
        V[0,0] += 1; V[3,0] += 1
        W[0,0] += 1; W[3,0] += 1
        
        # M2
        U[2,1] += 1; U[3,1] += 1
        V[0,1] += 1
        W[2,1] += 1
        
        # M3
        U[0,2] += 1
        V[2,2] += 1; V[3,2] -= 1
        W[1,2] += 1; W[2,2] += 1
        
        # M4
        U[3,3] += 1
        V[1,3] += 1; V[0,3] -= 1
        W[0,3] += 1; W[2,3] += 1
        
        # M5
        U[0,4] += 1; U[1,4] += 1
        V[3,4] += 1
        W[0,4] -= 1; W[1,4] += 1
        
        # M6
        U[2,5] += 1; U[0,5] -= 1
        V[0,5] += 1; V[1,5] += 1
        W[3,5] += 1
        
        # M7
        U[1,6] += 1; U[3,6] -= 1
        V[2,6] += 1; V[3,6] += 1
        W[0,6] += 1
        
        return U, V, W

    # Tensor product of two decompositions
    def tensor_product(U1, V1, W1, U2, V2, W2):
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 * R2
        
        n1, m1, p1 = U1.shape[0], V1.shape[0], W1.shape[0]
        n2, m2, p2 = U2.shape[0], V2.shape[0], W2.shape[0]
        
        # Dimensions must match: p1 == n2, m1 == m2, p2 == p1?
        # For matrix multiplication, U is (n*m, R), V is (m*p, R), W is (p*n, R)
        # Tensor product of M_{n1,m1,p1} and M_{n2,m2,p2}
        # Requires p1 == n2, m1 == m2, p2 == p1? No.
        # Standard tensor product for matrix mult:
        # (A1 B1) (A2 B2) = (A1 A2) (B1 B2)
        # n = n1*n2, m = m1*m2, p = p1*p2
        # Requires p1 == m2? No.
        # A1 is n1 x m1. B1 is m1 x p1.
        # A2 is n2 x m2. B2 is m2 x p2.
        # For product to make sense in tensor product sense:
        # We are computing (A1 \otimes A2) (B1 \otimes B2) = (A1 B1) \otimes (A2 B2) ? No.
        # It is (A1 \otimes A2) (B1 \otimes B2) = (A1 B1) \otimes (A2 B2) is false.
        # It is (A1 \otimes A2) (B1 \otimes B2) = (A1 B1) \otimes (A2 B2) if dimensions align.
        # A1: n1 x m1. B1: m1 x p1.
        # A2: n2 x m2. B2: m2 x p2.
        # A1 \otimes A2: n1*n2 x m1*m2.
        # B1 \otimes B2: m1*m2 x p1*p2.
        # Result: n1*n2 x p1*p2.
        # This works if we treat the multiplication as independent.
        # The rank is R1 * R2.
        
        U = np.zeros((n1 * n2 * m1 * m2, R)) # Wait, U shape is (n*m, R)
        # n = n1*n2, m = m1*m2.
        # U index: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
        # This is getting complicated.
        # Simpler: Just use schoolbook for 4x4? No, rank 64 vs 49.
        # Let's just use the known Strassen 4x4 rank 49.
        # I will hardcode the 4x4 Strassen decomposition or generate it carefully.
        
        # Actually, for 4x4, we can just use schoolbook (64) if 49 is too hard to construct correctly in code.
        # 110 with 49 core. If we use 64 core:
        # 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
        # So using schoolbook for 4x4 gives 125. No improvement.
        # We MUST use Strassen 4x4 (rank 49) to get improvement.
        
        # Let's implement the tensor product correctly.
        # U_total shape: (n1*n2 * m1*m2, R1*R2)
        # V_total shape: (m1*m2 * p1*p2, R1*R2)
        # W_total shape: (p1*p2 * n1*n2, R1*R2)
        
        U_out = np.zeros((n1 * n2 * m1 * m2, R))
        V_out = np.zeros((m1 * m2 * p1 * p2, R))
        W_out = np.zeros((p1 * p2 * n1 * n2, R))
        
        for r1 in range(R1):
            for r2 in range(R2):
                r = r1 * R2 + r2
                # u1, v1, w1 are vectors
                u1 = U1[:, r1]
                v1 = V1[:, r1]
                w1 = W1[:, r1]
                u2 = U2[:, r2]
                v2 = V2[:, r2]
                w2 = W2[:, r2]
                
                # Outer products
                # U: u1 \otimes u2
                # Index in U1: i1*m1 + j1
                # Index in U2: i2*m2 + j2
                # Index in U_out: (i1*n2 + i2)*(m1*m2) + (j1*m2 + j2)
                
                for idx1 in range(len(u1)):
                    if u1[idx1] == 0: continue
                    i1 = idx1 // m1
                    j1 = idx1 % m1
                    for idx2 in range(len(u2)):
                        if u2[idx2] == 0: continue
                        i2 = idx2 // m2
                        j2 = idx2 % m2
                        out_idx = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                        U_out[out_idx, r] += u1[idx1] * u2[idx2]
                        
                # V: v1 \otimes v2
                # Index in V1: j1*p1 + k1
                # Index in V2: j2*p2 + k2
                # Index in V_out: (j1*m2 + j2)*(p1*p2) + (k1*p2 + k2)
                # Wait, V1 is m1 x p1. V2 is m2 x p2.
                # V_out is (m1*m2) x (p1*p2).
                # Index: (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                
                for idx1 in range(len(v1)):
                    if v1[idx1] == 0: continue
                    j1 = idx1 // p1
                    k1 = idx1 % p1
                    for idx2 in range(len(v2)):
                        if v2[idx2] == 0: continue
                        j2 = idx2 // p2
                        k2 = idx2 % p2
                        out_idx = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                        V_out[out_idx, r] += v1[idx1] * v2[idx2]
                        
                # W: w1 \otimes w2
                # Index in W1: k1*n1 + i1
                # Index in W2: k2*n2 + i2
                # Index in W_out: (k1*p2 + k2)*(n1*n2) + (i1*n2 + i2)
                
                for idx1 in range(len(w1)):
                    if w1[idx1] == 0: continue
                    k1 = idx1 // n1
                    i1 = idx1 % n1
                    for idx2 in range(len(w2)):
                        if w2[idx2] == 0: continue
                        k2 = idx2 // n2
                        i2 = idx2 % n2
                        out_idx = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                        W_out[out_idx, r] += w1[idx1] * w2[idx2]
                        
        return U_out, V_out, W_out

    # Get 2x2 Strassen
    U2, V2, W2 = strassen_2x2()
    
    # Get 4x4 Strassen (Tensor product of two 2x2)
    U4, V4, W4 = tensor_product(U2, V2, W2, U2, V2, W2)
    
    # Now construct the 5x5 decomposition
    # We need to map the 4x4 indices (0..3) to 5x5 indices (0..4)
    # And add the schoolbook parts for the 5th row/col.
    
    # Total Rank
    R_total = 110
    
    U = np.zeros((25, R_total))
    V = np.zeros((25, R_total))
    W = np.zeros((25, R_total))
    
    col_offset = 0
    
    # 1. Core 4x4 (Strassen)
    # Maps 0..3 to 0..3
    U[:, col_offset:col_offset+49] = U4
    V[:, col_offset:col_offset+49] = V4
    W[:, col_offset:col_offset+49] = W4
    col_offset += 49
    
    # 2. Cross 4x4 (A_1..4,5 * B_5,1..4)
    # A indices: i=0..3, j=4. U index: i*5+4
    # B indices: j=4, k=0..3. V index: 4*5+k
    # C indices: k=0..3, i=0..3. W index: k*5+i
    # Rank 16.
    for i in range(4):
        for k in range(4):
            U[i*5+4, col_offset] = 1.0
            V[4*5+k, col_offset] = 1.0
            W[k*5+i, col_offset] = 1.0
            col_offset += 1
            
    # 3. Col 4x1 (A_1..4,1..4 * B_1..4,5)
    # A: i=0..3, j=0..3. U: i*5+j
    # B: j=0..3, k=4. V: j*5+4
    # C: k=4, i=0..3. W: 4*5+i
    for i in range(4):
        for j in range(4):
            U[i*5+j, col_offset] = 1.0
            V[j*5+4, col_offset] = 1.0
            W[4*5+i, col_offset] = 1.0
            col_offset += 1
            
    # 4. Corner 4x1 (A_1..4,5 * B_5,5)
    # A: i=0..3, j=4. U: i*5+4
    # B: j=4, k=4. V: 4*5+4
    # C: k=4, i=0..3. W: 4*5+i
    for i in range(4):
        U[i*5+4, col_offset] = 1.0
        V[4*5+4, col_offset] = 1.0
        W[4*5+i, col_offset] = 1.0
        col_offset += 1
        
    # 5. Row 1x4 (A_5,1..4 * B_1..4,1..4)
    # A: i=4, j=0..3. U: 4*5+j
    # B: j=0..3, k=0..3. V: j*5+k
    # C: k=0..3, i=4. W: k*5+4
    for j in range(4):
        for k in range(4):
            U[4*5+j, col_offset] = 1.0
            V[j*5+k, col_offset] = 1.0
            W[k*5+4, col_offset] = 1.0
            col_offset += 1
            
    # 6. Corner 1x4 (A_5,5 * B_5,1..4)
    # A: i=4, j=4. U: 4*5+4
    # B: j=4, k=0..3. V: 4*5+k
    # C: k=0..3, i=4. W: k*5+4
    for k in range(4):
        U[4*5+4, col_offset] = 1.0
        V[4*5+k, col_offset] = 1.0
        W[k*5+4, col_offset] = 1.0
        col_offset += 1
        
    # 7. Core 1x1 (A_5,1..4 * B_1..4,5)
    # A: i=4, j=0..3. U: 4*5+j
    # B: j=0..3, k=4. V: j*5+4
    # C: k=4, i=4. W: 4*5+4
    for j in range(4):
        U[4*5+j, col_offset] = 1.0
        V[j*5+4, col_offset] = 1.0
        W[4*5+4, col_offset] = 1.0
        col_offset += 1
        
    # 8. Corner 1x1 (A_5,5 * B_5,5)
    U[4*5+4, col_offset] = 1.0
    V[4*5+4, col_offset] = 1.0
    W[4*5+4, col_offset] = 1.0
    col_offset += 1
    
    # Verify col_offset is 110
    assert col_offset == 110, f"Rank mismatch: {col_offset}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
