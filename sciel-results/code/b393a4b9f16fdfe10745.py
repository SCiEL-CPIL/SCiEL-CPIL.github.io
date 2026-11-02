import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Strategy: 
    # 2x4 x 4x6 -> split B into two 4x3 parts
    # Each 2x4 x 4x3 -> split A into two 2x2, B into two 2x3
    # Each 2x2 x 2x3 -> split B into 2x1 + 2x2
    # 2x2 x 2x1: rank 2 (schoolbook)
    # 2x2 x 2x2: rank 7 (Strassen)
    # So 2x2 x 2x3: rank 9
    # 2x4 x 4x3: rank 18
    # 2x4 x 4x6: rank 36
    
    def build_strassen_2x2x2():
        """Returns U, V, W for 2x2 x 2x2 with rank 7."""
        rank = 7
        U = np.zeros((4, rank), dtype=np.int32)
        V = np.zeros((4, rank), dtype=np.int32)
        W = np.zeros((4, rank), dtype=np.int32)
        
        # Strassen formulas:
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B10-B11)
        # m4 = A11(B01-B10)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B10)
        # m7 = (A01-A11)(B01+B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C10, 2=C01, 3=C11 (c = k*2+i)
        
        # Term 1: (A00+A11)(B00+B11)
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1]  # C00+=1, C11+=1
        
        # Term 2: (A10+A11)B00
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 1, 0, -1]  # C10+=1, C11+=-1
        
        # Term 3: A00(B10-B11)
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, -1]
        W[:, 2] = [0, 0, 1, 1]  # C01+=1, C11+=1
        
        # Term 4: A11(B01-B10)
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, 1, -1, 0]
        W[:, 3] = [1, 1, 0, 0]  # C00+=1, C10+=1
        
        # Term 5: (A00+A01)B11
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [-1, 0, 1, 0]  # C00+=-1, C01+=1
        
        # Term 6: (A10-A00)(B00+B10)
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 0, 1, 0]
        W[:, 5] = [0, 0, 0, 1]  # C11+=1
        
        # Term 7: (A01-A11)(B01+B11)
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 1, 0, 1]
        W[:, 7-1] = [1, 0, 0, 0]  # C00+=1
        
        return U, V, W, rank
    
    def build_2x2x2x1():
        """Returns U, V, W for 2x2 x 2x1 with rank 2."""
        rank = 2
        U = np.zeros((4, rank), dtype=np.int32)
        V = np.zeros((2, rank), dtype=np.int32)
        W = np.zeros((2, rank), dtype=np.int32)
        
        # C = A * B where B is 2x1 (a column vector)
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # B indices: 0=B0, 1=B1
        # C indices: 0=C0, 1=C1
        
        # Term 1: A00*B0 + A10*B0 = (A00 for C0, A10 for C1) * B0
        U[:, 0] = [1, 0, 1, 0]
        V[:, 0] = [1, 0]
        W[:, 0] = [1, 1]  # Wait, this isn't right for rank-1
        
        # Actually for rank-1: U[a]*V[b]*W[c]
        # We need: sum over terms of U[a]*V[b]*W[c] = T[a,b,c]
        # T[0,0,0]=1 (A00*B0->C0), T[2,0,0]=1 (A10*B0->C0)? No.
        # T[a,b,c]: a is A index, b is B index, c is C index
        # A00 is a=0, B0 is b=0, C0 is c=0: T[0,0,0]=1
        # A01 is a=1, B1 is b=1, C0 is c=0: T[1,1,0]=1
        # A10 is a=2, B0 is b=0, C1 is c=1: T[2,0,1]=1
        # A11 is a=3, B1 is b=1, C1 is c=1: T[3,1,1]=1
        
        # Term 1: covers T[0,0,0] and T[2,0,1]
        # U[:,0] = [1,0,1,0], V[:,0] = [1,0], W[:,0] = [1,1]
        # Check: U[0]*V[0]*W[0] = 1*1*1 = 1 -> T[0,0,0] ✓
        # U[2]*V[0]*W[1] = 1*1*1 = 1 -> T[2,0,1] ✓
        # U[1]*V[0]*W[0] = 0*1*1 = 0 -> T[1,0,0] should be 0 ✓
        # U[3]*V[0]*W[1] = 0*1*1 = 0 -> T[3,0,1] should be 0 ✓
        
        U[:, 0] = [1, 0, 1, 0]
        V[:, 0] = [1, 0]
        W[:, 0] = [1, 1]
        
        # Term 2: covers T[1,1,0] and T[3,1,1]
        U[:, 1] = [0, 1, 0, 1]
        V[:, 1] = [0, 1]
        W[:, 1] = [1, 1]
        
        # Check: U[1]*V[1]*W[0] = 1*1*1 = 1 -> T[1,1,0] ✓
        # U[3]*V[1]*W[1] = 1*1*1 = 1 -> T[3,1,1] ✓
        
        return U, V, W, rank
    
    def build_2x2x2x3():
        """Returns U, V, W for 2x2 x 2x3 with rank 9."""
        rank = 9
        U = np.zeros((4, rank), dtype=np.int32)
        V = np.zeros((6, rank), dtype=np.int32)
        W = np.zeros((6, rank), dtype=np.int32)
        
        # B is 2x3, split into B1 (2x1, cols 0) and B2 (2x2, cols 1-2)
        # B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
        # C is 2x3, indices: 0=C00, 1=C10, 2=C01, 3=C11, 4=C02, 5=C12
        
        # Part 1: A * B1 (2x2 x 2x1), rank 2
        U1, V1, W1, r1 = build_2x2x2x1()
        # V1 has 2 entries (B00, B10), map to V[0:2]
        # W1 has 2 entries (C00, C10), map to W[0:2]
        U[:, 0:r1] = U1
        V[0:2, 0:r1] = V1
        W[0:2, 0:r1] = W1
        
        # Part 2: A * B2 (2x2 x 2x2), rank 7 (Strassen)
        U2, V2, W2, r2 = build_strassen_2x2x2()
        # V2 has 4 entries (B00,B01,B10,B11 for the 2x2 block)
        # These correspond to global B indices 1,2,3,4 (B01,B02,B11,B12)
        # Wait, B2 is cols 1-2 of B, so B2 = [[B01, B02], [B11, B12]]
        # Local indices: 0=B01, 1=B02, 2=B11, 3=B12
        # Global indices: 1, 2, 3, 4
        U[:, r1:r1+r2] = U2
        V[1:5, r1:r1+r2] = V2  # Map local 0-3 to global 1-4
        W[2:6, r1:r1+r2] = W2  # C2 is cols 1-2, so C01,C11,C02,C12 -> global 2,3,4,5
        
        return U, V, W, rank
    
    # Now assemble 2x4 x 4x3
    def build_2x4x4x3():
        """Returns U, V, W for 2x4 x 4x3 with rank 18."""
        rank = 18
        U = np.zeros((8, rank), dtype=np.int32)
        V = np.zeros((12, rank), dtype=np.int32)
        W = np.zeros((6, rank), dtype=np.int32)
        
        # A is 2x4: [A1 | A2] where A1, A2 are 2x2
        # A1 indices: 0,1,2,3 (A00,A01,A10,A11)
        # A2 indices: 4,5,6,7 (A02,A03,A12,A13)
        # B is 4x3: [B_top; B_bottom] where each is 2x3
        # B_top indices: 0-5 (B00,B01,B02,B10,B11,B12)
        # B_bottom indices: 6-11 (B20,B21,B22,B30,B31,B32)
        # C is 2x3: indices 0-5
        
        # Part 1: A1 * B_top, rank 9
        U1, V1, W1, r1 = build_2x2x2x3()
        U[0:4, 0:r1] = U1
        V[0:6, 0:r1] = V1
        W[0:6, 0:r1] = W1
        
        # Part 2: A2 * B_bottom, rank 9
        U2, V2, W2, r2 = build_2x2x2x3()
        U[4:8, r1:r1+r2] = U2
        V[6:12, r1:r1+r2] = V2
        W[0:6, r1:r1+r2] = W2
        
        return U, V, W, rank
    
    # Assemble 2x4 x 4x6
    def build_2x4x4x6():
        """Returns U, V, W for 2x4 x 4x6 with rank 36."""
        rank = 36
        U = np.zeros((8, rank), dtype=np.int32)
        V = np.zeros((24, rank), dtype=np.int32)
        W = np.zeros((12, rank), dtype=np.int32)
        
        # Split B (4x6) into B_left (4x3) and B_right (4x3)
        # B_left indices: 0-11
        # B_right indices: 12-23
        # C is 2x6: C_left (2x3) indices 0-5, C_right (2x3) indices 6-11
        
        # Part 1: A * B_left, rank 18
        U1, V1, W1, r1 = build_2x4x4x3()
        U[:, 0:r1] = U1
        V[0:12, 0:r1] = V1
        W[0:6, 0:r1] = W1
        
        # Part 2: A * B_right, rank 18
        U2, V2, W2, r2 = build_2x4x4x3()
        U[:, r1:r1+r2] = U2
        V[12:24, r1:r1+r2] = V2
        W[6:12, r1:r1+r2] = W2
        
        return U, V, W, rank
    
    U_full, V_full, W_full, rank_full = build_2x4x4x6()
    
    # Verify
    T_rec = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for k in range(rank_full):
        T_rec += np.einsum('a,b,c->abc', U_full[:, k], V_full[:, k], W_full[:, k])
    
    if not np.array_equal(T_rec, T):
        # Fallback to schoolbook
        best_rank = n * m * p
        U_school = np.zeros((U_dim, best_rank), dtype=np.int32)
        V_school = np.zeros((V_dim, best_rank), dtype=np.int32)
        W_school = np.zeros((W_dim, best_rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
