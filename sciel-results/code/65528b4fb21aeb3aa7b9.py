"""Rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                u_row = U_np[i*m+j]
                for k in range(p):
                    s = np.dot(u_row * V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank14():
        """Construct the rank-14 Strassen decomposition for 2x2 * 2x4."""
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Block 1: A (2x2) * B_left (2x2, cols 0,1)
        # B indices: B00=0, B01=1, B10=4, B11=5
        # C indices: C00=0, C10=1, C01=2, C11=3
        
        # r=0: (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: A00*(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: A11*(B10-B00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: (A00+A01)*B11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: (A10-A00)*(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: (A01-A11)*(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Block 2: A (2x2) * B_right (2x2, cols 2,3)
        # B indices: B02=2, B03=3, B12=6, B13=7
        # C indices: C02=4, C12=5, C03=6, C13=7
        
        # r=7: (A00+A11)*(B02+B13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: (A10+A11)*B02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: A00*(B03-B13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: A11*(B12-B02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: (A00+A01)*B13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: (A10-A00)*(B02+B03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: (A01-A11)*(B12+B13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        return {"U": U, "V": V, "W": W}

    def try_rank13_via_merge(r14):
        """
        Attempt to reduce rank from 14 to 13 by merging two Strassen terms.
        Two terms can be merged if their U-vectors are identical (up to scale),
        allowing the B and C factors to be combined linearly.
        """
        if time.time() > deadline - 5:
            return None
            
        U = [row[:] for row in r14['U']]
        V = [row[:] for row in r14['V']]
        W = [row[:] for row in r14['W']]
        R = 14
        
        # Identify pairs with identical U-vectors
        # U vectors (as tuples):
        # r=0:  (1,0,0,1)  -- a00+a11
        # r=1:  (0,0,1,1)  -- a10+a11
        # r=2:  (1,0,0,0)  -- a00
        # r=3:  (0,0,0,1)  -- a11
        # r=4:  (1,1,0,0)  -- a00+a01
        # r=5:  (0,0,1,-1) -- a10-a00
        # r=6:  (0,1,0,-1) -- a01-a11
        # r=7:  (1,0,0,1)  -- a00+a11  (same as r=0)
        # r=8:  (0,0,1,1)  -- a10+a11  (same as r=1)
        # r=9:  (1,0,0,0)  -- a00       (same as r=2)
        # r=10: (0,0,0,1)  -- a11       (same as r=3)
        # r=11: (1,1,0,0)  -- a00+a01   (same as r=4)
        # r=12: (0,0,1,-1) -- a10-a00   (same as r=5)
        # r=13: (0,1,0,-1) -- a01-a11   (same as r=6)
        
        # Pairs to try merging: (0,7), (1,8), (2,9), (3,10), (4,11), (5,12), (6,13)
        merge_pairs = [(0,7), (1,8), (2,9), (3,10), (4,11), (5,12), (6,13)]
        
        for a, b in merge_pairs:
            if time.time() > deadline - 3:
                break
            
            # Try: merge term b into term a by adding their V and W vectors
            # New term a: U_a (unchanged), V_a + V_b, W_a + W_b
            # Remove term b, keep terms 0..a-1, a (modified), a+1..12
            
            U_test = [row[:] for row in U if row is not None and row.index(0) != -1]  # placeholder
            # Build test matrices by removing term b
            U_test = []
            V_test = []
            W_test = []
            for i in range(R):
                if i == b:
                    continue
                U_test.append(U[i][:])
                V_test.append(V[i][:])
                W_test.append(W[i][:])
            
            # Modify term a to absorb term b
            for idx in range(8):
                V_test[a][idx] += V[b][idx]
                W_test[a][idx] += W[b][idx]
            
            if verify(U_test, V_test, W_test):
                return {"U": U_test, "V": V_test, "W": W_test}
            
            # Try with sign flip on V
            U_test = []
            V_test = []
            W_test = []
            for i in range(R):
                if i == b:
                    continue
                U_test.append(U[i][:])
                V_test.append(V[i][:])
                W_test.append(W[i][:])
            for idx in range(8):
                V_test[a][idx] += V[b][idx]
                W_test[a][idx] -= W[b][idx]
            if verify(U_test, V_test, W_test):
                return {"U": U_test, "V": V_test, "W": W_test}
            
            # Try with sign flip on W
            U_test = []
            V_test = []
            W_test = []
            for i in range(R):
                if i == b:
                    continue
                U_test.append(U[i][:])
                V_test.append(V[i][:])
                W_test.append(W[i][:])
            for idx in range(8):
                V_test[a][idx] -= V[b][idx]
                W_test[a][idx] += W[b][idx]
            if verify(U_test, V_test, W_test):
                return {"U": U_test, "V": W_test, "W": W_test}
        
        return None

    def naive_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-14 decomposition
    r14 = build_rank14()
    if verify(r14['U'], r14['V'], r14['W']):
        # Attempt rank-13
        r13 = try_rank13_via_merge(r14)
        if r13 is not None:
            return r13
        return r14
    
    # Fallback
    return naive_decomp()
# EVOLVE-BLOCK-END
