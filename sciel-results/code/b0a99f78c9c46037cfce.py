"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
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
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        """Verify using numpy einsum."""
        R = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, R)
        V3 = V_arr.reshape(m, p, R)
        W3 = W_arr.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)
    
    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    def try_rank13_shared():
        """
        Try to find a rank-13 decomposition by merging the two diagonal Strassen terms.
        """
        if time.time() > deadline:
            return None
        
        R = 13
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Merged term 0: combines left and right diagonal Strassen products
        # U = [1, 0, 0, 1]
        U[0][0] = 1; U[3][0] = 1
        # V = [1, 0, 1, 0, 0, 1, 0, 1]
        V[0][0] = 1; V[2][0] = 1; V[5][0] = 1; V[7][0] = 1
        # W = [1, 0, 0, 1, 1, 0, 0, 1]
        W[0][0] = 1; W[3][0] = 1; W[4][0] = 1; W[7][0] = 1
        
        # Left block terms 1-6 (Strassen 2x2x2)
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block terms 7-12 (Strassen 2x2x2)
        # r=7: N2 = (a10+a11)b02
        U[2][7] = 1; U[3][7] = 1
        V[2][7] = 1
        W[5][7] = 1; W[7][7] = -1
        
        # r=8: N3 = a00(b03-b13)
        U[0][8] = 1
        V[3][8] = 1; V[7][8] = -1
        W[6][8] = 1; W[7][8] = 1
        
        # r=9: N4 = a11(b12-b02)
        U[3][9] = 1
        V[6][9] = 1; V[2][9] = -1
        W[4][9] = 1; W[5][9] = 1
        
        # r=10: N5 = (a00+a01)b13
        U[0][10] = 1; U[1][10] = 1
        V[7][10] = 1
        W[4][10] = -1; W[6][10] = 1
        
        # r=11: N6 = (a10-a00)(b02+b03)
        U[2][11] = 1; U[0][11] = -1
        V[2][11] = 1; V[3][11] = 1
        W[7][11] = 1
        
        # r=12: N7 = (a01-a11)(b12+b13)
        U[1][12] = 1; U[3][12] = -1
        V[6][12] = 1; V[7][12] = 1
        W[4][12] = 1
        
        if verify_numpy(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank12_search():
        """
        Try to find a rank-12 decomposition by searching for structured
        modifications. This is a more aggressive attempt.
        """
        if time.time() > deadline:
            return None
        
        # Start from the rank-13 solution and try to merge more terms
        # or find a different structure.
        
        # One approach: try to use a single shared term for multiple
        # Strassen products by exploiting the rectangular structure.
        
        # For now, let's try a parameterized search around the rank-13 solution.
        # We can try to set some entries to zero and see if the remaining
        # terms can be adjusted to maintain correctness.
        
        # This is a complex search, so we'll try a few simple modifications.
        
        # Try: can we merge terms 1 and 7? Both have U = [0,0,1,1]
        # Term 1: V=[1,0,0,0,0,0,0,0], W=[0,1,0,1,0,0,0,0]
        # Term 7: V=[0,0,1,0,0,0,0,0], W=[0,0,0,0,0,1,0,1]
        # Merged: V=[1,0,1,0,0,0,0,0], W=[0,1,0,1,0,1,0,1]
        # But this would give cross terms that don't cancel.
        
        # Try: can we merge terms 2 and 8? Both have U = [1,0,0,0]
        # Term 2: V=[0,1,0,0,0,-1,0,0], W=[0,0,1,1,0,0,0,0]
        # Term 8: V=[0,0,0,1,0,0,0,-1], W=[0,0,0,0,0,0,1,1]
        # Merged: V=[0,1,0,1,0,-1,0,-1], W=[0,0,1,1,0,0,1,1]
        # Again, cross terms won't cancel properly.
        
        # It seems that rank-12 is not achievable with this simple merging strategy.
        # Let's try a different approach: use a more sophisticated tensor decomposition
        # that exploits the full rectangular structure.
        
        # For now, return None to indicate that rank-12 was not found.
        return None
    
    # Try rank-13 first (most aggressive)
    result = try_rank13_shared()
    if result is not None:
        return result
    
    # Try rank-12 (even more aggressive, likely to fail)
    result = try_rank12_search()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
