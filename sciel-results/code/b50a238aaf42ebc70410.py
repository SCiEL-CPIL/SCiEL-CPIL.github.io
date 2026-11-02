import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

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

    def get_rank14_decomp():
        """Return the known rank-14 Strassen-based decomposition."""
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
        
        return U, V, W

    def try_remove_term(U, V, W, R):
        """
        Try to remove each term r by expressing it as a linear combination
        of the remaining R-1 terms. Returns (U_new, V_new, W_new) if successful.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for remove_idx in range(R):
            if time.time() > deadline - 5:
                return None
            
            # Build the tensor for the term to be removed
            # The rank-1 tensor T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
            # We need to check if T_remove can be written as sum_{r' != remove} c_r' * T_r'
            
            # For each (i,j,k) in the output, we have:
            # U[i,remove]*V[j,remove]*W[k,remove] = sum_{r' != remove} c_r' * U[i,r']*V[j,r']*W[k,r']
            
            # This is a linear system. Let's set it up.
            # Unknowns: c_0, c_1, ..., c_{R-1} excluding c_remove
            # Equations: one for each (i,j,k) where the LHS is nonzero
            
            # Actually, let's think differently. The tensor T is a 4x8x8 array.
            # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # If we remove term r0, we need:
            # T[i,j,k] = U[i,r0]*V[j,r0]*W[k,r0] + sum_{r != r0} c_r * U[i,r]*V[j,r]*W[k,r]
            # So: sum_{r != r0} c_r * U[i,r]*V[j,r]*W[k,r] = T[i,j,k] - U[i,r0]*V[j,r0]*W[k,r0]
            
            # Let's compute the full tensor T
            T = np.zeros((nm, mp, pn))
            for r in range(R):
                for i in range(nm):
                    for j in range(mp):
                        for k in range(pn):
                            T[i,j,k] += U_np[i,r] * V_np[j,r] * W_np[k,r]
            
            # Now for each removal candidate, solve the linear system
            # Variables: c_r for r != remove_idx
            # The system is: for all (i,j,k), sum_{r != remove_idx} c_r * U[i,r]*V[j,r]*W[k,r] = T[i,j,k] - U[i,remove_idx]*V[j,remove_idx]*W[k,remove_idx]
            
            # This is a linear system A * c = b where A has dimensions (nm*mp*pn) x (R-1)
            # But many entries of A will be zero, and the system is overdetermined.
            
            # Let's use least squares and check if the residual is small
            A = np.zeros((nm*mp*pn, R-1))
            b = np.zeros(nm*mp*pn)
            
            idx = 0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        col = 0
                        for r in range(R):
                            if r == remove_idx:
                                continue
                            A[idx, col] = U_np[i,r] * V_np[j,r] * W_np[k,r]
                            col += 1
                        b[idx] = T[i,j,k] - U_np[i,remove_idx] * V_np[j,remove_idx] * W_np[k,remove_idx]
                        idx += 1
            
            # Solve least squares
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except:
                continue
            
            # Check if the solution is good
            if len(residuals) > 0 and residuals[0] < 1e-8:
                # Found a valid removal! Construct the new decomposition
                U_new = [[0]*(R-1) for _ in range(nm)]
                V_new = [[0]*(R-1) for _ in range(mp)]
                W_new = [[0]*(R-1) for _ in range(pn)]
                
                col = 0
                for r in range(R):
                    if r == remove_idx:
                        continue
                    for i in range(nm):
                        U_new[i][col] = U[i][r]
                    for j in range(mp):
                        V_new[j][col] = V[j][r]
                    for k in range(pn):
                        W_new[k][col] = W[k][r]
                    col += 1
                
                # Wait, this doesn't account for the linear combination coefficients!
                # The new decomposition should have:
                # U_new[i,col] = U[i,r] * c_r (for the scaled version)
                # Actually, we can absorb the coefficients into one of the matrices
                # Let's scale U by c_r
                col = 0
                for r in range(R):
                    if r == remove_idx:
                        continue
                    for i in range(nm):
                        U_new[i][col] = U[i][r] * c[col]
                    for j in range(mp):
                        V_new[j][col] = V[j][r]
                    for k in range(pn):
                        W_new[k][col] = W[k][r]
                    col += 1
                
                # Verify
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new
                else:
                    # Try scaling V instead
                    col = 0
                    for r in range(R):
                        if r == remove_idx:
                            continue
                        for i in range(nm):
                            U_new[i][col] = U[i][r]
                        for j in range(mp):
                            V_new[j][col] = V[j][r] * c[col]
                        for k in range(pn):
                            W_new[k][col] = W[k][r]
                        col += 1
                    if verify(U_new, V_new, W_new):
                        return U_new, V_new, W_new
                    else:
                        # Try scaling W
                        col = 0
                        for r in range(R):
                            if r == remove_idx:
                                continue
                            for i in range(nm):
                                U_new[i][col] = U[i][r]
                            for j in range(mp):
                                V_new[j][col] = V[j][r]
                            for k in range(pn):
                                W_new[k][col] = W[k][r] * c[col]
                            col += 1
                        if verify(U_new, V_new, W_new):
                            return U_new, V_new, W_new
        
        return None

    # Get the rank-14 decomposition
    U14, V14, W14 = get_rank14_decomp()
    
    # Verify it's correct
    if not verify(U14, V14, W14):
        return naive_decomp()
    
    # Try to reduce to rank 13
    result = try_remove_term(U14, V14, W14, 14)
    if result is not None:
        U13, V13, W13 = result
        if verify(U13, V13, W13):
            return {"U": U13, "V": V13, "W": W13}
    
    # Fall back to rank 14
    return {"U": U14, "V": V14, "W": W14}
