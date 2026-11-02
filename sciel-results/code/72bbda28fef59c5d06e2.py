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
    
    def strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
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
        
        # Right block (cols 2,3 of B) - standard Strassen
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
        return None
    
    def try_rank13_nls():
        """
        Nonlinear least squares search for rank-13 decomposition.
        Start from rank-14 Strassen and try to merge/perturb.
        """
        if time.time() > deadline - 10:
            return None
        
        R = 13
        rng = np.random.RandomState(123)
        
        # Get the rank-14 solution as starting point
        sol14 = strassen_rank14()
        if sol14 is None:
            return None
        
        U14 = np.array(sol14['U'], dtype=float)  # (4, 14)
        V14 = np.array(sol14['V'], dtype=float)  # (8, 14)
        W14 = np.array(sol14['W'], dtype=float)  # (8, 14)
        
        # Try to find rank-13 by perturbing and optimizing
        best_result = None
        best_err = float('inf')
        
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            # Strategy: take rank-14, zero out one term, and optimize
            # Or: start with random rank-13 and optimize
            
            if trial < 15:
                # Strategy 1: Start from rank-14, drop one term, optimize
                drop_idx = trial % 14
                U = U14[:, [r for r in range(14) if r != drop_idx]].copy()
                V = V14[:, [r for r in range(14) if r != drop_idx]].copy()
                W = W14[:, [r for r in range(14) if r != drop_idx]].copy()
            else:
                # Strategy 2: Random initialization with structure
                U = np.zeros((4, R))
                V = np.zeros((8, R))
                W = np.zeros((8, R))
                
                # Initialize with Strassen-like patterns
                # Use the 7 left Strassen terms for first 7
                for r in range(7):
                    U[:, r] = rng.normal(0, 1, 4)
                    V[:2, r] = rng.normal(0, 1, 2)
                    W[:4, r] = rng.normal(0, 1, 4)
                
                # Use the 6 right Strassen terms for next 6
                for r in range(7, 13):
                    U[:, r] = rng.normal(0, 1, 4)
                    V[2:, r] = rng.normal(0, 1, 2)
                    W[4:, r] = rng.normal(0, 1, 4)
            
            # Optimize using alternating linear solves
            for als_iter in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Solve for V (linear given U, W)
                A_V = np.zeros((16, 8 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = j * p * R + k * R + r
                                A_V[eq_idx, col_idx] = U[i * m + j, r] * W[k * n + i, r]
                
                rhs = np.ones(16)
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                    V_new = V_flat.reshape(8, R).T
                except:
                    continue
                
                # Solve for W (linear given U, V)
                A_W = np.zeros((16, 8 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = k * n * R + i * R + r
                                A_W[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
                
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                    W_new = W_flat.reshape(8, R).T
                except:
                    continue
                
                # Solve for U (linear given V, W)
                A_U = np.zeros((16, 4 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = i * m * R + j * R + r
                                A_U[eq_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
                
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                    U_new = U_flat.reshape(4, R).T
                except:
                    continue
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                            err += (s - 1.0) ** 2
                err = np.sqrt(err / 16.0)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        # Verify exactness
                        if verify(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                
                U, V, W = U_new, V_new, W_new
        
        return None
    
    def try_rank13_structured():
        """
        Try structured rank-13 by sharing one term between blocks.
        """
        if time.time() > deadline - 15:
            return None
        
        R = 13
        rng = np.random.RandomState(456)
        
        # The idea: have 6 terms for left block, 6 for right block, and 1 shared
        # Or: 7 left, 6 right with one shared, etc.
        
        best_result = None
        best_err = float('inf')
        
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((4, R))
            V = np.zeros((8, R))
            W = np.zeros((8, R))
            
            # Pattern: first 6 terms affect left block (k=0,1)
            # Next 6 terms affect right block (k=2,3)
            # Last term affects both
            
            for r in range(6):
                U[:, r] = rng.choice([-1, 0, 1], 4)
                V[:2, r] = rng.choice([-1, 0, 1], 2)
                W[:4, r] = rng.choice([-1, 0, 1], 4)
            
            for r in range(6, 12):
                U[:, r] = rng.choice([-1, 0, 1], 4)
                V[2:, r] = rng.choice([-1, 0, 1], 2)
                W[4:, r] = rng.choice([-1, 0, 1], 4)
            
            # Shared term
            U[:, 12] = rng.choice([-1, 0, 1], 4)
            V[:2, 12] = rng.choice([-1, 0, 1], 2)
            V[2:, 12] = rng.choice([-1, 0, 1], 2)
            W[:4, 12] = rng.choice([-1, 0, 1], 4)
            W[4:, 12] = rng.choice([-1, 0, 1], 4)
            
            # Optimize
            for als_iter in range(30):
                if time.time() > deadline - 3:
                    break
                
                # Solve for V
                A_V = np.zeros((16, 8 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = j * p * R + k * R + r
                                A_V[eq_idx, col_idx] = U[i * m + j, r] * W[k * n + i, r]
                
                rhs = np.ones(16)
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                    V = V_flat.reshape(8, R).T
                except:
                    continue
                
                # Solve for W
                A_W = np.zeros((16, 8 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = k * n * R + i * R + r
                                A_W[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
                
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                    W = W_flat.reshape(8, R).T
                except:
                    continue
                
                # Solve for U
                A_U = np.zeros((16, 4 * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(R):
                                col_idx = i * m * R + j * R + r
                                A_U[eq_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
                
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                    U = U_flat.reshape(4, R).T
                except:
                    continue
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (s - 1.0) ** 2
                err = np.sqrt(err / 16.0)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        if verify(U.tolist(), V.tolist(), W.tolist()):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try rank-13 searches
    result = try_rank13_nls()
    if result is not None:
        return result
    
    result = try_rank13_structured()
    if result is not None:
        return result
    
    # Fallback to rank-14 Strassen
    result = strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
