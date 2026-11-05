import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify_exact(U, V, W):
        """Verify exact decomposition using Fraction arithmetic."""
        n, m, p = 2, 4, 5
        rank = len(U[0])
        
        U_f = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        V_f = [[Fraction(V[j][r]) for r in range(rank)] for j in range(m*p)]
        W_f = [[Fraction(W[k][r]) for r in range(rank)] for k in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_f[i*m+j][r] * V_f[j*p+k][r] * W_f[k*n+i][r]
                    if s != Fraction(1):
                        return False
        return True
    
    def verify_float(U, V, W, tol=1e-9):
        """Fast float verification."""
        n, m, p = 2, 4, 5
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def strassen_2x2x2():
        """Strassen's 7-rank decomposition for 2x2 * 2x2."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: A00 * (B11 - B10)
        U[0, 0] = 1; V[3, 0] = 1; V[2, 0] = -1; W[0, 0] = 1; W[3, 0] = 1
        # M2: (A00 + A01) * B11
        U[0, 1] = 1; U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1; W[1, 1] = 1
        # M3: (A10 + A11) * B00
        U[2, 2] = 1; U[3, 2] = 1; V[0, 2] = 1; W[1, 2] = 1; W[2, 2] = 1
        # M4: A11 * (B01 - B00)
        U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[3, 3] = -1
        # M5: (A00 + A10) * (B00 + B01)
        U[0, 4] = 1; U[2, 4] = 1; V[0, 4] = 1; V[1, 4] = 1; W[0, 4] = -1; W[3, 4] = 1
        # M6: (A01 - A10) * (B10 + B11)
        U[1, 5] = 1; U[2, 5] = -1; V[2, 5] = 1; V[3, 5] = 1; W[2, 5] = 1; W[3, 5] = 1
        # M7: (A00 - A11) * (B00 + B11)
        U[0, 6] = 1; U[3, 6] = -1; V[0, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W
    
    def schoolbook_2x2x3():
        """Schoolbook rank-12 decomposition for 2x2 * 2x3."""
        U = np.zeros((4, 12))
        V = np.zeros((6, 12))
        W = np.zeros((6, 12))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    def search_2x2x3(target_rank, max_trials=100, max_steps=200):
        """Search for a low-rank decomposition of M_{2,2,3}."""
        np.random.seed(42)
        nn, mm, pp = 2, 2, 3
        
        best_U, best_V, best_W, best_err = None, None, None, float('inf')
        
        for trial in range(max_trials):
            if time.time() > _deadline - 30:
                break
            
            U = np.random.randn(nn*mm, target_rank) * 0.5
            V = np.random.randn(mm*pp, target_rank) * 0.5
            W = np.random.randn(pp*nn, target_rank) * 0.5
            
            for step in range(max_steps):
                if time.time() > _deadline - 30:
                    break
                
                U_3d = U.reshape(nn, mm, target_rank)
                V_3d = V.reshape(mm, pp, target_rank)
                W_3d = W.reshape(pp, nn, target_rank)
                
                err = 0.0
                for i in range(nn):
                    for j in range(mm):
                        for k in range(pp):
                            s = np.sum(U_3d[i,j,:] * V_3d[j,k,:] * W_3d[k,i,:])
                            err += (s - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-12:
                    return best_U, best_V, best_W
                
                # Gradient-like update
                lr = 0.02
                U += lr * np.random.randn(nn*mm, target_rank)
                V += lr * np.random.randn(mm*pp, target_rank)
                W += lr * np.random.randn(pp*nn, target_rank)
        
        if best_err < 1e-9:
            return best_U, best_V, best_W
        return None, None, None
    
    # Get Strassen decomposition
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Try to find a low-rank decomposition for M_{2,2,3}
    U_223 = V_223 = W_223 = None
    rank_223 = 12  # default to schoolbook
    
    for target_rank in [10, 11]:
        if time.time() > _deadline - 40:
            break
        result = search_2x2x3(target_rank, max_trials=50, max_steps=100)
        if result[0] is not None:
            U_223, V_223, W_223 = result
            rank_223 = target_rank
            break
    
    if U_223 is None:
        U_223, V_223, W_223 = schoolbook_2x2x3()
        rank_223 = 12
    
    # Construct full M_{2,4,5} decomposition
    total_rank = 2 * (7 + rank_223)
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    for block in range(2):
        a_col_offset = block * 2
        
        # Strassen part
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j, r]
                    if val != 0:
                        U_full[i*m + (a_col_offset + j), col_offset + r] = val
            for j in range(2):
                for k in range(2):
                    val = V_s[j*2+k, r]
                    if val != 0:
                        V_full[(block*2 + j)*p + k, col_offset + r] = val
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i, r]
                    if val != 0:
                        W_full[k*n + i, col_offset + r] = val
        col_offset += 7
        
        # 2x2x3 part
        for r in range(rank_223):
            for i in range(2):
                for j in range(2):
                    val = U_223[i*2+j, r]
                    if val != 0:
                        U_full[i*m + (a_col_offset + j), col_offset + r] = val
            for j in range(2):
                for k in range(3):
                    val = V_223[j*3+k, r]
                    if val != 0:
                        V_full[(block*2 + j)*p + (2 + k), col_offset + r] = val
            for k in range(3):
                for i in range(2):
                    val = W_223[k*2+i, r]
                    if val != 0:
                        W_full[(2 + k)*n + i, col_offset + r] = val
        col_offset += rank_223
    
    # Verify
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if verify_float(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
