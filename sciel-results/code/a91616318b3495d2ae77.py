import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 2, 8
    
    # Build the multiplication tensor T[a,b,c]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition using ALS
    def try_rank(rank, seed=42):
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for iteration in range(80):
            if time.time() > deadline - 15:
                return None
            
            # Update U
            for a in range(n*m):
                M = np.zeros((m*p * p*n, rank))
                for b in range(m*p):
                    for c in range(p*n):
                        M[b*(p*n) + c, :] = V[b, :] * W[c, :]
                rhs = T[a, :, :].ravel()
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    return None
            
            # Update V
            for b in range(m*p):
                M2 = np.zeros((n*m * p*n, rank))
                for a in range(n*m):
                    for c in range(p*n):
                        M2[a*(p*n) + c, :] = U[a, :] * W[c, :]
                rhs = T[:, b, :].ravel()
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M2, rhs, rcond=None)
                except:
                    return None
            
            # Update W
            for c in range(p*n):
                M3 = np.zeros((n*m * m*p, rank))
                for a in range(n*m):
                    for b in range(m*p):
                        M3[a*(m*p) + b, :] = U[a, :] * V[b, :]
                rhs = T[:, :, c].ravel()
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M3, rhs, rcond=None)
                except:
                    return None
            
            # Check error
            error = 0.0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += (val - T[a, b, c]) ** 2
            if error < 1e-12:
                return (U, V, W)
        
        return None
    
    # Try ranks from 24 upward
    for rank in [24, 25, 26, 27]:
        if time.time() > deadline - 25:
            break
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 20:
                break
            result = try_rank(rank, seed)
            if result is not None:
                U, V, W = result
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: rank 28 with correct Strassen coefficients
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        # m1 = (a00+a11)(b00+b11) -> c01 += 1, c11 += 1
        U[0][col] = 1; U[3][col] = 1
        V[0*p+2*b][col] = 1; V[1*p+(2*b+1)][col] = 1
        W[2*b*n+1][col] = 1; W[(2*b+1)*n+1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00 -> c10 += 1, c11 -= 1
        U[2][col] = 1; U[3][col] = 1
        V[0*p+2*b][col] = 1
        W[(2*b+1)*n+0][col] = 1; W[(2*b+1)*n+1][col] = -1
        col += 1
        
        # m3 = a00(b10-b00) -> c00 -= 1, c01 += 1
        U[0][col] = 1
        V[1*p+2*b][col] = 1; V[0*p+2*b][col] = -1
        W[2*b*n+0][col] = -1; W[2*b*n+1][col] = 1
        col += 1
        
        # m4 = a11(b01-b11) -> c00 += 1, c11 += 1
        U[3][col] = 1
        V[0*p+(2*b+1)][col] = 1; V[1*p+(2*b+1)][col] = -1
        W[2*b*n+0][col] = 1; W[(2*b+1)*n+1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11 -> c00 += 1, c11 += 1
        U[0][col] = 1; U[1][col] = 1
        V[1*p+(2*b+1)][col] = 1
        W[2*b*n+0][col] = 1; W[(2*b+1)*n+1][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b10) -> c10 += 1
        U[2][col] = 1; U[0][col] = -1
        V[0*p+2*b][col] = 1; V[1*p+2*b][col] = 1
        W[(2*b+1)*n+0][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b01+b11) -> c00 += 1
        U[1][col] = 1; U[3][col] = -1
        V[0*p+(2*b+1)][col] = 1; V[1*p+(2*b+1)][col] = 1
        W[2*b*n+0][col] = 1
        col += 1
    
    # Verify rank-28
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    valid = True
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                actual = np.dot(U_np[a, :], V_np[b, :] * W_np[c, :])
                if abs(actual - T[a, b, c]) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook rank 32
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    c2 = 0
    for i2 in range(n):
        for j2 in range(m):
            for k2 in range(p):
                U_fb[i2 * m + j2][c2] = 1
                V_fb[j2 * p + k2][c2] = 1
                W_fb[k2 * n + i2][c2] = 1
                c2 += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
