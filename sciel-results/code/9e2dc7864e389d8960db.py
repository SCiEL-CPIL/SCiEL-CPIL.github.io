"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, construct the rank-28 solution (known valid)
    def build_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        
        # Part 1: j=2, k=0..4 (schoolbook: 2*5=10 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
        # Part 2: j=0,1, k=4 (schoolbook: 2*2=4 terms)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
                
        # Part 3: j=0,1, k=0,1 (Strassen 2x2: 7 terms)
        # M1
        U[0][r] = 1; U[4][r] = 1  # i=0,j=0 and i=1,j=1
        V[0][r] = 1; V[6][r] = 1  # j=0,k=0 and j=1,k=1
        W[0][r] = 1; W[3][r] = 1  # k=0,i=0 and k=1,i=1
        r += 1
        # M2
        U[3][r] = 1; U[4][r] = 1  # i=1,j=0 and i=1,j=1
        V[0][r] = 1
        W[2][r] = 1; W[3][r] = -1  # k=1,i=0 and k=1,i=1
        r += 1
        # M3
        U[0][r] = 1
        V[1][r] = 1; V[6][r] = -1  # j=0,k=1 and j=1,k=1
        W[1][r] = 1; W[3][r] = 1  # k=0,i=1 and k=1,i=1
        r += 1
        # M4
        U[4][r] = 1
        V[5][r] = 1; V[0][r] = -1  # j=1,k=0 and j=0,k=0
        W[0][r] = 1; W[2][r] = 1  # k=0,i=0 and k=1,i=0
        r += 1
        # M5
        U[0][r] = 1; U[1][r] = 1  # i=0,j=0 and i=0,j=1
        V[6][r] = 1
        W[0][r] = -1; W[1][r] = 1  # k=0,i=0 and k=0,i=1
        r += 1
        # M6
        U[3][r] = 1; U[0][r] = -1  # i=1,j=0 and i=0,j=0
        V[0][r] = 1; V[1][r] = 1  # j=0,k=0 and j=0,k=1
        W[3][r] = 1  # k=1,i=1
        r += 1
        # M7
        U[1][r] = 1; U[4][r] = -1  # i=0,j=1 and i=1,j=1
        V[5][r] = 1; V[6][r] = 1  # j=1,k=0 and j=1,k=1
        W[0][r] = 1  # k=0,i=0
        r += 1
        
        # Part 4: j=0,1, k=2,3 (Strassen 2x2: 7 terms)
        # M1
        U[0][r] = 1; U[4][r] = 1
        V[2][r] = 1; V[8][r] = 1  # j=0,k=2 and j=1,k=3
        W[4][r] = 1; W[7][r] = 1  # k=2,i=0 and k=3,i=1
        r += 1
        # M2
        U[3][r] = 1; U[4][r] = 1
        V[2][r] = 1
        W[6][r] = 1; W[7][r] = -1  # k=3,i=0 and k=3,i=1
        r += 1
        # M3
        U[0][r] = 1
        V[3][r] = 1; V[8][r] = -1  # j=0,k=3 and j=1,k=3
        W[5][r] = 1; W[7][r] = 1  # k=2,i=1 and k=3,i=1
        r += 1
        # M4
        U[4][r] = 1
        V[7][r] = 1; V[2][r] = -1  # j=1,k=2 and j=0,k=2
        W[4][r] = 1; W[6][r] = 1  # k=2,i=0 and k=3,i=0
        r += 1
        # M5
        U[0][r] = 1; U[1][r] = 1
        V[8][r] = 1
        W[4][r] = -1; W[5][r] = 1  # k=2,i=0 and k=2,i=1
        r += 1
        # M6
        U[3][r] = 1; U[0][r] = -1
        V[2][r] = 1; V[3][r] = 1  # j=0,k=2 and j=0,k=3
        W[7][r] = 1  # k=3,i=1
        r += 1
        # M7
        U[1][r] = 1; U[4][r] = -1
        V[7][r] = 1; V[8][r] = 1  # j=1,k=2 and j=1,k=3
        W[4][r] = 1  # k=2,i=0
        r += 1
        
        return U, V, W, rank
    
    U28, V28, W28, rank28 = build_rank28()
    
    # Verify rank-28
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.sum(Ua[idx_u, :] * Va[idx_v, :] * Wa[idx_w, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U28, V28, W28, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U28, V28, W28, rank28
    
    # Try to find rank-24 solution using optimization
    target_rank = 24
    nm, mp, pn = n*m, m*p, p*n  # 6, 15, 10
    
    # Objective: sum of squares of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    def objective(x):
        if time.time() > deadline - 5:
            return 1e10
        Ux = x[:target_rank*nm].reshape(nm, target_rank)
        Vx = x[target_rank*nm:target_rank*(nm+mp)].reshape(mp, target_rank)
        Wx = x[target_rank*(nm+mp):].reshape(pn, target_rank)
        
        # Compute all 30 tensor entries
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(Ux[idx_u, :] * Vx[idx_v, :] * Wx[idx_w, :])
                    errors.append(val - 1.0)
        return np.sum(np.array(errors) ** 2)
    
    # Try multiple random initializations
    np.random.seed(42)
    best_obj = 1e10
    best_x = None
    
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        x0 = np.random.randn(target_rank * (nm + mp + pn)) * 0.5
        
        # Also try initializing from the rank-28 solution (padded with zeros)
        if attempt == 0:
            x0 = np.zeros(target_rank * (nm + mp + pn))
            U28a = np.array(U28, dtype=float)
            V28a = np.array(V28, dtype=float)
            W28a = np.array(W28, dtype=float)
            # Take first 24 columns
            x0[:target_rank*nm] = U28a[:, :target_rank].flatten()
            x0[target_rank*nm:target_rank*(nm+mp)] = V28a[:, :target_rank].flatten()
            x0[target_rank*(nm+mp):] = W28a[:, :target_rank].flatten()
        
        try:
            result = minimize(objective, x0, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-6})
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Check if we found a good solution
    if best_x is not None and best_obj < 1e-6:
        Ux = best_x[:target_rank*nm].reshape(nm, target_rank)
        Vx = best_x[target_rank*nm:target_rank*(nm+mp)].reshape(mp, target_rank)
        Wx = best_x[target_rank*(nm+mp):].reshape(pn, target_rank)
        
        # Round to half-integers
        U_rounded = np.round(Ux * 2) / 2
        V_rounded = np.round(Vx * 2) / 2
        W_rounded = np.round(Wx * 2) / 2
        
        # Verify the rounded solution
        def verify_rounded(Ur, Vr, Wr):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        s = np.sum(Ur[idx_u, :] * Vr[idx_v, :] * Wr[idx_w, :])
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        if verify_rounded(U_rounded, V_rounded, W_rounded):
            # Convert to lists of ints/half-ints
            U_list = [[int(v) if v == int(v) else float(v) for v in row] for row in U_rounded]
            V_list = [[int(v) if v == int(v) else float(v) for v in row] for row in V_rounded]
            W_list = [[int(v) if v == int(v) else float(v) for v in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 25 as well
    if time.time() < deadline - 20:
        for target_rank in [25]:
            nm, mp, pn = n*m, m*p, p*n
            best_obj = 1e10
            best_x = None
            
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                x0 = np.random.randn(target_rank * (nm + mp + pn)) * 0.5
                if attempt == 0:
                    x0 = np.zeros(target_rank * (nm + mp + pn))
                    U28a = np.array(U28, dtype=float)
                    V28a = np.array(V28, dtype=float)
                    W28a = np.array(W28, dtype=float)
                    x0[:target_rank*nm] = U28a[:, :target_rank].flatten()
                    x0[target_rank*nm:target_rank*(nm+mp)] = V28a[:, :target_rank].flatten()
                    x0[target_rank*(nm+mp):] = W28a[:, :target_rank].flatten()
                
                def objective_r(x, tr=target_rank):
                    if time.time() > deadline - 5:
                        return 1e10
                    Ux = x[:tr*nm].reshape(nm, tr)
                    Vx = x[tr*nm:tr*(nm+mp)].reshape(mp, tr)
                    Wx = x[tr*(nm+mp):].reshape(pn, tr)
                    errors = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                val = np.sum(Ux[idx_u, :] * Vx[idx_v, :] * Wx[idx_w, :])
                                errors.append(val - 1.0)
                    return np.sum(np.array(errors) ** 2)
                
                try:
                    result = minimize(objective_r, x0, method='BFGS', 
                                    options={'maxiter': 150, 'gtol': 1e-6})
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_x = result.x
                except Exception:
                    pass
            
            if best_x is not None and best_obj < 1e-6:
                Ux = best_x[:target_rank*nm].reshape(nm, target_rank)
                Vx = best_x[target_rank*nm:target_rank*(nm+mp)].reshape(mp, target_rank)
                Wx = best_x[target_rank*(nm+mp):].reshape(pn, target_rank)
                
                U_rounded = np.round(Ux * 2) / 2
                V_rounded = np.round(Vx * 2) / 2
                W_rounded = np.round(Wx * 2) / 2
                
                def verify_rounded2(Ur, Vr, Wr):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i * m + j
                                idx_v = j * p + k
                                idx_w = k * n + i
                                s = np.sum(Ur[idx_u, :] * Vr[idx_v, :] * Wr[idx_w, :])
                                if abs(s - 1.0) > 1e-9:
                                    return False
                    return True
                
                if verify_rounded2(U_rounded, V_rounded, W_rounded):
                    U_list = [[int(v) if v == int(v) else float(v) for v in row] for row in U_rounded]
                    V_list = [[int(v) if v == int(v) else float(v) for v in row] for row in V_rounded]
                    W_list = [[int(v) if v == int(v) else float(v) for v in row] for row in W_rounded]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return the best found (rank 28)
    U_list = [[int(v) if v == int(v) else float(v) for v in row] for row in U28]
    V_list = [[int(v) if v == int(v) else float(v) for v in row] for row in V28]
    W_list = [[int(v) if v == int(v) else float(v) for v in row] for row in W28]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
