"""Matrix multiplication 4x4x6 via 4x4x4 Strassen + 4x4x2 Strassen-block."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _strassen_2x2():
    """2x2x2 Strassen factors. U: (4,7), V: (4,7), W: (4,7)."""
    n = m = p = 2
    r = 7
    U = [[0.0]*r for _ in range(4)]
    V = [[0.0]*r for _ in range(4)]
    W = [[0.0]*r for _ in range(4)]
    prods = [
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
    ]
    for c,(ub,vb,wb) in enumerate(prods):
        for (i,j,cv) in ub: U[i*m+j][c] += cv
        for (j,k,cv) in vb: V[j*p+k][c] += cv
        for (i,k,cv) in wb: W[k*n+i][c] += cv
    return U, V, W

def _build_444():
    """4x4x4 Strassen via Kronecker of 2x2 Strassen. Returns U(16,49), V(16,49), W(16,49)."""
    Us, Vs, Ws = _strassen_2x2()
    r = 49
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1+i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1+j2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            U[i*4+j][col] = Us[i1*2+j1][r1]*Us[i2*2+j2][r2]
    for j1 in range(2):
        for j2 in range(2):
            j = 2*j1+j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1+k2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            V[j*4+k][col] = Vs[j1*2+k1][r1]*Vs[j2*2+k2][r2]
    for k1 in range(2):
        for k2 in range(2):
            k = 2*k1+k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1+i2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            W[k*4+i][col] = Ws[k1*2+i1][r1]*Ws[k2*2+i2][r2]
    return U, V, W

def _verify(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, Ua.shape[1])
    Vr = Va.reshape(m, p, Va.shape[1])
    Wr = Wa.reshape(p, n, Wa.shape[1])
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = float(np.max(np.abs(T - 1.0)))
    return err

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linear dependencies among columns."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    
    # Build the tensor and check for reducibility
    # A column c is redundant if the tensor it contributes is in the span of others
    # This is hard to check directly, so try: for each column, set it to 0 and check if
    # we can adjust other columns to compensate (linear system)
    
    # Simpler: try removing one column at a time and checking if the remaining
    # columns can still span the tensor (via SVD of the tensor slices)
    
    # Build the 3-mode tensor
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    
    # The tensor is T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For each c, define T_c[i,j,k] = U[i,j,c]*V[j,k,c]*W[k,i,c]
    # We want to find if any T_c is a linear combination of others
    
    # This is expensive, so do a heuristic: try removing each column and
    # check if the residual can be absorbed by scaling others
    
    # Better heuristic: look for pairs of columns that are proportional
    # in all three factors
    keep = list(range(r))
    
    for attempt in range(r):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        for ci in range(len(keep)):
            if time.time() > deadline - 15.0:
                break
            for cj in range(ci+1, len(keep)):
                if time.time() > deadline - 15.0:
                    break
                ci_idx = keep[ci]
                cj_idx = keep[cj]
                
                # Check if column cj is proportional to column ci in all three matrices
                # U[:,ci] ~ alpha * U[:,cj], V[:,ci] ~ beta * V[:,cj], W[:,ci] ~ gamma * W[:,cj]
                # with alpha*beta*gamma = 1 for the product to match
                
                Uci = Ua[:, ci_idx]
                Ucj = Ua[:, cj_idx]
                Vci = Va[:, ci_idx]
                Vcj = Va[:, cj_idx]
                Wci = Wa[:, ci_idx]
                Wcj = Wa[:, cj_idx]
                
                # Find alpha such that Uci ~ alpha*Ucj
                # Use least squares
                if np.linalg.norm(Ucj) < 1e-10 or np.linalg.norm(Uci) < 1e-10:
                    continue
                alpha = np.dot(Uci, Ucj) / np.dot(Ucj, Ucj)
                if np.linalg.norm(Uci - alpha*Ucj) > 1e-8 * np.linalg.norm(Uci):
                    continue
                
                if np.linalg.norm(Vcj) < 1e-10 or np.linalg.norm(Vci) < 1e-10:
                    continue
                beta = np.dot(Vci, Vcj) / np.dot(Vcj, Vcj)
                if np.linalg.norm(Vci - beta*Vcj) > 1e-8 * np.linalg.norm(Vci):
                    continue
                
                if np.linalg.norm(Wcj) < 1e-10 or np.linalg.norm(Wci) < 1e-10:
                    continue
                gamma = np.dot(Wci, Wcj) / np.dot(Wcj, Wcj)
                if np.linalg.norm(Wci - gamma*Wcj) > 1e-8 * np.linalg.norm(Wci):
                    continue
                
                # Check if alpha*beta*gamma ≈ 1
                if abs(alpha*beta*gamma - 1.0) > 1e-6:
                    continue
                
                # Column cj is redundant! Remove it
                keep.remove(cj_idx)
                found = True
                break
            if found:
                break
        
        if not found:
            break
    
    if len(keep) < r:
        U_new = Ua[:, keep].tolist()
        V_new = Va[:, keep].tolist()
        W_new = Wa[:, keep].tolist()
        err = _verify(U_new, V_new, W_new, n, m, p)
        if err < 1e-6:
            return U_new, V_new, W_new
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Check deadline before starting heavy computation
    if time.time() > deadline - 5.0:
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

    Us, Vs, Ws = _strassen_2x2()
    U444, V444, W444 = _build_444()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442  # 77

    U = [[0.0]*total_rank for _ in range(n*m)]  # 16 x 77
    V = [[0.0]*total_rank for _ in range(m*p)]  # 24 x 77
    W = [[0.0]*total_rank for _ in range(p*n)]  # 24 x 77

    # Part 1: 4x4x4 Strassen for columns 0-3 of B -> columns 0-3 of C
    for idx in range(16):
        for c in range(rank_444):
            if U444[idx][c] != 0:
                U[idx][c] = U444[idx][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank_444):
                if V444[j*4+k][c] != 0:
                    V[j*6+k][c] = V444[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(rank_444):
                if W444[k*4+i][c] != 0:
                    W[k*4+i][c] = W444[k*4+i][c]

    # Part 2: 4x4x2 Strassen-block for columns 4-5 of B -> columns 4-5 of C
    products = [
        (0, 0),  # A[0:2,0:2] * B[0:2,4:6]
        (0, 1),  # A[0:2,2:4] * B[2:4,4:6]
        (1, 0),  # A[2:4,0:2] * B[0:2,4:6]
        (1, 1),  # A[2:4,2:4] * B[2:4,4:6]
    ]
    
    for p_idx, (I, J) in enumerate(products):
        col_off = rank_444 + p_idx*7
        bot = J  # J=0 means B top rows, J=1 means B bottom rows
        
        for i0 in range(2):
            for j0 in range(2):
                i = 2*I + i0
                j = 2*J + j0
                for s in range(7):
                    v = Us[2*i0+j0][s]
                    if v: U[i*4+j][col_off+s] += v
        for j0 in range(2):
            for k0 in range(2):
                jg = 2*bot + j0
                kg = 4 + k0
                for s in range(7):
                    v = Vs[2*j0+k0][s]
                    if v: V[jg*6+kg][col_off+s] += v
        for i0 in range(2):
            for k0 in range(2):
                ig = 2*I + i0
                kg = 4 + k0
                for s in range(7):
                    v = Ws[2*k0+i0][s]
                    if v: W[kg*4+ig][col_off+s] += v

    # Deadline check before rank reduction
    if time.time() > deadline - 20.0:
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by finding proportional columns
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)

    # Verification
    err = _verify(U, V, W, n, m, p)
    
    if err > 1e-6:
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

    return {"U": U, "V": V, "W": W}
