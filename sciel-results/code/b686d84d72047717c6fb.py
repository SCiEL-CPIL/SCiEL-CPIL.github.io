"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time as _time

def _build_strassen_2x2x2(p, t, U, V, W, base, n=2):
    """Add Strassen rank-7 terms for 2x2x2 block t into arrays."""
    v_b00 = 0 * p + 2 * t
    v_b01 = 0 * p + 2 * t + 1
    v_b10 = 1 * p + 2 * t
    v_b11 = 1 * p + 2 * t + 1
    
    w_c00 = (2 * t) * n + 0
    w_c01 = (2 * t + 1) * n + 0
    w_c10 = (2 * t) * n + 1
    w_c11 = (2 * t + 1) * n + 1
    
    for s in range(7):
        c = base + s
        if s == 0:
            U[0][c] += 1; U[3][c] += 1
            V[v_b00][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1; W[w_c11][c] += 1
        elif s == 1:
            U[2][c] += 1; U[3][c] += 1
            V[v_b00][c] += 1
            W[w_c10][c] += 1; W[w_c11][c] += -1
        elif s == 2:
            U[0][c] += 1
            V[v_b01][c] += 1; V[v_b11][c] += -1
            W[w_c01][c] += 1; W[w_c11][c] += 1
        elif s == 3:
            U[3][c] += 1
            V[v_b10][c] += 1; V[v_b00][c] += -1
            W[w_c00][c] += 1; W[w_c10][c] += 1
        elif s == 4:
            U[0][c] += 1; U[1][c] += 1
            V[v_b11][c] += 1
            W[w_c00][c] += -1; W[w_c01][c] += 1
        elif s == 5:
            U[2][c] += 1; U[0][c] += -1
            V[v_b00][c] += 1; V[v_b01][c] += 1
            W[w_c11][c] += 1
        elif s == 6:
            U[1][c] += 1; U[3][c] += -1
            V[v_b10][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1

def _try_find_2x2x4_rank13(deadline):
    """Try to find a rank-13 decomposition for 2x2 x 2x4 matmul."""
    n, m, p = 2, 2, 4
    rank = 13
    
    # Nonzero tensor entries: (a, b, c) where T=1
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # A idx: i*m+k, B idx: k*p+j, C idx: j*n+i
    terms = []
    for i in range(n):
        for j in range(p):
            for k in range(m):
                terms.append((i*m+k, k*p+j, j*n+i))
    
    # Use alternating least squares / iterative refinement
    for seed in range(50):
        if _time.time() > deadline:
            break
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n*m, rank)) * 0.5
        V = rng.standard_normal((m*p, rank)) * 0.5
        W = rng.standard_normal((p*n, rank)) * 0.5
        
        for it in range(200):
            if _time.time() > deadline:
                break
            # Alternating: fix V,W solve U; fix U,W solve V; fix U,V solve W
            # Solve U: for each a, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # For fixed a: U[a,:] is a vector. The equation for each (b,c) with T[a,b,c]=1:
            #   U[a,:] dot (V[b,:]*W[c,:]) = 1
            # For each (b,c) with T[a,b,c]=0:
            #   U[a,:] dot (V[b,:]*W[c,:]) = 0
            # This is a linear system: M_a @ U[a,:] = rhs_a
            # where M_a rows are V[b,:]*W[c,:] for all (b,c), rhs is T[a,b,c]
            
            max_res = 0
            for a in range(n*m):
                # Build system
                rows = []
                rhs = []
                for b in range(m*p):
                    for c in range(p*n):
                        vec = V[b, :] * W[c, :]
                        rows.append(vec)
                        # Find T[a,b,c]
                        # a = i*m+k, b = k'*p+j, c = j'*n+i'
                        # T=1 iff i=i', k=k', j=j'
                        i = a // m; k = a % m
                        kp = b // p; j = b % p
                        jp = c // n; ip = c % n
                        if i == ip and k == kp and j == jp:
                            rhs.append(1.0)
                        else:
                            rhs.append(0.0)
                M = np.array(rows)
                rhs_arr = np.array(rhs)
                # Least squares solve
                sol, res, rank_s, sv = np.linalg.lstsq(M, rhs_arr, rcond=None)
                U[a, :] = sol
            
            for b in range(m*p):
                rows = []
                rhs = []
                for a in range(n*m):
                    for c in range(p*n):
                        vec = U[a, :] * W[c, :]
                        rows.append(vec)
                        i = a // m; k = a % m
                        kp = b // p; j = b % p
                        jp = c // n; ip = c % n
                        if i == ip and k == kp and j == jp:
                            rhs.append(1.0)
                        else:
                            rhs.append(0.0)
                M = np.array(rows)
                rhs_arr = np.array(rhs)
                sol, res, rank_s, sv = np.linalg.lstsq(M, rhs_arr, rcond=None)
                V[b, :] = sol
            
            for c in range(p*n):
                rows = []
                rhs = []
                for a in range(n*m):
                    for b in range(m*p):
                        vec = U[a, :] * V[b, :]
                        rows.append(vec)
                        i = a // m; k = a % m
                        kp = b // p; j = b % p
                        jp = c // n; ip = c % n
                        if i == ip and k == kp and j == jp:
                            rhs.append(1.0)
                        else:
                            rhs.append(0.0)
                M = np.array(rows)
                rhs_arr = np.array(rhs)
                sol, res, rank_s, sv = np.linalg.lstsq(M, rhs_arr, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            max_res = 0
            for (a, b, c) in terms:
                pred = np.dot(U[a, :], V[b, :] * W[c, :])
                max_res = max(max_res, abs(pred - 1.0))
            # Also check zeros
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        i = a // m; k = a % m
                        kp = b // p; j = b % p
                        jp = c // n; ip = c % n
                        if not (i == ip and k == kp and j == jp):
                            pred = np.dot(U[a, :], V[b, :] * W[c, :])
                            max_res = max(max_res, abs(pred))
            
            if max_res < 1e-8:
                return U, V, W
    
    return None

def construct():
    """
    Rank-39 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Tries rank-13 per 2x2x4 block; falls back to Strassen rank-42.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    deadline = _time.time() + 110
    
    # Try rank-13 for 2x2x4
    result = _try_find_2x2x4_rank13(deadline)
    
    if result is not None:
        U_24, V_24, W_24 = result
        num_blocks = p // 4  # 3
        rank = num_blocks * 13  # 39
        
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        
        for t in range(num_blocks):
            base = t * 13
            for r in range(13):
                c = base + r
                for a in range(n*m):
                    U[a][c] = float(U_24[a, r])
                for k in range(m):
                    for j in range(4):
                        b_local = k * 4 + j
                        b_global = k * p + 4*t + j
                        V[b_global][c] = float(V_24[b_local, r])
                for j in range(4):
                    for i in range(n):
                        c_local = j * n + i
                        c_global = (4*t + j) * n + i
                        W[c_global][c] = float(W_24[c_local, r])
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Strassen rank-42
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        _build_strassen_2x2x2(p, t, U, V, W, t * 7, n)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
