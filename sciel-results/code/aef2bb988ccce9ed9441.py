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

def _try_find_1x2x12_rank12(deadline):
    """
    Try to find a rank-12 decomposition for 1x2 x 2x12 matmul.
    This is a vector (length 2) times matrix (2x12) -> vector (length 12).
    A idx: k (0,1), B idx: k*p+j, C idx: j
    T[a,b,c] = 1 iff a==b//p and c==b%p
    """
    n, m, p = 1, 2, 12
    rank = 12
    
    # Nonzero tensor entries
    terms = []
    for j in range(p):
        for k in range(m):
            terms.append((k, k*p+j, j))
    
    # Use alternating least squares
    for seed in range(100):
        if _time.time() > deadline:
            break
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((m, rank)) * 0.3
        V = rng.standard_normal((m*p, rank)) * 0.3
        W = rng.standard_normal((p, rank)) * 0.3
        
        for it in range(300):
            if _time.time() > deadline:
                break
            
            # Solve U: for each a, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            for a in range(m):
                rows = []
                rhs = []
                for b in range(m*p):
                    for c in range(p):
                        vec = V[b, :] * W[c, :]
                        rows.append(vec)
                        kp = b // p; j = b % p
                        if a == kp and c == j:
                            rhs.append(1.0)
                        else:
                            rhs.append(0.0)
                M = np.array(rows)
                rhs_arr = np.array(rhs)
                sol, _, _, _ = np.linalg.lstsq(M, rhs_arr, rcond=None)
                U[a, :] = sol
            
            # Solve V: for each b, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            for b in range(m*p):
                rows = []
                rhs = []
                for a in range(m):
                    for c in range(p):
                        vec = U[a, :] * W[c, :]
                        rows.append(vec)
                        kp = b // p; j = b % p
                        if a == kp and c == j:
                            rhs.append(1.0)
                        else:
                            rhs.append(0.0)
                M = np.array(rows)
                rhs_arr = np.array(rhs)
                sol, _, _, _ = np.linalg.lstsq(M, rhs_arr, rcond=None)
                V[b, :] = sol
            
            # Solve W: for each c, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            for c in range(p):
                rows = []
                rhs = []
                for a in range(m):
                    for b in range(m*p):
                        vec = U[a, :] * V[b, :]
                        rows.append(vec)
                        kp = b // p; j = b % p
                        if a == kp and c == j:
                            rhs.append(1.0)
                        else:
                            rhs.append(0.0)
                M = np.array(rows)
                rhs_arr = np.array(rhs)
                sol, _, _, _ = np.linalg.lstsq(M, rhs_arr, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            max_res = 0
            for (a, b, c) in terms:
                pred = np.dot(U[a, :], V[b, :] * W[c, :])
                max_res = max(max_res, abs(pred - 1.0))
            for a in range(m):
                for b in range(m*p):
                    for c in range(p):
                        kp = b // p; j = b % p
                        if not (a == kp and c == j):
                            pred = np.dot(U[a, :], V[b, :] * W[c, :])
                            max_res = max(max_res, abs(pred))
            
            if max_res < 1e-10:
                return U, V, W
    
    return None

def construct():
    """
    Rank-24 exact tensor decomposition via row-aggregation:
    Decompose 2x2x12 into two independent 1x2x12 subproblems.
    Falls back to Strassen rank-42 if rank-12 search fails.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    deadline = _time.time() + 115
    
    # Try rank-12 for 1x2x12 (each row independently)
    result = _try_find_1x2x12_rank12(deadline)
    
    if result is not None:
        U_12, V_12, W_12 = result
        rank = 2 * 12  # 24: two rows, each with rank-12
        
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        
        for row in range(n):
            base = row * 12
            for r in range(12):
                c = base + r
                # U: A[i,k] where i=row, k=0,1
                for k in range(m):
                    U[row*m+k][c] = float(U_12[k, r])
                # V: B[k,j]
                for k in range(m):
                    for j in range(p):
                        V[k*p+j][c] = float(V_12[k*p+j, r])
                # W: C[j,i] where i=row, j=0..11
                for j in range(p):
                    W[j*n+row][c] = float(W_12[j, r])
        
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
