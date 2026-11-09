"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.

Strategy:
Partition the 5 j-indices into groups G0={0,1}, G1={2,3}, G2={4}
Partition the 5 k-indices into groups H0={0,1}, H1={2,3}, H2={4}

This creates a 3x3 block structure:
- A is 2x5, viewed as [A0 | A1 | A2] where A0,A1 are 2x2 and A2 is 2x1
- B is 5x5, viewed as 3x3 blocks B[ab] of sizes |Ga| x |Hb|
- C is 2x5, viewed as [C0 | C1 | C2] where C0,C1 are 2x2 and C2 is 2x1

For each block pair (a,b) where |Ga|=2 and |Hb|=2 (i.e., a in {0,1}, b in {0,1}):
  Use Strassen's 7-factor 2x2x2 algorithm.
For all other blocks:
  Use schoolbook (rank = 2*|Ga|*|Hb|).

Total rank = 4*7 + (2*2*1 + 2*1*2 + 2*2*1 + 2*1*2 + 2*1*1)
           = 28 + (4 + 4 + 4 + 4 + 2)
           = 28 + 18
           = 46
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

_DEADLINE_START = None
_TIME_BUDGET = 135.0  # seconds, keep 15s safety margin before 150s kill


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


def _verify(U, V, W, n, m, p, num_tests=5):
    """Verify decomposition using multiple random test matrices."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    for test in range(num_tests):
        rng = np.random.RandomState(1000 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        
        # W uses k*n+i indexing, so C_flat[l] where l = k*n + i corresponds to C[i,k]
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        
        C_comp = np.zeros(p * n)
        for c in range(rank):
            vA = float(U_np[:, c] @ A)
            vB = float(V_np[:, c] @ B)
            if vA != 0.0 and vB != 0.0:
                C_comp += W_np[:, c] * vA * vB
        
        if not np.allclose(C_comp, C_true_wn, atol=1e-9):
            return False
    
    return True


def _build_strassen_2x5x5():
    """
    Build a rank-46 decomposition for 2x5x5.
    
    Partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4}.
    
    For 2x2 blocks: use Strassen's 7 factors.
    For other blocks: schoolbook.
    
    Total rank = 4*7 + schoolbook for remaining blocks
    Schoolbook blocks: (a=0,b=2): 2*2*1=4, (a=1,b=2): 2*2*1=4,
                       (a=2,b=0): 2*1*2=4, (a=2,b=1): 2*1*2=4,
                       (a=2,b=2): 2*1*1=2
    Total = 28 + 4+4+4+4+2 = 28+18 = 46
    """
    n, m, p = N, M, P
    
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    # Strassen's 7 factors for 2x2x2
    # Each: (u_entries, v_entries, w_entries)
    # u_entries: [(i, j_local, coeff), ...]
    # v_entries: [(j_local, k_local, coeff), ...]
    # w_entries: [(k_local, i, coeff), ...]
    
    strassen_7 = [
        # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        # m2 = (A[1,0]+A[1,1]) * B[0,0]
        # C[1,0] += 1, C[1,1] -= 1
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        # m3 = A[0,0] * (B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        # m4 = A[1,1] * (B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        # m5 = (A[0,0]+A[0,1]) * B[1,1]
        # C[0,0] -= 1, C[0,1] += 1
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # C[1,1] += 1
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # C[0,0] += 1
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]
    
    U_entries = []  # (factor_id, u_row, value)
    V_entries = []  # (factor_id, v_row, value)
    W_entries = []  # (factor_id, w_row, value)
    
    fid = 0
    for a in range(3):
        for b in range(3):
            jg = j_groups[a]
            kg = k_groups[b]
            sj = len(jg)
            sk = len(kg)
            
            if sj == 2 and sk == 2:
                for (u_ent, v_ent, w_ent) in strassen_7:
                    for (i, jl, c) in u_ent:
                        gj = jg[jl]
                        U_entries.append((fid, i * m + gj, c))
                    for (jl, kl, c) in v_ent:
                        gj = jg[jl]
                        gk = kg[kl]
                        V_entries.append((fid, gj * p + gk, c))
                    for (kl, i, c) in w_ent:
                        gk = kg[kl]
                        W_entries.append((fid, gk * n + i, c))
                    fid += 1
            else:
                for i in range(n):
                    for jl in range(sj):
                        for kl in range(sk):
                            gj = jg[jl]
                            gk = kg[kl]
                            U_entries.append((fid, i * m + gj, 1))
                            V_entries.append((fid, gj * p + gk, 1))
                            W_entries.append((fid, gk * n + i, 1))
                            fid += 1
    
    rank = fid
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (f, r, v) in U_entries:
        U[r][f] += v
    for (f, r, v) in V_entries:
        V[r][f] += v
    for (f, r, v) in W_entries:
        W[r][f] += v
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = N, M, P
    
    # Build the Strassen-based rank-46 construction
    result = _build_strassen_2x5x5()
    
    # Verify it works
    if _verify(result["U"], result["V"], result["W"], n, m, p, num_tests=5):
        return result
    
    # Fallback to schoolbook rank 50
    return _build_schoolbook()
