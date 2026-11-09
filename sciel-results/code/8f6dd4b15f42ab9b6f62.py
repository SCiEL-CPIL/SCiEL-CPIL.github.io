"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)

    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11

    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c10, c11

    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c01, c11

    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1  # c00, c10

    # M5 = (a00+a10)b11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01

    # M6 = (a11-a10)(b00+b10)
    U[2, 5] = -1; U[3, 5] = 1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # c11

    # M7 = (a01-a11)(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c00

    return U, V, W


def _verify_decomp(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_2x4x2_from_strassen(U_s, V_s, W_s):
    """Build 2x4x2 decomposition (rank 14) from two 2x2x2 Strassen blocks."""
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Block 1: j in {0,1}
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 4 + jl, r] = U_s[i * 2 + jl, r]
                for k in range(2):
                    V[jl * 2 + k, r] = V_s[jl * 2 + k, r]
                    W[k * 2 + i, r] = W_s[k * 2 + i, r]

    # Block 2: j in {2,3}
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 4 + (jl + 2), r + 7] = U_s[i * 2 + jl, r]
                for k in range(2):
                    V[(jl + 2) * 2 + k, r + 7] = V_s[jl * 2 + k, r]
                    W[k * 2 + i, r + 7] = W_s[k * 2 + i, r]

    return U, V, W


def _build_2x4x6_from_2x4x2(U1, V1, W1):
    """
    Build 2x4x6 decomposition from a 2x4x2 block by replicating for k-blocks.
    
    The 2x4x2 block covers k in {0,1}. To extend to k in {0,1,2,3,4,5}, we add
    three copies of the block shifted by 0, 2, 4 in the k-index.
    
    This gives rank 42.
    """
    n, m, p1 = 2, 4, 2
    rank1 = U1.shape[1]  # 14
    p2 = 6
    rank2 = rank1 * 3  # 42
    
    U = np.zeros((n * m, rank2), dtype=float)
    V = np.zeros((m * p2, rank2), dtype=float)
    W = np.zeros((p2 * n, rank2), dtype=float)
    
    for b in range(3):
        offset = b * rank1
        U[:, offset:offset + rank1] = U1
        for j in range(m):
            for k in range(p1):
                V[j * p2 + (k + b * p1), offset:offset + rank1] = V1[j * p1 + k, :]
        for k in range(p1):
            for i in range(n):
                W[(k + b * p1) * n + i, offset:offset + rank1] = W1[k * p1 + i, :]
    
    return U, V, W


def _try_randomized_improvement(U0, V0, W0, n, m, p, rank, deadline, seed=42):
    """
    Attempt to find a lower-rank decomposition using greedy set-cover on sparse witness patterns.
    
    Strategy: Generate candidate rank-1 terms (sparse U, V, W vectors), then greedily select
    a minimal subset whose linear combination exactly reproduces the multiplication tensor.
    
    Returns (U, V, W, new_rank) if improvement found, else (None, None, None, rank).
    """
    if time.time() > deadline:
        return None, None, None, rank
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Generate candidate sparse patterns
    rng = np.random.RandomState(seed)
    candidates = []
    
    # Add the known schoolbook terms as candidates
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                candidates.append((u, v, w))
    
    # Generate random sparse candidates
    num_random = 200
    for _ in range(num_random):
        if time.time() > deadline - 2:
            break
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        # Random sparse support
        u_supp = set(rng.choice(n * m, size=rng.randint(1, 4), replace=False))
        v_supp = set(rng.choice(m * p, size=rng.randint(1, 4), replace=False))
        w_supp = set(rng.choice(p * n, size=rng.randint(1, 4), replace=False))
        
        for idx in u_supp:
            u[idx] = rng.choice([-1, 1])
        for idx in v_supp:
            v[idx] = rng.choice([-1, 1])
        for idx in w_supp:
            w[idx] = rng.choice([-1, 1])
        
        candidates.append((u, v, w))
    
    # Greedy set-cover: select candidates that cover the most uncovered (i,j,k) tuples
    all_tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    covered = {t: 0.0 for t in all_tuples}
    selected = []
    
    # Pre-compute coverage for each candidate
    def compute_coverage(u, v, w):
        cov = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = u[i * m + j] * v[j * p + k] * w[k * n + i]
                    if val != 0:
                        cov[(i, j, k)] = val
        return cov
    
    candidate_coverages = [compute_coverage(u, v, w) for u, v, w in candidates]
    
    # Greedy selection
    remaining = list(range(len(candidates)))
    max_iter = min(len(candidates), 50)
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        if not remaining:
            break
        
        # Find candidate that covers most uncovered tuples
        best_idx = -1
        best_gain = 0
        for idx in remaining:
            gain = 0
            for t in all_tuples:
                if abs(covered[t]) < 1e-10:  # Not yet covered
                    if t in candidate_coverages[idx]:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx == -1 or best_gain == 0:
            break
        
        # Select this candidate
        selected.append(best_idx)
        for t, val in candidate_coverages[best_idx].items():
            covered[t] += val
        
        remaining.remove(best_idx)
        
        # Check if all tuples are covered
        if all(abs(covered[t] - 1.0) < 1e-10 for t in all_tuples):
            break
    
    if not selected:
        return None, None, None, rank
    
    # Build the decomposition from selected candidates
    new_rank = len(selected)
    if new_rank >= rank:
        return None, None, None, rank
    
    U_new = np.zeros((n * m, new_rank))
    V_new = np.zeros((m * p, new_rank))
    W_new = np.zeros((p * n, new_rank))
    
    for r, idx in enumerate(selected):
        U_new[:, r] = candidates[idx][0]
        V_new[:, r] = candidates[idx][1]
        W_new[:, r] = candidates[idx][2]
    
    # Verify
    if _verify_decomp(U_new, V_new, W_new, n, m, p, new_rank):
        return U_new, V_new, W_new, new_rank
    
    return None, None, None, rank


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 6

    U_s, V_s, W_s = _strassen_2x2x2()

    if not _verify_decomp(U_s, V_s, W_s, 2, 2, 2, 7):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U1, V1, W1 = _build_2x4x2_from_strassen(U_s, V_s, W_s)

    if not _verify_decomp(U1, V1, W1, 2, 4, 2, 14):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_total, V_total, W_total = _build_2x4x6_from_2x4x2(U1, V1, W1)
    total_rank = U_total.shape[1]  # 42

    if not _verify_decomp(U_total, V_total, W_total, n, m, p, total_rank):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try randomized improvement if we have time
    if time.time() < deadline - 5.0:
        U_imp, V_imp, W_imp, imp_rank = _try_randomized_improvement(
            U_total, V_total, W_total, n, m, p, total_rank, deadline
        )
        if U_imp is not None and imp_rank < total_rank:
            if _verify_decomp(U_imp, V_imp, W_imp, n, m, p, imp_rank):
                U_total, V_total, W_total, total_rank = U_imp, V_imp, W_imp, imp_rank

    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
