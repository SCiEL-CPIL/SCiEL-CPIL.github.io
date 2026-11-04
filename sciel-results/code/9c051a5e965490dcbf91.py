"""QAPLIB wil100 solver: vectorized best-improvement 2-opt with diverse initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _precompute_ab(a, b):
    """Precompute a*b element-wise for faster delta computation."""
    return a * b


def _swap_delta_vec(ab, b, p, i, j):
    """
    Vectorized O(n) computation of cost change when swapping positions i and j.
    Uses the decomposition:
      delta = sum_{k != i,j} a[i,k]*b[p_i,p_k] + a[k,j]*b[p_k,p_j]
            - a[i,k]*b[p_j,p_k] - a[k,j]*b[p_k,p_i]
            + a[j,i]*b[p_j,p_i] - a[i,j]*b[p_i,p_j]
    """
    n = len(p)
    pi = p[i]
    pj = p[j]

    # Build row/col slices excluding i and j
    # We need sum over k of:
    #   a[i,k] * (b[pi, p[k]] - b[pj, p[k]])   -- i's contribution at each k
    #   a[k,j] * (b[p[k], pj] - b[p[k], pi])   -- j's contribution at each k
    # Then handle the i<->j cross terms.

    # For efficiency, compute using full vectors and subtract i,j entries
    # term_k1[k] = a[i,k] * b[pi, p[k]]
    # term_k2[k] = a[i,k] * b[pj, p[k]]
    # term_k3[k] = a[k,j] * b[p[k], pj]
    # term_k4[k] = a[k,j] * b[p[k], pi]

    # b[pi, p] is a vector of length n: b[pi, p[0]], b[pi, p[1]], ...
    b_pi_p = b[pi, p]  # shape (n,)
    b_pj_p = b[pj, p]  # shape (n,)
    b_p_pj = b[p, pj]  # shape (n,)
    b_p_pi = b[p, pi]  # shape (n,)

    a_i = a[i, :]  # shape (n,)
    a_j_col = a[:, j]  # shape (n,)

    # delta_k = sum_{k} a[i,k]*(b[pi,pk] - b[pj,pk]) + a[k,j]*(b[pk,pj] - b[pk,pi])
    # but we must exclude k=i and k=j from this sum, and handle those separately.

    term1 = a_i * (b_pi_p - b_pj_p)  # contribution from row i at each k
    term2 = a_j_col * (b_p_pj - b_p_pi)  # contribution from col j at each k

    delta_k = np.sum(term1) + np.sum(term2)

    # Subtract the k=i terms (already included above)
    # At k=i: term1[i] = a[i,i]*(b[pi,pi] - b[pj,pi])
    #         term2[i] = a[i,j]*(b[pi,pj] - b[pi,pi])
    # At k=j: term1[j] = a[i,j]*(b[pi,pj] - b[pj,pj])
    #         term2[j] = a[j,j]*(b[pj,pj] - b[pj,pi])

    # The correct delta should NOT include k=i or k=j in the "other" sum.
    # Instead, the i,j cross terms are:
    #   + a[i,j]*b[pj,pi] - a[i,j]*b[pi,pj]  (i now at pj, j now at pi)
    #   - a[i,j]*b[pi,pj] + a[i,j]*b[pj,pi]  ... wait let me re-derive.

    # Original contribution involving (i,j) pair: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # New contribution: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    # So delta_cross = a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    #                = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])

    # For the k=i and k=j terms in delta_k, they represent interactions
    # between facility i (or j) and itself, and between i and j.
    # Let's just compute the full sum and subtract the i,j diagonal/self terms:
    #
    # The "other" sum should be over k not in {i,j}:
    #   sum_{k not in {i,j}} [a[i,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pj]-b[pk,pi])]
    #
    # We computed sum over ALL k. Subtract k=i and k=j:
    # k=i: a[i,i]*(b[pi,pi]-b[pj,pi]) + a[i,j]*(b[pi,pj]-b[pi,pi])
    # k=j: a[i,j]*(b[pi,pj]-b[pj,pj]) + a[j,j]*(b[pj,pj]-b[pj,pi])

    sub_i = a[i, i] * (b[pi, pi] - b[pj, pi]) + a[i, j] * (b[pi, pj] - b[pi, pi])
    sub_j = a[i, j] * (b[pi, pj] - b[pj, pj]) + a[j, j] * (b[pj, pj] - b[pj, pi])

    delta_other = delta_k - sub_i - sub_j

    # Self terms: facility i moves from pi to pj, facility j moves from pj to pi.
    # The a[i,i]*b[pi,pi] becomes a[i,i]*b[pj,pj], etc.
    delta_self_i = a[i, i] * (b[pj, pj] - b[pi, pi])
    delta_self_j = a[j, j] * (b[pi, pi] - b[pj, pj])

    # Cross term between i and j:
    delta_cross = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])

    return int(delta_other + delta_self_i + delta_self_j + delta_cross)


def _best_improvement_pass(ab, b, a, p, deadline):
    """
    One full pass of best-improvement: scan all pairs, apply the single best swap.
    Returns (p, cost, best_delta).
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    for i in range(n):
        if time.time() > deadline:
            break
        pi = p[i]
        # Precompute b[pi, p] for this i
        b_pi_p = b[pi, p]
        a_i = a[i, :]
        for j in range(i + 1, n):
            pj = p[j]
            delta = _swap_delta_vec(ab, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta

    return p, cost, best_delta


def _multi_best_improvement(ab, b, a, p, deadline, max_passes=200):
    """
    Run multiple best-improvement passes until no improvement or deadline.
    """
    n = len(p)
    cost = _cost(a, b, p)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        p, cost, bd = _best_improvement_pass(ab, b, a, p, deadline)
        if bd == 0:
            break
    return p, cost


def _first_improvement_pass(ab, b, a, p, deadline):
    """First-improvement: apply first negative-delta swap found, restart scan."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    passes = 0
    while improved and passes < 500:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _swap_delta_vec(ab, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break
            if improved:
                break
    return p, cost


def _spectral_init(a, b, n, rng):
    """Spectral initialization using eigenvector alignment."""
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(15, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]
    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    cost_matrix = -(fac_emb @ loc_emb.T)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _rank_init(a, b, n, rng, reverse=False):
    """Rank-matching initialization."""
    a_rs = a.sum(axis=1).astype(np.float64)
    b_rs = b.sum(axis=1).astype(np.float64)
    if reverse:
        heavy = np.argsort(a_rs, kind="stable")
        central = np.argsort(-b_rs, kind="stable")
    else:
        heavy = np.argsort(-a_rs, kind="stable")
        central = np.argsort(b_rs, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _hungarian_init(a, b, n, rng):
    """Hungarian initialization using surrogate cost matrix."""
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)
    # Surrogate: C[i,l] = a_row_sum[i] * b_row_sum[l]
    C = np.outer(a_rs.astype(np.float64), b_rs.astype(np.float64))
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _surrogate_init(a, b, p, n, rng):
    """Build surrogate from current assignment, solve, return new p."""
    n = len(p)
    # C[i,l] = sum_k a[i,k]*b[p[k],l] + sum_k a[k,i]*b[l,p[k]]
    # = (a @ b[:, p].T)[i,l] + (a.T @ b[p, :])[i,l]
    term1 = a @ b[:, p].T
    term2 = a.T @ b[p, :]
    C = (term1 + term2).astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(C)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    return p_new


def _frank_wolfe_init(a, b, p_init, n, deadline):
    """
    Frank-Wolfe (conditional gradient) on the Birkhoff polytope.
    Objective: f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k]*b[k,l]*X[j,l]
             = trace(X^T A X B^T)  ... let me be careful.
    
    Cost = sum_{i,j} a[i,j] * b[p[i],p[j]]
    Continuous: f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
               = sum_{k,l} b[k,l] * sum_{i,j} a[i,j] * X[i,k] * X[j,l]
               = sum_{k,l} b[k,l] * (X^T A X)[k,l]
               = <B, X^T A X> = trace(B^T X^T A X)
    
    Gradient wrt X: d/dX trace(B^T X^T A X)
    Let M = X^T A X, so f = trace(B^T M) = sum_kl B[k,l] M[k,l]
    M[k,l] = sum_ij X[i,k] A[i,j] X[j,l]
    df/dX[i,k] = sum_jl A[i,j] X[j,l] B[l,k] + sum_jl A[j,i] X[j,l] B[k,l]
               = (A X B^T)[i,k] + (A^T X B)[i,k]
    
    So grad = A @ X @ B.T + A.T @ X @ B
    """
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0

    # Initialize with a small perturbation to avoid being stuck
    eps = 0.01
    X = X * (1 - eps) + eps / n

    best_X = X.copy()
    best_val = np.inf

    step_size = 0.1
    for it in range(300):
        if time.time() > deadline:
            break

        # Gradient
        AX = a_sym @ X
        grad = AX @ b_sym + AX.T @ b_sym  # A X B^T + (A X B^T)^T for symmetric case
        # Actually for symmetric: grad = 2 * A X B
        grad = 2.0 * (a_sym @ X @ b_sym)

        # Linear minimization over Birkhoff polytope (Hungarian on -grad)
        row_ind, col_ind = linear_sum_assignment(-grad)
        V = np.zeros((n, n), dtype=np.float64)
        V[row_ind, col_ind] = 1.0

        # Frank-Wolfe step: X_new = (1 - gamma) * X + gamma * V
        # Adaptive step size
        X_new = (1 - step_size) * X + step_size * V

        # Evaluate
        val = _cost(a, b, np.empty(0))  # placeholder, compute properly
        # Actually compute continuous cost:
        M = X_new.T @ a_sym @ X_new
        c_cont = np.sum(b_sym * M)

        if c_cont < best_val:
            best_val = c_cont
            best_X = X_new.copy()

        X = X_new

        # Decay step size
        step_size *= 0.98

    # Extract permutation from best_X
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _multi_swap_perturb(p, n, rng, k):
    """Apply k random swaps."""
    p_new = p.copy()
    for _ in range(k):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new


def _block_reverse_perturb(p, n, rng):
    """Reverse a random block."""
    p_new = p.copy()
    block_len = rng.integers(3, min(15, n // 3))
    start = rng.integers(0, n - block_len + 1)
    p_new[start:start + block_len] = p_new[start:start + block_len][::-1]
    return p_new


def _surrogate_perturb(a, b, p, n, rng):
    """Perturb based on surrogate cost contributions."""
    p_new = p.copy()
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)
    contrib = a_rs * b_rs[p]
    k = rng.integers(3, 8)
    top = np.argsort(-contrib)[:k]
    for idx in top:
        j = rng.integers(0, n)
        if j != idx:
            p_new[idx], p_new[j] = p_new[j], p_new[idx]
    return p_new


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    ab = _precompute_ab(a, b)
    rng = np.random.default_rng(42)

    best_p = None
    best_cost = np.inf

    # Phase 1: Diverse initialization with intensive local search
    inits = []

    # 1a: Rank matching
    p_r = _rank_init(a, b, n, rng)
    inits.append(p_r)

    # 1b: Reverse rank matching
    p_rr = _rank_init(a, b, n, rng, reverse=True)
    inits.append(p_rr)

    # 1c: Spectral
    p_s = _spectral_init(a, b, n, rng)
    inits.append(p_s)

    # 1d: Hungarian surrogate
    p_h = _hungarian_init(a, b, n, rng)
    inits.append(p_h)

    # 1e: Random
    inits.append(rng.permutation(n).astype(np.int64))

    # 1f: Another random
    inits.append(rng.permutation(n).astype(np.int64))

    # Polish each initialization with best-improvement
    phase1_budget = 10.0
    time_per_init = phase1_budget / len(inits)
    for idx, p_init in enumerate(inits):
        t_end = t_start + (idx + 1) * time_per_init
        if time.time() > t_end - 0.5:
            t_end = time.time() + 0.5
        p_pol, c_pol = _multi_best_improvement(ab, b, a, p_init.copy(), t_end, max_passes=50)
        if c_pol < best_cost:
            best_cost = c_pol
            best_p = p_pol.copy()

    # Phase 2: Frank-Wolfe continuous relaxation (time-limited)
    if time.time() < deadline - 30.0:
        fw_deadline = min(time.time() + 15.0, deadline - 25.0)
        p_fw = _frank_wolfe_init(a, b, best_p, n, fw_deadline)
        p_fw_pol, c_fw_pol = _multi_best_improvement(ab, b, a, p_fw, deadline - 20.0, max_passes=30)
        if c_fw_pol < best_cost:
            best_cost = c_fw_pol
            best_p = p_fw_pol.copy()

    # Phase 3: Surrogate-guided restarts
    if time.time() < deadline - 25.0:
        for _ in range(3):
            if time.time() > deadline - 20.0:
                break
            p_surr = _surrogate_init(a, b, best_p, n, rng)
            # Small perturbation
            p_surr = _multi_swap_perturb(p_surr, n, rng, 3)
            p_pol, c_pol = _multi_best_improvement(ab, b, a, p_surr, deadline - 15.0, max_passes=40)
            if c_pol < best_cost:
                best_cost = c_pol
                best_p = p_pol.copy()

    # Phase 4: Iterated Local Search with mixed perturbations
    current_p = best_p.copy()
    current_cost = best_cost
    ils_deadline = deadline - 10.0
    temp = 1.0

    while time.time() < ils_deadline:
        # Choose perturbation
        r = rng.random()
        if r < 0.3:
            k = 2 + int(temp * 5)
            perturbed = _multi_swap_perturb(current_p, n, rng, k)
        elif r < 0.5:
            perturbed = _block_reverse_perturb(current_p, n, rng)
        elif r < 0.7:
            perturbed = _surrogate_perturb(a, b, current_p, n, rng)
        else:
            # Surrogate-guided + small random
            p_surr = _surrogate_init(a, b, current_p, n, rng)
            perturbed = _multi_swap_perturb(p_surr, n, rng, 2)

        p_pol, c_pol = _multi_best_improvement(ab, b, a, perturbed, ils_deadline, max_passes=30)

        if c_pol < best_cost:
            best_cost = c_pol
            best_p = p_pol.copy()
            current_p = p_pol.copy()
            current_cost = c_pol
            temp = max(0.2, temp * 0.9)
        elif c_pol < current_cost:
            delta = current_cost - c_pol
            accept_prob = np.exp(-delta / (temp * 500.0 + 1.0))
            if rng.random() < accept_prob:
                current_p = p_pol.copy()
                current_cost = c_pol
        else:
            temp = max(0.2, temp * 0.95)

    # Phase 5: Final intensive polish
    if time.time() < deadline - 2.0:
        final_p, final_cost = _multi_best_improvement(
            ab, b, a, best_p.copy(), deadline - 0.5, max_passes=100
        )
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()

    # Phase 6: One more first-improvement sweep if time remains
    if time.time() < deadline - 1.0:
        fi_p, fi_cost = _first_improvement_pass(ab, b, a, best_p.copy(), deadline - 0.2)
        if fi_cost < best_cost:
            best_cost = fi_cost
            best_p = fi_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
