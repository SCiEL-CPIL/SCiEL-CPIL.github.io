"""Multi-start local search for Maximum Cut with spectral initialization,
pairwise and triple flips, ILS, and SA polish.

Improvements:
1. Better pairwise flip candidate selection (neighbors of top candidates).
2. Triple-flip moves to escape deeper local optima.
3. More aggressive SA with multiple restarts.
4. Greedy construction initialization.
5. Tighter time management (115s deadline).
6. Vectorized operations for speed.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def _cut_value(s, A, E):
    """Compute cut value: s in {+1,-1}, cut = E/2 - 0.25 * s^T A s."""
    return 0.5 * E - 0.25 * float(s @ (A @ s))


def _fiedler_vector(A, n, rng):
    """Compute Fiedler vector sign vector. Falls back to random."""
    try:
        A_sparse = csr_matrix(A)
        D = np.diag(A.sum(axis=1))
        L = D - A_sparse
        L = csr_matrix(L)
        try:
            vals, vecs = eigs(L, k=2, sigma=0.0, which='LM', maxiter=500)
            idx = np.argsort(vals.real)
            fiedler_vec = vecs[:, idx[1]].real
        except Exception:
            L_dense = L.toarray()
            vals, vecs = np.linalg.eigh(L_dense)
            fiedler_vec = vecs[:, 1]
        s = np.sign(fiedler_vec)
        s[s == 0] = 1.0
        return s
    except Exception:
        return rng.choice(np.array([-1.0, 1.0]), size=n)


def _greedy_init(A, n, rng):
    """Greedy construction: add vertices one by one, choosing the side that
    maximizes immediate cut gain."""
    s = np.zeros(n)
    remaining = list(range(n))
    rng.shuffle(remaining)
    # Start with a random vertex
    first = remaining.pop()
    s[first] = 1.0
    # For each remaining vertex, assign to the side that gives more cut edges
    for v in remaining:
        # Count edges to +1 side and -1 side
        neighbors = np.where(A[v] > 0)[0]
        if len(neighbors) == 0:
            s[v] = 1.0
            continue
        pos_edges = np.sum(A[v, neighbors] * (s[neighbors] == 1.0))
        neg_edges = np.sum(A[v, neighbors] * (s[neighbors] == -1.0))
        if pos_edges >= neg_edges:
            s[v] = -1.0
        else:
            s[v] = 1.0
    s[s == 0] = 1.0
    return s


def _local_search(s, A, E, deadline, rng=None, use_pairwise=True, use_triple=False):
    """Steepest-ascent local search with incremental Ax update.
    Supports single-flip, pairwise-flip, and triple-flip moves."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    Ax = A @ s

    if rng is None:
        rng = np.random.default_rng()

    # Precompute neighbor lists for pairwise candidate selection
    neighbor_sets = [np.where(A[i] > 0)[0] for i in range(n)]

    while True:
        if time.time() > deadline:
            break

        # --- Single flip move ---
        gains = s * Ax
        i = int(np.argmax(gains))
        best_single_gain = gains[i]

        # --- Pairwise flip move ---
        best_pair_gain = -np.inf
        best_pair_i, best_pair_j = -1, -1

        if use_pairwise:
            # Candidate set: top-15 single-flip candidates + their neighbors
            K = min(15, n)
            top_idx = np.argsort(-gains)[:K]
            candidates = set(top_idx.tolist())
            for idx in top_idx:
                candidates.update(neighbor_sets[idx].tolist())
            candidates = list(candidates)

            # Also include some random candidates for diversity
            if len(candidates) < 30:
                extra = rng.choice(n, size=min(30 - len(candidates), n), replace=False)
                candidates.extend(extra.tolist())

            # Limit candidate set size
            if len(candidates) > 60:
                candidates = candidates[:60]

            # Check all pairs in candidate set
            for ii in range(len(candidates)):
                i2 = candidates[ii]
                for jj in range(ii + 1, len(candidates)):
                    j2 = candidates[jj]
                    pair_gain = gains[i2] + gains[j2] - 2.0 * s[i2] * s[j2] * A[i2, j2]
                    if pair_gain > best_pair_gain:
                        best_pair_gain = pair_gain
                        best_pair_i, best_pair_j = i2, j2

        # --- Triple flip move (optional) ---
        best_triple_gain = -np.inf
        best_triple_i, best_triple_j, best_triple_k = -1, -1, -1

        if use_triple:
            # Only check triples among top-8 single-flip candidates
            K3 = min(8, n)
            top3 = np.argsort(-gains)[:K3]
            for ii in range(len(top3)):
                i2 = top3[ii]
                for jj in range(ii + 1, len(top3)):
                    j2 = top3[jj]
                    for kk in range(jj + 1, len(top3)):
                        k2 = top3[kk]
                        # Triple flip gain:
                        # g_ijk = g_i + g_j + g_k
                        #        - 2*s_i*s_j*A_ij - 2*s_j*s_k*A_jk - 2*s_i*s_k*A_ik
                        #        + 4*s_i*s_j*s_k*A_ij*A_jk... 
                        # Actually, let me derive properly.
                        # When flipping i, j, k:
                        # Change in s^T A s = -2*s_i*sum_q A[i,q]*s[q] - 2*s_j*sum_q A[j,q]*s[q] - 2*s_k*sum_q A[k,q]*s[q]
                        #   + 4*s_i*s_j*A[i,j] + 4*s_j*s_k*A[j,k] + 4*s_i*s_k*A[i,k]
                        #   - 8*s_i*s_j*s_k*A[i,j]*A[j,k]... no, this is getting complex.
                        # 
                        # Simpler: the gain from flipping i is g_i = s_i * Ax_i
                        # After flipping i, the gain from flipping j becomes:
                        #   s_j * (Ax_j - 2*s_i*A[j,i])
                        # After flipping i and j, the gain from flipping k becomes:
                        #   s_k * (Ax_k - 2*s_i*A[k,i] - 2*s_j*A[k,j])
                        # Total gain = g_i + s_j*(Ax_j - 2*s_i*A[j,i]) + s_k*(Ax_k - 2*s_i*A[k,i] - 2*s_j*A[k,j])
                        # = s_i*Ax_i + s_j*Ax_j - 2*s_i*s_j*A[i,j] + s_k*Ax_k - 2*s_i*s_k*A[i,k] - 2*s_j*s_k*A[j,k]
                        triple_gain = (gains[i2] + gains[j2] + gains[k2]
                                      - 2.0 * s[i2] * s[j2] * A[i2, j2]
                                      - 2.0 * s[i2] * s[k2] * A[i2, k2]
                                      - 2.0 * s[j2] * s[k2] * A[j2, k2])
                        if triple_gain > best_triple_gain:
                            best_triple_gain = triple_gain
                            best_triple_i, best_triple_j, best_triple_k = i2, j2, k2

        # Decide which move to take
        if best_triple_gain > best_pair_gain and best_triple_gain > best_single_gain and best_triple_gain > 1e-9:
            i2, j2, k2 = best_triple_i, best_triple_j, best_triple_k
            old_s_i, old_s_j, old_s_k = s[i2], s[j2], s[k2]
            s[i2] = -s[i2]
            s[j2] = -s[j2]
            s[k2] = -s[k2]
            Ax -= 2.0 * old_s_i * A[:, i2]
            Ax -= 2.0 * old_s_j * A[:, j2]
            Ax -= 2.0 * old_s_k * A[:, k2]
        elif best_pair_gain > best_single_gain and best_pair_gain > 1e-9:
            i2, j2 = best_pair_i, best_pair_j
            old_s_i, old_s_j = s[i2], s[j2]
            s[i2] = -s[i2]
            s[j2] = -s[j2]
            Ax -= 2.0 * old_s_i * A[:, i2]
            Ax -= 2.0 * old_s_j * A[:, j2]
        elif best_single_gain > 1e-9:
            old_s_i = s[i]
            s[i] = -s[i]
            Ax -= 2.0 * old_s_i * A[:, i]
        else:
            break

    return s, _cut_value(s, A, E)


def _perturb(s, rng, k_min=5, k_max=15):
    """Randomly flip k vertices."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    k = int(rng.integers(k_min, k_max + 1))
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _perturb_adaptive(s, rng, no_improve, n):
    """Adaptive perturbation: larger when stuck longer."""
    s = s.astype(np.float64).copy()
    base_k = 5 + min(no_improve, 20)
    k = int(rng.integers(max(3, base_k - 3), base_k + 10))
    k = min(k, n // 2)
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _simulated_annealing(s, A, E, deadline, rng, t_start=5.0, t_end=0.1, cooling=0.999):
    """Simulated annealing with incremental Ax update and best-solution tracking."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    Ax = A @ s
    best_s = s.copy()
    best_val = _cut_value(s, A, E)
    cur = best_val
    T = t_start

    while T > t_end:
        if time.time() > deadline:
            break
        i = int(rng.integers(0, n))
        gain = s[i] * Ax[i]

        if gain > 0 or (gain < 0 and rng.random() < np.exp(gain / T)):
            old_s_i = s[i]
            s[i] = -s[i]
            Ax -= 2.0 * old_s_i * A[:, i]
            cur += gain
            if cur > best_val:
                best_val = cur
                best_s = s.copy()

        T *= cooling

    return best_s, best_val


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    E = 0.5 * float(A.sum())

    start = time.time()
    deadline = start + 115.0  # Safety margin under 150s limit
    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = []

    # 1. Spectral (Fiedler vector) initialization
    fiedler_s = _fiedler_vector(A, n, rng)
    inits.append(fiedler_s)
    inits.append(-fiedler_s)

    # 2. Greedy construction
    greedy_s = _greedy_init(A, n, rng)
    inits.append(greedy_s)
    inits.append(-greedy_s)

    # 3. Structured starts
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1.0
    inits.append(alt)
    inits.append(-alt)

    # 4. Random starts
    for _ in range(20):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=n))

    best_s = None
    best_val = -1.0

    # Phase 1: Local search from all starts (with pairwise flips)
    for s0 in inits:
        if time.time() > deadline - 35.0:
            break
        s, val = _local_search(s0, A, E, deadline - 35.0, rng=rng, use_pairwise=True, use_triple=False)
        if val > best_val:
            best_val = val
            best_s = s.copy()

    # Phase 2: ILS on the best solution with adaptive perturbation
    if best_s is not None and time.time() < deadline - 30.0:
        current_s = best_s.copy()
        current_val = best_val
        no_improve_count = 0
        max_iter = 3000
        for _ in range(max_iter):
            if time.time() > deadline - 30.0:
                break
            if no_improve_count > 5:
                pert_s = _perturb_adaptive(current_s, rng, no_improve_count, n)
            else:
                pert_s = _perturb(current_s, rng)
            ls_s, ls_val = _local_search(pert_s, A, E, deadline - 30.0, rng=rng, use_pairwise=True, use_triple=False)
            if ls_val > current_val:
                current_s = ls_s
                current_val = ls_val
                no_improve_count = 0
                if current_val > best_val:
                    best_val = current_val
                    best_s = current_s.copy()
            else:
                no_improve_count += 1

    # Phase 3: Simulated annealing polish (multiple restarts)
    if best_s is not None and time.time() < deadline - 15.0:
        sa_deadline = deadline - 15.0
        for sa_run in range(5):
            if time.time() > sa_deadline:
                break
            s, val = _simulated_annealing(
                best_s, A, E, sa_deadline, rng,
                t_start=5.0, t_end=0.1, cooling=0.999
            )
            s, val = _local_search(s, A, E, sa_deadline, rng=rng, use_pairwise=True, use_triple=False)
            if val > best_val:
                best_val = val
                best_s = s.copy()

    # Phase 4: Final high-precision local search with triple flips
    if best_s is not None and time.time() < deadline:
        s, val = _local_search(best_s, A, E, deadline, rng=rng, use_pairwise=True, use_triple=True)
        if val > best_val:
            best_val = val
            best_s = s

    return (best_s > 0).astype(int)
