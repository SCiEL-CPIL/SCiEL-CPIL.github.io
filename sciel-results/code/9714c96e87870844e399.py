"""Kissing number in dimension 11 - Constructive D11+ lattice with SA fallback."""
import numpy as np
import itertools
import time

D = 11

def _verify(vectors):
    """Check nonzero, min pairwise distance >= max norm."""
    if vectors.size == 0:
        return False
    norms2 = np.sum(vectors * vectors, axis=1)
    if np.any(norms2 == 0):
        return False
    max_norm2 = np.max(norms2)
    # min pairwise squared distance
    # Use GEMM for distance matrix
    G = vectors @ vectors.T
    d2 = norms2[:, None] + norms2[None, :] - 2 * G
    np.fill_diagonal(d2, np.inf)
    min_d2 = np.min(d2)
    return min_d2 >= max_norm2 - 1e-9

def _constructive_d11plus():
    """Generate shortest vectors of D11+ lattice (norm^2 = 4).
    Returns integer array of shape (k, 11)."""
    vecs = []
    # Weight-4 vectors: four nonzero entries in {+1,-1}, even coordinate sum.
    # Even sum means an even number of -1's among the four (0,2,4 minuses).
    # Count: C(11,4) * (C(4,0)+C(4,2)+C(4,4)) = 330 * (1+6+1) = 330*8 = 2640.
    for positions in itertools.combinations(range(D), 4):
        for nminus in (0, 2, 4):
            for minus_pos in itertools.combinations(range(4), nminus):
                v = [0] * D
                for i, pos in enumerate(positions):
                    v[pos] = -1 if i in minus_pos else 1
                vecs.append(v)
    # Weight-2 vectors: two nonzero entries, one +1 and one -1 (odd sum = odd).
    # Actually for D11+ the odd-parity part: vectors with two nonzero entries
    # (+1,-1) give coordinate sum 0 (even) -> in D11, norm^2=2 not 4.
    # The 104 additional: weight-2 with entries (+1,+1)? sum=2 even, in D11, norm^2=2.
    # The true 104 are the half-integer vectors with norm^2=4:
    # (1/2,...,1/2) has norm^2=11/4. To get norm^2=4=16/4, need 16 of the 11
    # half-entries to be... not possible with 11 entries.
    # Reconsider: D11+ = {x in Z^11 : sum even} union {x in (Z+1/2)^11 : ...}.
    # The 2304 = 2200 + 104. 2200 weight-4 even-sum + 104 of form with two
    # entries being +/-1 and the rest... Let's just generate weight-4 even-sum
    # (2640) and weight-2 with entries (+1,-1) scaled? No.
    # Safest: generate weight-4 even-sum AND weight-2 (+1,+1) and (+1,-1),
    # then verify; keep only those that pass pairwise distance check globally.
    for positions in itertools.combinations(range(D), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[positions[0]] = s1
                v[positions[1]] = s2
                vecs.append(v)
    arr = np.array(vecs, dtype=np.int64)
    return arr

def _sa_fallback():
    """Original SA-based search for max independent set of W4 + all W1."""
    start_time = time.time()
    deadline = start_time + 110

    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    vectors_v4 = np.array(v4_list, dtype=np.int64)
    N4 = len(vectors_v4)

    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def polish(in_set):
        in_set = in_set.copy()
        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1
        changed = True
        while changed:
            changed = False
            cand = np.where((cc == 0) & (~in_set))[0]
            for v in cand:
                if cc[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    changed = True
        return in_set, int(in_set.sum())

    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            if len(not_in_set) == 0:
                break
            r = rng.random()
            if r < 0.55 and len(not_in_set) > 0:
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1
            elif r < 0.75 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    stuck += 1
                else:
                    stuck += 1
            else:
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                        stuck += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                cc[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            cc[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck = 0
                            else:
                                stuck += 1
                        else:
                            stuck += 1
            if stuck > 2000:
                T = max(T, 2.0)
                stuck = 0
            T *= cooling
            if T < min_T:
                T = min_T
        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            if len(cands) > 1:
                scores = np.array([np.sum(neighbor_counts[neighbors[c]]) for c in cands])
                chosen = cands[int(np.argmin(scores))]
            else:
                chosen = cands[0]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0 and len(to_remove) > 0:
                degrees[rem] -= np.sum(conflicts[rem][:, to_remove], axis=1)
        return in_set

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    in_set_greedy = greedy_min_degree()
    if time.time() < deadline - 15:
        sa_time = min((deadline - 15 - time.time()) * 0.4, 12.0)
        set_g, size_g = sa_local_search(in_set_greedy, max_time=sa_time)
        if size_g > best_size:
            best_size = size_g
            best_in_set = set_g.copy()
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        for restart in range(2):
            if time.time() >= deadline - 10:
                break
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            n_rem = max(2, int(len(indices) * 0.15))
            if len(indices) > n_rem:
                to_remove = rng.choice(indices, size=n_rem, replace=False)
                perturbed[to_remove] = False
            sa_time = min(remaining / 2, 10.0)
            set_p, size_p = sa_local_search(perturbed, max_time=sa_time)
            if size_p > best_size:
                best_size = size_p
                best_in_set = set_p.copy()

    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    v1_list = []
    for pos in range(D):
        v = [0] * D
        v[pos] = 2
        v1_list.append(v)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    return np.vstack([result_v4, vectors_v1])

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Primary: constructive D11+ generation.
    # Generate weight-4 even-sum vectors (the core of the kissing config).
    # Then verify the full set meets the distance condition.
    try:
        cand = _constructive_d11plus()
        # Filter to those with norm^2 == 4 (weight-4 vectors).
        n2 = np.sum(cand * cand, axis=1)
        cand = cand[n2 == 4]
        # Now we need a subset with min pairwise distance^2 >= 4.
        # For weight-4 vectors, dot <= 2 iff distance^2 >= 4.
        # The D11+ kissing set is a maximal such set. Take all weight-4 even-sum
        # and check; if the full set passes, return it. Otherwise fall back to SA.
        if len(cand) > 0 and time.time() < deadline - 20:
            if _verify(cand):
                return cand
        # If full candidate set fails, restrict to weight-4 even-sum only and verify.
        w4_even = []
        for positions in itertools.combinations(range(D), 4):
            for nminus in (0, 2, 4):
                for minus_pos in itertools.combinations(range(4), nminus):
                    v = [0] * D
                    for i, pos in enumerate(positions):
                        v[pos] = -1 if i in minus_pos else 1
                    w4_even.append(v)
        w4_even = np.array(w4_even, dtype=np.int64)
        if len(w4_even) > 0 and time.time() < deadline - 20:
            if _verify(w4_even):
                return w4_even
    except Exception:
        pass

    # Fallback: SA-based search (guaranteed valid by construction of independent set).
    result = _sa_fallback()
    return result
