import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 120s kill

    def build_structure(n):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)

        triple_to_4sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_4sets[t].append(qidx)
        triple_to_5sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_5sets[t].append(qidx)
        triple_to_4sets = [np.array(x, dtype=np.int32) for x in triple_to_4sets]
        triple_to_5sets = [np.array(x, dtype=np.int32) for x in triple_to_5sets]

        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        return M, triples, quad_triples, quint_triples, triple_to_4sets, triple_to_5sets, nwords, quad_masks, quint_masks

    def init_colors(n, seed, strategy='modular'):
        rng = np.random.RandomState(seed)
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        if strategy == 'random':
            return (rng.randint(0, 2, size=M)).astype(np.int8)
        elif strategy == 'spectral':
            # Use eigenvalue-based initialization: color by sign of principal eigenvector of adjacency
            # Approximate with a random projection + threshold
            A = np.zeros((n, n))
            for a, b, c in triples:
                A[a, b] += 1
                A[a, c] += 1
                A[b, c] += 1
            try:
                eigvals, eigvecs = np.linalg.eigh(A)
                # Use the eigenvector corresponding to the largest eigenvalue
                idx = np.argmax(eigvals)
                vec = eigvecs[:, idx]
                colors = np.zeros(M, dtype=np.int8)
                for i, (a, b, c) in enumerate(triples):
                    val = vec[a] + vec[b] + vec[c]
                    colors[i] = 0 if val > 0 else 1
            except:
                colors = (rng.randint(0, 2, size=M)).astype(np.int8)
            return colors
        else:
            # Modular: (a+b+c) % 3
            colors = np.zeros(M, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1
            return colors

    def search_for_n(n, time_limit, seed=0):
        M, triples, quad_triples, quint_triples, triple_to_4sets, triple_to_5sets, nwords, quad_masks, quint_masks = build_structure(n)

        rng = np.random.RandomState(seed)

        # Try multiple initializations
        strategies = ['modular', 'random', 'spectral']
        best_result = None
        best_violations = float('inf')

        for strat in strategies:
            if time.time() > start_time + time_limit:
                break
            colors = init_colors(n, seed + hash(strat) % 10000, strat)

            # Build red/blue bitsets
            red_set = np.zeros(nwords, dtype=np.uint64)
            blue_set = np.zeros(nwords, dtype=np.uint64)
            for t in range(M):
                if colors[t] == 0:
                    red_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
                else:
                    blue_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

            def count_violations(red_set, blue_set):
                r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
                b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
                return int(r4) + int(b5)

            def get_violating_quads(red_set):
                return np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]

            def get_violating_quints(blue_set):
                return np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]

            def flip_score_single(t, red_set, blue_set):
                old = colors[t]
                if old == 0:
                    qmask = quad_masks[triple_to_4sets[t]]
                    broken = np.all((qmask & red_set[None, :]) == qmask, axis=1).sum()
                    qmask = quint_masks[triple_to_5sets[t]]
                    without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                    created = np.all((without_t & blue_set[None, :]) == without_t, axis=1).sum()
                    return int(broken) - int(created)
                else:
                    qmask = quint_masks[triple_to_5sets[t]]
                    broken = np.all((qmask & blue_set[None, :]) == qmask, axis=1).sum()
                    qmask = quad_masks[triple_to_4sets[t]]
                    without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                    created = np.all((without_t & red_set[None, :]) == without_t, axis=1).sum()
                    return int(broken) - int(created)

            def apply_flip(t, red_set, blue_set):
                old = colors[t]
                bit = np.uint64(1) << np.uint64(t & 63)
                w = t >> 6
                if old == 0:
                    red_set[w] &= ~bit
                    blue_set[w] |= bit
                    colors[t] = 1
                else:
                    blue_set[w] &= ~bit
                    red_set[w] |= bit
                    colors[t] = 0
                return old

            def flip_pair_score(t1, t2, red_set, blue_set):
                """Score of flipping both t1 and t2 simultaneously."""
                old1, old2 = colors[t1], colors[t2]
                bit1 = np.uint64(1) << np.uint64(t1 & 63)
                bit2 = np.uint64(1) << np.uint64(t2 & 63)
                w1, w2 = t1 >> 6, t2 >> 6

                # Temporarily remove both from their current sets
                new_red = red_set.copy()
                new_blue = blue_set.copy()
                if old1 == 0:
                    new_red[w1] &= ~bit1
                    new_blue[w1] |= bit1
                else:
                    new_blue[w1] &= ~bit1
                    new_red[w1] |= bit1
                if old2 == 0:
                    new_red[w2] &= ~bit2
                    new_blue[w2] |= bit2
                else:
                    new_blue[w2] &= ~bit2
                    new_red[w2] |= bit2

                v_before = count_violations(red_set, blue_set)
                v_after = count_violations(new_red, new_blue)
                return v_before - v_after  # positive = improvement

            # Phase 1: Conflict-driven single-flip elimination
            max_iter = 30000
            for it in range(max_iter):
                if time.time() > start_time + time_limit:
                    break
                viol_quads = get_violating_quads(red_set)
                viol_quints = get_violating_quints(blue_set)
                if len(viol_quads) == 0 and len(viol_quints) == 0:
                    return colors.tolist()

                candidate_set = set()
                for q in viol_quads:
                    for t in quad_triples[q]:
                        candidate_set.add(int(t))
                for q in viol_quints:
                    for t in quint_triples[q]:
                        candidate_set.add(int(t))

                if not candidate_set:
                    break

                best_t = -1
                best_s = 0
                for t in candidate_set:
                    s = flip_score_single(t, red_set, blue_set)
                    if s > best_s:
                        best_s = s
                        best_t = t

                if best_t == -1:
                    # No single flip helps; try a random pair flip
                    t1 = int(rng.randint(0, M))
                    t2 = int(rng.randint(0, M))
                    if t1 != t2:
                        apply_flip(t1, red_set, blue_set)
                        apply_flip(t2, red_set, blue_set)
                    continue

                apply_flip(best_t, red_set, blue_set)

            # Phase 2: Pairwise exchange (2-flip) local search
            T = 5.0
            cooling = 0.9995
            min_T = 0.05
            no_improve_count = 0
            best_local_viol = count_violations(red_set, blue_set)

            while time.time() < start_time + time_limit:
                # Try a random pair of triples to flip
                t1 = int(rng.randint(0, M))
                t2 = int(rng.randint(0, M))
                if t1 == t2:
                    t2 = (t2 + 1) % M

                s = flip_pair_score(t1, t2, red_set, blue_set)

                # Apply the pair flip
                apply_flip(t1, red_set, blue_set)
                apply_flip(t2, red_set, blue_set)

                v_now = count_violations(red_set, blue_set)
                if s < 0 or (T > min_T and rng.random() < np.exp(-s / T)):
                    # Accept
                    if v_now < best_local_viol:
                        best_local_viol = v_now
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    if best_local_viol == 0:
                        return colors.tolist()
                else:
                    # Reject: undo
                    apply_flip(t1, red_set, blue_set)
                    apply_flip(t2, red_set, blue_set)
                    no_improve_count += 1

                if no_improve_count > 5000:
                    # Restart with perturbation
                    for _ in range(20):
                        t = int(rng.randint(0, M))
                        apply_flip(t, red_set, blue_set)
                    no_improve_count = 0
                    T = max(T, 2.0)

                T *= cooling

            v_final = count_violations(red_set, blue_set)
            if v_final < best_violations:
                best_violations = v_final
                best_result = colors.tolist()

            if best_violations == 0:
                return best_result

        return best_result

    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    # Try increasing n with multiple restarts
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        per_n_budget = min(remaining - 2.0, 30.0)
        if per_n_budget <= 0:
            break
        n_restarts = 3
        per_restart = per_n_budget / n_restarts
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            res = search_for_n(n, t_limit, seed=n * 1000 + r * 17 + 3)
            if res is not None:
                found = res
                break
        if found is not None:
            best_n = n
            best_colors = found

    return {'n': best_n, 'colors': best_colors}
