import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin

    def search_for_n(n, time_limit, seed=0, spectral_mode=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Incidence: which 4-sets and 5-sets contain each triple
        quad_triples = []   # list of 4 triple-indices per 4-set
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []  # list of 10 triple-indices per 5-set
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)

        # For each triple, the 4-sets and 5-sets it belongs to
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

        # Bit masks
        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        rng = np.random.RandomState(seed)

        # --- Spectral Initialization ---
        # We create a geometric embedding of vertices.
        # Mode 0: Circle embedding
        # Mode 1: Grid-like embedding
        # Mode 2: Random embedding
        # The idea is to use the "position" of the triple's vertices to decide color,
        # promoting spatially distributed colors.
        
        if spectral_mode == 0:
            # Circle: vertices on a unit circle
            angles = 2 * np.pi * np.arange(n) / n
            # Project triple center to 1D (angle sum mod pi)
            # If sum of angles is close to 0, red; else blue? 
            # Better: Use the sign of the sine of the average angle or similar.
            # Let's use a simpler spectral proxy: 
            # Color based on (sum of vertex indices) mod something is already tried.
            # Let's try: Color based on the projection of the triple centroid onto a random axis.
            # This is effectively a random linear cut, which is a standard spectral heuristic.
            axis = rng.randn(n)
            vertex_scores = axis
        elif spectral_mode == 1:
            # 2D grid projection
            side = int(np.ceil(np.sqrt(n)))
            xs = (np.arange(n) % side) / side
            ys = (np.arange(n) // side) / side
            # Project onto diagonal
            vertex_scores = xs + ys
        else:
            # Random spectral-like (just random, but normalized)
            vertex_scores = rng.randn(n)

        # For each triple, compute a score based on the sum of its vertices' scores
        # and threshold it to create a balanced coloring.
        triple_scores = np.zeros(M)
        for i, (a, b, c) in enumerate(triples):
            triple_scores[i] = vertex_scores[a] + vertex_scores[b] + vertex_scores[c]
        
        # Threshold at median to ensure roughly 50/50 split
        median_score = np.median(triple_scores)
        colors = np.zeros(M, dtype=np.int8)
        colors[triple_scores < median_score] = 0
        colors[triple_scores >= median_score] = 1

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

        def flip_score(t, red_set, blue_set):
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

        # Phase 1: Greedy violation elimination
        max_phase1_iter = 50000
        for it in range(max_phase1_iter):
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
                s = flip_score(t, red_set, blue_set)
                if s > best_s:
                    best_s = s
                    best_t = t

            if best_t == -1:
                t = int(rng.randint(0, M))
                apply_flip(t, red_set, blue_set)
                continue

            apply_flip(best_t, red_set, blue_set)

        # Phase 2: Simulated annealing
        T = 2.0
        cooling = 0.9999
        min_T = 0.01
        while time.time() < start_time + time_limit:
            t = int(rng.randint(0, M))
            s = flip_score(t, red_set, blue_set)
            apply_flip(t, red_set, blue_set)
            if s < 0 or (T > min_T and rng.random() < np.exp(-s / T)):
                pass
            else:
                apply_flip(t, red_set, blue_set)
            if count_violations(red_set, blue_set) == 0:
                return colors.tolist()
            T *= cooling

        return None

    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    # Try increasing n with multiple restarts and spectral modes
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        per_n_budget = min(remaining - 2.0, 35.0)
        if per_n_budget <= 0:
            break
        
        # Use 3 spectral modes for diversity
        n_restarts = 3
        per_restart = per_n_budget / n_restarts
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            # Vary seed and spectral mode
            seed = n * 1000 + r * 13 + 7
            mode = r % 3
            res = search_for_n(n, t_limit, seed=seed, spectral_mode=mode)
            if res is not None:
                found = res
                break
        
        if found is not None:
            best_n = n
            best_colors = found

    return {'n': best_n, 'colors': best_colors}
