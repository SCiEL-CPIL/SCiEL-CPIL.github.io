import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # 2. Build conflict graph: conflict if intersection size is exactly 3
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        mi = masks[i]
        for j in range(i + 1, N_S):
            if (mi & masks[j]).bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)

    adj_list = [list(s) for s in adj]

    # Vectorized adjacency (boolean matrix) for fast blocked-set / energy ops
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        if adj_list[i]:
            adj_bool[i, adj_list[i]] = True

    # 3. Search machinery -------------------------------------------------

    def greedy_random(rng):
        """Randomized greedy independent set: random permutation, skip conflicts."""
        order = rng.permutation(N_S)
        selected = np.zeros(N_S, dtype=bool)
        for v in order:
            if not np.any(selected & adj_bool[v]):
                selected[v] = True
        return set(np.where(selected)[0])

    def local_search_fast(init_set, rng, max_iter=300):
        """Vectorized add / 1-swap / 2-swap local search. Returns best set seen."""
        current = set(init_set)
        best = set(current)
        best_size = len(current)
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            in_set = np.zeros(N_S, dtype=bool)
            for v in current:
                in_set[v] = True
            blocked = np.any(adj_bool[in_set], axis=0) if current else np.zeros(N_S, dtype=bool)
            cand = np.where(~(in_set | blocked))[0]
            if len(cand) > 0:
                current.add(int(cand[rng.randint(len(cand))]))
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                continue
            # Stuck: try 2-swap (remove 2, add 1-2)
            if len(current) >= 2:
                c_list = list(current)
                i1 = rng.randint(len(c_list))
                i2 = rng.randint(len(c_list))
                if i2 == i1:
                    i2 = (i1 + 1) % len(c_list)
                r1, r2 = c_list[i1], c_list[i2]
                new_set = current - {r1, r2}
                in2 = np.zeros(N_S, dtype=bool)
                for v in new_set:
                    in2[v] = True
                bl2 = np.any(adj_bool[in2], axis=0) if new_set else np.zeros(N_S, dtype=bool)
                cand2 = np.where(~(in2 | bl2))[0]
                if len(cand2) >= 2:
                    a1 = int(cand2[rng.randint(len(cand2))])
                    a2 = int(cand2[rng.randint(len(cand2))])
                    if a2 == a1:
                        a2 = int(cand2[(np.where(cand2 == a1)[0][0] + 1) % len(cand2)])
                    new_set.add(a1)
                    new_set.add(a2)
                    current = new_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                elif len(cand2) == 1:
                    new_set.add(int(cand2[0]))
                    current = new_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                # else: revert
            elif current:
                # 1-swap
                v_remove = next(iter(current))
                current.discard(v_remove)
                in2 = np.zeros(N_S, dtype=bool)
                for v in current:
                    in2[v] = True
                bl2 = np.any(adj_bool[in2], axis=0) if current else np.zeros(N_S, dtype=bool)
                cand2 = np.where(~(in2 | bl2))[0]
                if len(cand2) > 0:
                    current.add(int(cand2[rng.randint(len(cand2))]))
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                else:
                    current.add(v_remove)
        return best

    # 4. Run search -------------------------------------------------------
    rng = np.random.RandomState(42)
    best_size = 0
    best_set = set()

    # Phase A: randomized greedy + local search restarts
    while time.time() < deadline - 20.0:
        init_set = greedy_random(rng)
        final_set = local_search_fast(init_set, rng, max_iter=150)
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)

    # Phase B: polish best set with more iterations
    if best_size > 0:
        polished = local_search_fast(best_set, rng, max_iter=500)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = set(polished)

    # 5. Construct vectors: all sign patterns on each selected support.
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    # 6. E8 lattice fallback (240 vectors)
    def e8_fallback():
        # E8 lattice vectors: weight 2 with entries ±1, or weight 4 with entries ±1/2
        # Scale by 2 to make integer: weight-4 vectors become integer with ±1
        # Actually, use the standard E8 root lattice: 240 vectors of norm 2
        # In 8D, E8 roots: 112 of type (±1,±1,0,...) and 128 of type (±1/2,...)
        # Scale by 2: 112 vectors with two ±2 entries, 128 vectors with eight ±1 entries
        # Pad to 11D with zeros
        vecs = []
        # Type 1: two nonzero entries ±1 (after scaling by 2, these are ±2)
        # Actually let's use the unscaled E8 and just pad
        # E8 roots: 240 vectors in R^8 with norm 2
        # Type A: 112 vectors: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(8, dtype=np.int8)
                        v[i] = si
                        v[j] = sj
                        vecs.append(v)
        # Type B: 128 vectors: all entries ±1/2 with even number of minus signs
        # Scale by 2: all entries ±1 with even number of minus signs
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                vecs.append(np.array(signs, dtype=np.int8))
        # Pad to 11D
        result = np.zeros((len(vecs), 11), dtype=np.int8)
        for k, v in enumerate(vecs):
            result[k, :8] = v
        return result

    e8_vecs = e8_fallback()

    if not vectors:
        return e8_vecs

    result = np.array(vectors, dtype=np.int8)
    # Use whichever is larger
    if len(result) < len(e8_vecs):
        return e8_vecs
    return result
