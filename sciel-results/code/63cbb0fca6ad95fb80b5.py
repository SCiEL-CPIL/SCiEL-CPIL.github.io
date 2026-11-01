"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # --- Build W4 vectors (weight 4, entries in {-1,0,1}, norm^2 = 4) ---
    # --- Build W1 vectors (weight 1, entries in {-2,0,2}, norm^2 = 4) ---
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N4 = len(v4_candidates)
    if N4 == 0:
        return vectors_v1

    # --- Structural decomposition ---
    # Kissing constraint for two W4 vectors: |dot| <= 2 (norm^2 = 4).
    # Two W4 vectors with DIFFERENT 4-position supports share at most 3
    # positions, so |dot| <= 3 < 4 -> ALWAYS compatible.  Hence any set of
    # supports can coexist.  Conflicts exist only BETWEEN sign variants that
    # share the SAME support (they share 4 positions; |dot| reaches 4 when they
    # differ in exactly one sign).  The problem therefore decomposes into a
    # per-support 16-node independent-set problem, plus all W1 vectors which are
    # compatible with everything (|dot| <= 1 <= 2).

    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        support_to_indices.setdefault(support, []).append(i)

    supports = list(support_to_indices.keys())
    n_supports = len(supports)

    def max_independent_set_16(cm):
        """Exact maximum independent set on a 16-vertex graph (cm: 16x16 bool).
        Returns (size, list_of_indices)."""
        n = 16
        adj = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if i != j and cm[i, j]:
                    m |= (1 << j)
            adj[i] = m
        deg = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])

        best = {'size': 0, 'set': []}

        def rec(available, chosen, size):
            if available == 0:
                if size > best['size']:
                    best['size'] = size
                    best['set'] = list(chosen)
                return
            if size + bin(available).count('1') <= best['size']:
                return
            verts = [v for v in order if (available >> v) & 1]
            v = max(verts, key=lambda x: bin(adj[x] & available).count('1'))
            vbit = 1 << v
            chosen.append(v)
            rec(available & ~(vbit | adj[v]), chosen, size + 1)
            chosen.pop()
            rec(available & ~vbit, chosen, size)

        rec((1 << n) - 1, [], 0)
        return best['size'], best['set']

    final_w4_indices = []
    for si, support in enumerate(supports):
        idxs = support_to_indices[support]
        sub = vectors_v4[idxs]
        dots = sub @ sub.T
        cm = dots > 2
        np.fill_diagonal(cm, False)
        size, chosen_local = max_independent_set_16(cm)
        final_w4_indices.extend(idxs[c] for c in chosen_local)

    result_v4 = vectors_v4[final_w4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    # --- Safety verification: enforce kissing property, drop violators ---
    n = len(final_vectors)
    if n > 1 and time.time() < deadline - 5:
        V = final_vectors.astype(np.float64)
        norms2 = np.einsum('ij,ij->i', V, V)
        max_norm2 = norms2.max()
        dots = V @ V.T
        dist2 = norms2[:, None] + norms2[None, :] - 2.0 * dots
        np.fill_diagonal(dist2, np.inf)
        ok_pairs = dist2 >= (max_norm2 - 1e-9)

        in_set = np.ones(n, dtype=bool)
        # Greedily remove vertices causing the most violations.
        while time.time() < deadline - 5:
            viol = (~ok_pairs) & in_set[:, None] & in_set[None, :]
            counts = viol.sum(axis=1)
            counts[~in_set] = -1
            worst = int(np.argmax(counts))
            if counts[worst] <= 0:
                break
            in_set[worst] = False

        if in_set.sum() < n:
            final_vectors = final_vectors[in_set]

    return final_vectors

# EVOLVE-BLOCK-END
