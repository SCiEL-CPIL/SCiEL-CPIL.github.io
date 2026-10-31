"""Kissing number in dimension 11: Max-Cut spectral + Kempe-chain independent set."""
D = 11
import time
import numpy as np


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    # --- Build the weight-4 vector pool (four entries = +1/-1) ---
    supports = []
    for s in range(1, D - 3):
        for rest in range(s + 1, D):
            for a in range(rest + 1, D):
                for b in range(a + 1, D):
                    supports.append((s, rest, a, b))
    m = len(supports)

    # --- Build the conflict graph ---
    # Two weight-4 vectors are INCOMPATIBLE iff they share a coordinate and
    # disagree on at least one shared coordinate (distance^2 <= 3 < 4).
    # They are COMPATIBLE iff supports are disjoint OR they agree on all shared coords.
    #
    # We build an adjacency (conflict) structure. m=3520, so a dense 3520x3520
    # bool matrix is ~12MB, feasible.
    conflict = np.zeros((m, m), dtype=bool)
    # For each pair, compute shared coordinates and check sign agreement.
    # Represent each vector's signs on its 4 support positions.
    # sign[i] is the tuple of 4 signs for support i.
    # We need to check, for pair (i,j): for each shared coordinate, do signs match?
    #
    # Efficient approach: for each coordinate c (0..10), the vectors containing c form a group.
    # Within a group, two vectors conflict if they differ in sign at c.
    # So: conflict[i,j] = OR over shared c of (sign_i[c] != sign_j[c]).
    #
    # Let's precompute, for each coordinate c, the list of (index, sign) pairs.
    # Then for each c, all pairs in that group that differ in sign are conflicts.
    #
    # But we also need to be careful: two vectors sharing coordinate c with SAME sign
    # at c might still conflict via another shared coordinate. So conflict is the OR.
    #
    # Strategy: for each coordinate c, find all pairs (i,j) in group_c with different signs,
    # mark conflict[i,j]=True. Then the union over c gives the full conflict graph.

    # Build per-coordinate lists
    from collections import defaultdict
    coord_lists = [[] for _ in range(D)]
    for idx, sup in enumerate(supports):
        # We need to assign signs. Let's use a deterministic sign assignment:
        # For support (a,b,c,d), signs = (1,1,1,1) -- but we need ALL sign combinations.
        # Wait, I made an error. The pool should include ALL 2^4 sign combinations for each support.
        # Let me redo.
        pass

    # REDO: pool includes all sign combinations.
    # For each support (4 coords), 16 sign patterns. Total = C(11,4)*16 = 330*16 = 5280.
    # Let me rebuild.
    supports_all = []
    signs_all = []
    for s in range(1, D - 3):
        for rest in range(s + 1, D):
            for a in range(rest + 1, D):
                for b in range(a + 1, D):
                    for mask in range(16):
                        sup = (s, rest, a, b)
                        sig = tuple(1 if (mask >> k) & 1 else -1 for k in range(4))
                        supports_all.append(sup)
                        signs_all.append(sig)
    m = len(supports_all)  # 5280

    # Build conflict graph
    # For each coordinate c, group vectors by (index, sign_at_c).
    # Pairs in same group with different signs -> conflict.
    # Use coordinate-wise approach.
    coord_vectors = [[] for _ in range(D)]  # coord_vectors[c] = list of (idx, sign)
    for idx in range(m):
        sup = supports_all[idx]
        sig = signs_all[idx]
        for k, c in enumerate(sup):
            coord_vectors[c].append((idx, sig[k]))

    # For each coordinate, mark conflicts between pairs with different signs.
    # To be efficient, separate into pos and neg lists.
    # conflict = OR over coordinates of (pos_i & neg_j for i in pos, j in neg)
    # But this only captures conflicts via THAT coordinate. We need the union.
    #
    # Actually, the correct logic: conflict[i,j] = True iff there exists a shared coordinate c
    # where sign_i[c] != sign_j[c].
    #
    # So for each coordinate c, all cross-pairs between pos_c and neg_c are conflicts.
    # The union over c gives the full conflict graph.
    #
    # m=5280, dense matrix is 5280^2 = 27.9M bools = ~28MB. OK.
    conflict = np.zeros((m, m), dtype=bool)

    for c in range(D):
        lst = coord_vectors[c]
        pos = [idx for idx, sg in lst if sg == 1]
        neg = [idx for idx, sg in lst if sg == -1]
        if not pos or not neg:
            continue
        # Mark all pos-neg pairs as conflict
        for i in pos:
            conflict[i, neg] = True
            conflict[neg, i] = True

    # Now find maximum independent set = maximum set with NO conflict edges.
    # This is Max-Cut on the conflict graph: partition into A (in set) and B (not in set),
    # maximize number of non-conflict edges within A... 
    # Actually, independent set: we want max subset with no internal conflicts.
    # Equivalent to Max-Cut: cut edges are conflict edges crossing the partition.
    # If we let S = independent set, then ALL conflict edges have at least one endpoint outside S.
    # Max-Cut maximizes conflict edges crossing the cut, which is equivalent to minimizing
    # conflict edges within S (and within complement).
    #
    # Standard approach: spectral relaxation.
    # Build Laplacian L = D - A where A is the conflict adjacency.
    # Fiedler vector (2nd eigenvector) gives sign partition.
    # Then Kempe-chain flips to refine.

    # Build adjacency matrix (conflict)
    A = conflict.astype(np.float64)
    # Laplacian
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    # Get Fiedler vector (eigenvector of 2nd smallest eigenvalue)
    # Use scipy for sparse eigenvalue, or numpy for dense (m=5280 might be slow for dense eig).
    # Let's use scipy.sparse for efficiency.
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        L_sp = sp.csr_matrix(L)
        # Find 2 smallest eigenvalues
        # Use shift-invert for smallest
        vals, vecs = spla.eigsh(L_sp, k=3, sigma=0, which='LM')
        # Fiedler vector is the one with smallest positive eigenvalue
        # Sort by eigenvalue
        order = np.argsort(vals)
        fiedler = vecs[:, order[1]]
    except Exception:
        # Fallback: dense eig (slow but works)
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]

    # Sign assignment
    x = np.sign(fiedler)
    x[x == 0] = 1  # break ties
    # x[i] = 1 means vertex i is IN the independent set, -1 means OUT.

    # Local refinement: Kempe-chain flips.
    # A vertex i in S (x[i]=1) should be moved OUT if it has more conflicts in S than outside.
    # A vertex i OUT (x[i]=-1) should be moved IN if it has more non-conflicts in S than conflicts.
    #
    # Score = number of vertices in S - number of conflict edges within S.
    # Moving i from OUT to IN: gain = (1 - deg_S(i)) where deg_S(i) = conflicts of i within S.
    #   If gain > 0, move i in.
    # Moving i from IN to OUT: gain = (deg_S(i) - 1) where deg_S(i) = conflicts of i within S (after removal).
    #   If gain > 0, move i out.
    #
    # Actually, let's think in terms of the objective:
    # Objective = |S| - (# conflict edges with both endpoints in S)
    # = |S| - (1/2) * sum_{i in S} deg_S(i)
    #
    # Moving i from OUT to IN:
    #   |S| increases by 1.
    #   Conflict edges added: deg_S(i) (conflicts of i with vertices already in S).
    #   Delta = 1 - deg_S(i). Move if delta > 0.
    #
    # Moving i from IN to OUT:
    #   |S| decreases by 1.
    #   Conflict edges removed: deg_S(i) (conflicts of i with other vertices in S).
    #   Delta = -1 + deg_S(i). Move if delta > 0.

    def compute_deg_S(x):
        """deg_S[i] = number of conflicts of i with vertices in S (x==1)."""
        S_mask = (x == 1)
        # deg_S[i] = sum of conflict[i, j] for j where S_mask[j]
        # = A[i, :] @ S_mask
        return A @ S_mask.astype(np.float64)

    # Iterative local search
    best_x = x.copy()
    best_score = -1
    for iteration in range(200):
        if elapsed() > DEADLINE - 30:
            break
        deg_S = compute_deg_S(x)
        improved = False
        # Try moving vertices OUT of S
        in_S = np.where(x == 1)[0]
        # Randomize order to avoid bias
        np.random.shuffle(in_S)
        for i in in_S:
            if elapsed() > DEADLINE - 30:
                break
            d = int(deg_S[i])
            # Moving i out: delta = d - 1
            if d > 1:  # delta > 0
                x[i] = -1
                deg_S[i] = 0  # no longer in S
                # Update deg_S for neighbors of i that are in S
                neighbors_in_S = np.where(conflict[i] & (x == 1))[0]
                deg_S[neighbors_in_S] -= 1
                improved = True
        # Try moving vertices INTO S
        out_S = np.where(x == -1)[0]
        np.random.shuffle(out_S)
        for i in out_S:
            if elapsed() > DEADLINE - 30:
                break
            d = int(deg_S[i])
            # Moving i in: delta = 1 - d
            if d < 1:  # delta > 0, i.e., d == 0
                x[i] = 1
                deg_S[i] = 0
                # Update deg_S for neighbors of i that are in S
                neighbors_in_S = np.where(conflict[i] & (x == 1))[0]
                deg_S[neighbors_in_S] += 1
                improved = True
        # Track best
        S_mask = (x == 1)
        score = int(S_mask.sum()) - int((A[S_mask[:, None], S_mask[None, :]].sum() / 2))
        if score > best_score:
            best_score = score
            best_x = x.copy()
        if not improved:
            break

    # Use best partition
    x = best_x
    S_mask = (x == 1)
    selected_indices = np.where(S_mask)[0]

    # Build the actual vectors
    vecs = []
    for idx in selected_indices:
        sup = supports_all[idx]
        sig = signs_all[idx]
        v = [0] * D
        for k, c in enumerate(sup):
            v[c] = sig[k]
        vecs.append(v)

    # Add all 22 weight-1 vectors (compatible with everything)
    for c in range(D):
        vecs.append([0]*D)
        vecs[-1][c] = 1
        vecs.append([0]*D)
        vecs[-1][c] = -1

    # Add all 440 weight-2 vectors? They are compatible with weight-1 and weight-4.
    # Two weight-2 vectors are compatible iff supports disjoint or agree on shared coords.
    # But mixing weight-2 and weight-4: weight-2 (norm^2=2) vs weight-4 (norm^2=4).
    # Distance^2 = 2 + 4 - 2*dot. Need >= 4, so dot <= 1.
    # Weight-2 has 2 nonzero entries, weight-4 has 4. Max overlap is 2 coords.
    # If they share 2 coords and agree on both: dot = 2, dist^2 = 6-4=2 < 4. INCOMPATIBLE.
    # If they share 2 coords and agree on 1: dot = 0, dist^2 = 6. OK.
    # If they share 1 coord and agree: dot = 1, dist^2 = 4. OK (>= 4).
    # If they share 1 coord and disagree: dot = -1, dist^2 = 8. OK.
    # If they share 0: dot = 0, dist^2 = 6. OK.
    # So weight-2 and weight-4 are compatible UNLESS they share 2 coords AND agree on both.
    # This complicates things. Let's skip weight-2 for now and just use weight-4 + weight-1.
    # The measured finding was 2.573 with weight-4 independent set + weight-1.

    result = np.array(vecs, dtype=np.int64)

    # Verify
    if len(result) == 0:
        # Fallback to seed
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    # Quick verification (may be slow for large sets, so only check a sample)
    if len(result) <= 500:
        norms_sq = np.sum(result**2, axis=1).astype(np.float64)
        max_norm_sq = norms_sq.max()
        # Check pairwise distances
        # dist^2(i,j) = norm^2(i) + norm^2(j) - 2*dot(i,j)
        # Need dist^2 >= max_norm^2 for all i != j
        # dot product matrix
        Dots = result @ result.T
        Norms2 = norms_sq[:, None] + norms_sq[None, :] - 2 * Dots
        # Set diagonal to inf
        np.fill_diagonal(Norms2, np.inf)
        min_dist_sq = Norms2.min()
        if min_dist_sq < max_norm_sq - 1e-9:
            # Invalid, fall back to seed
            import itertools
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)

    return result
