import random
import time
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 320.0

    n_points = N * N
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)

    # Precompute conflict bitsets: conflict[i] is a bitset where bit j is set
    # if adding point j to a set containing point i would create an isosceles triangle.
    # Two points i, j conflict if there exists a third point k such that {i, j, k}
    # forms an isosceles triangle.
    #
    # For a set S, a new point p is valid iff for all a in S, the pair (p, a) doesn't
    # form an isosceles triangle with any other point in S.
    # Equivalently: for all a in S, no other point b in S is such that (p, a, b) is isosceles.
    #
    # The conflict graph approach: edge (i, j) exists if i and j cannot both be in a
    # no-isosceles set. This happens when there exists k such that {i,j,k} is isosceles.
    # But actually the constraint is stronger: we need that for any triple in the set,
    # none is isosceles. So i and j are compatible if no k exists making {i,j,k} isosceles,
    # AND also we need to check that adding j doesn't create an isosceles with any existing pair.
    #
    # Actually, the correct formulation: a set S is valid iff for all distinct a,b,c in S,
    # {a,b,c} is not isosceles. So two points i,j are "conflicting" if there exists k such
    # that {i,j,k} is isosceles. Then a valid set is an independent set in this conflict graph.
    #
    # Wait, that's not quite right either. If i and j don't conflict directly (no k makes {i,j,k}
    # isosceles), but i and k are in the set and j and k are in the set, we need {i,j,k} not
    # isosceles. So the conflict is really on triples.
    #
    # Let me reconsider. The constraint is: for all distinct a, b, c in S, the triangle (a,b,c)
    # is not isosceles. This means for any pair (a,b) in S, and any c in S\{a,b}, the distances
    # da, db, dc from c must all be distinct.
    #
    # For the greedy approach with bitsets: when we want to add point p to set S, we need to
    # check that for every a in S, the distance d(p,a) is different from d(p,b) for all b in S,
    # and also different from d(a,b) for all pairs.
    #
    # A simpler bitset approach: for each point p, precompute for each existing point a,
    # the set of points that would conflict with p given that a is in the set. But this is
    # complex.
    #
    # Let me use a different approach: precompute for each pair of points (i, j), the set of
    # points k that would make {i, j, k} isosceles. Then when checking if p can be added to S,
    # for each a in S, we need p not to be in the "forbidden" set for pair (p, a) relative to S.
    #
    # Actually, let me just use the direct approach but with NumPy vectorization for the inner check.
    # The key insight from the knowledge: use vectorized distance checks.

    # Precompute all pairwise squared distances
    # coords has shape (n_points, 2)
    # dist_sq[i, j] = squared distance between point i and point j
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)  # shape (n_points, n_points)

    # For greedy construction, when adding point p to set S (represented as indices),
    # we need: for all a, b in S with a != b, and p not in S:
    #   d(p,a) != d(p,b) and d(p,a) != d(a,b) and d(p,b) != d(a,b)
    #
    # Vectorized check: given p and S (array of indices):
    #   d_pa = dist_sq[p, S]  # distances from p to each point in S
    #   d_S = dist_sq[np.ix_(S, S)]  # pairwise distances within S
    #   Check: all d_pa values are distinct (no two equal)
    #   Check: for each a in S, d_pa[a] != d_S[a, b] for all b != a
    #
    # Let me implement this with a helper function.

    def can_add(p_idx, selected_indices):
        """Check if point p_idx can be added to the set of selected_indices."""
        k = len(selected_indices)
        if k == 0:
            return True
        if k == 1:
            return True  # Two points never form a triangle

        sel = selected_indices
        # Distances from p to each selected point
        d_p = dist_sq[p_idx, sel]  # shape (k,)

        # Check all d_p values are distinct
        # Use sorting for O(k log k) uniqueness check
        if k > 2:
            sorted_dp = np.sort(d_p)
            if np.any(np.diff(sorted_dp) == 0):
                return False

        # Check d_p[i] != d_S[i, j] for all i, j with i != j
        # d_S[i, j] = dist_sq[sel[i], sel[j]]
        d_S = dist_sq[np.ix_(sel, sel)]  # shape (k, k)

        # For each i, d_p[i] should not equal d_S[i, j] for any j != i
        # And d_p[i] should not equal d_S[j, i] for any j != i (but d_S is symmetric)
        # Actually we need: for the triangle (p, sel[i], sel[j]), the three distances are:
        #   d(p, sel[i]) = d_p[i]
        #   d(p, sel[j]) = d_p[j]
        #   d(sel[i], sel[j]) = d_S[i, j]
        # These three must all be distinct.
        #
        # We already checked d_p[i] != d_p[j] for all i != j (all d_p distinct).
        # Now we need d_p[i] != d_S[i, j] and d_p[j] != d_S[i, j] for all i != j.
        #
        # For a fixed i, j (i != j): need d_p[i] != d_S[i,j] and d_p[j] != d_S[i,j]
        # Equivalently: d_S[i,j] not in {d_p[i], d_p[j]}
        #
        # Vectorized: for all pairs (i,j) with i < j, check d_S[i,j] != d_p[i] and d_S[i,j] != d_p[j]

        if k > 2:
            # Create upper triangle mask
            i_idx, j_idx = np.triu_indices(k, k=1)
            d_S_upper = d_S[i_idx, j_idx]  # shape (num_pairs,)
            d_p_i = d_p[i_idx]
            d_p_j = d_p[j_idx]

            if np.any(d_S_upper == d_p_i) or np.any(d_S_upper == d_p_j):
                return False

        return True

    best_set = []
    best_len = 0

    num_restarts = 200

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        random.seed(12345 + r)
        order = list(range(n_points))
        random.shuffle(order)

        selected = []

        for p_idx in order:
            if time.time() > deadline:
                break

            if can_add(p_idx, selected):
                selected.append(p_idx)

        if len(selected) > best_len:
            best_len = len(selected)
            best_set = [points[i] for i in selected]

    return best_set
# EVOLVE-BLOCK-END
