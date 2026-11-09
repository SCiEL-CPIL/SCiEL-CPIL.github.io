# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)  # C(16,4) = 1820

    # Precompute: for each pair (i,j), their intersection
    # A sunflower triple (a,b,c) exists when:
    #   |a∩b| = |a∩c| = |b∩c|  AND  a∩b∩c == a∩b (i.e., the pairwise intersection is the same set)
    # Actually the condition is: the three pairwise intersections are all equal.
    # Let I_ab = a∩b, I_ac = a∩c, I_bc = b∩c.
    # Sunflower iff I_ab == I_ac == I_bc.

    # Build conflict graph: edge (i,j) means i and j are "conflict partners"
    # But actually we need: three sets form a sunflower.
    # The constraint is: no three distinct sets all have equal pairwise intersections.
    # This is a 3-uniform hypergraph constraint, not a graph constraint.
    #
    # Alternative approach: For each set s, and each possible intersection value t (a subset of s),
    # if we already have two sets a,b in F with a∩s = b∩s = t, then adding s creates a sunflower
    # (since a∩b must also equal t for it to be a sunflower... wait, no).
    #
    # Let me re-read: "No three distinct sets may have all three pairwise intersections equal"
    # So for sets A, B, C: A∩B == A∩C == B∩C.
    #
    # This is equivalent to: there exists a set T such that A∩B = A∩C = B∩C = T.
    # Note T can be empty.
    #
    # Key insight: If A∩B = A∩C = T, then B∩C must also equal T.
    # A∩B = T means A and B share exactly the elements in T.
    # A∩C = T means A and C share exactly the elements in T.
    # For B∩C = T: B and C must share exactly T.
    #
    # So the conflict is a 3-clique in a specific sense. Let's think of it differently.
    #
    # For a fixed set A and a fixed subset T ⊆ A:
    #   The set of all B such that A∩B = T forms a group.
    #   Within that group, any two sets B1, B2 that also satisfy B1∩B2 = T form a sunflower with A.
    #
    # So the constraint is: for any A, T, and any two distinct B1, B2 in the group {B : A∩B = T},
    # we must NOT have B1∩B2 = T.
    #
    # This is complex. Let me use a different structural approach:
    # Maximum independent set in a 3-uniform hypergraph is hard.
    #
    # Instead, let's use a smart greedy with better ordering + local search.
    # But the directive says to make a STRUCTURAL change.
    #
    # New structural approach: Model as maximum independent set on a derived graph.
    # Two sets "conflict" if they can be part of a sunflower with some third set.
    # Actually, let's just precompute all sunflower triples and use a different strategy.

    # Precompute all sunflower triples
    # For each triple (i,j,k), check if it's a sunflower
    # C(1820, 3) is too large (~1e9). Need a smarter way.

    # Better: For each pair (i,j) with intersection T = set_i & set_j,
    # find all k such that set_k & set_i == T and set_k & set_j == T.
    # These k form sunflowers with (i,j).

    # Build for each pair (i,j): the list of k that form a sunflower
    # Then the constraint is: for each sunflower triple {i,j,k}, not all three can be in F.
    # This is a 3-SAT / hypergraph independent set problem.

    # Structural change: Use a different representation.
    # For each element t (subset of [16]), define a "type" based on which sets contain it.
    #
    # Actually, let me try a completely different approach:
    # Use the fact that k=4 is small. For each 4-set, its "signature" relative to others
    # is determined by its 6 pairwise intersections with other sets.
    #
    # Let me try: maximum independent set via branch and bound on the conflict hypergraph.
    # But 3-uniform hypergraph MIS is NP-hard and hard to bound.
    #
    # Alternative structural change: Use integer programming / LP relaxation with rounding.
    # Or: Use a different greedy strategy based on "potential" ordering.
    #
    # Let me try: Precompute for each set s, the number of "near-conflicts" it has.
    # A near-conflict of s is a set t such that there exists u with s∩t = s∩u = t∩u.
    # Order sets by ascending near-conflict count, then greedy.
    #
    # Actually, the most promising structural change: use a local search / hill climbing
    # starting from a good greedy solution, trying to swap in better sets.

    # First, let's do a fast greedy to get a good starting point
    rng = random.Random(42)

    # Precompute intersections for speed
    # inter[i][j] = bitmask of intersection of all_sets[i] and all_sets[j]
    # This is 1820x1820 = ~3.3M entries, each a small int. Feasible.

    # Actually, let's precompute for each set i, and each possible 4-subset mask m,
    # what is i & m. We can use a lookup table.
    # There are 2^16 = 65536 possible masks, but only 1820 are 4-sets.
    # Let's index all_sets by their position and precompute inter[i][j].

    # For memory efficiency, store as a list of dicts or just compute on the fly.
    # 1820 * 1820 = 3.3M bitwise ANDs is fast.

    # Let's precompute the intersection matrix
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        si = all_sets[i]
        for j in range(i + 1, total):
            v = si & all_sets[j]
            inter[i][j] = v
            inter[j][i] = v

    # Now, for the greedy: when adding set m to family F,
    # we need to check: for each existing pair (a, b) in F,
    # if inter[a][b] == inter[a][new] == inter[b][new], conflict.
    #
    # More efficiently: for each existing set a in F,
    #   t = inter[a][new]
    #   Check if there's another set b in F with inter[a][b] == t AND inter[b][new] == t
    #
    # Precompute for each a and each possible t: list of b in F where inter[a][b] == t
    # This is the same as the current program's inter_maps.

    # Let me try a different structural approach: 
    # Use the "potential" of a set = number of sunflower triples it participates in.
    # Prefer sets with low potential.

    # Compute for each set i, how many sunflower triples it's in.
    # A sunflower triple containing i: pairs (j,k) with j≠k≠i, j<k,
    # such that inter[i][j] == inter[i][k] == inter[j][k].
    #
    # For each i, group other sets j by t = inter[i][j].
    # Within each group G_t, count pairs (j,k) with inter[j][k] == t.
    # Sum over all t to get total sunflower triples containing i.

    # This is O(total^2 * avg_group_size) which might be slow.
    # Let's approximate: for each i, for each j, t = inter[i][j].
    # We need to count, among sets with the same t, how many pairs have inter == t.
    #
    # This is getting complex. Let me just use a smarter greedy + local search.

    # Structural change: Use a "simulated annealing" style local search.
    # Start with greedy, then try to replace one set with another to increase size
    # or maintain size while improving "quality" for future additions.

    # Actually, let me try the most impactful structural change:
    # Use a maximum independent set solver on a 2-SAT-like reduction.
    #
    # Key observation: The sunflower condition A∩B = A∩C = B∩C = T
    # can be rewritten. For fixed A and T, the sets B with A∩B = T form a class.
    # Within that class, B1 and B2 conflict if B1∩B2 = T.
    #
    # So for each (A, T), we have a graph on {B : A∩B = T} where edges are pairs
    # with B1∩B2 = T. We can select at most one from each connected component? No,
    # we can select any independent set from that graph, but the constraint is per-triple.
    #
    # This is still a hypergraph problem. Let me just implement a good local search.

    # Phase 1: Greedy initialization with multiple random orders
    best_family = []
    best_size = 0

    num_greedy = 50
    for run in range(num_greedy):
        if time.time() > deadline - 20:
            break

        order = list(range(total))
        rng.shuffle(order)

        F = []
        inter_maps = []  # inter_maps[i] = {t: [j1, j2, ...]} where inter[F[i]][F[j]] = t

        for idx in order:
            m = all_sets[idx]
            conflict = False

            for i in range(len(F)):
                t = inter[i][idx]
                lst = inter_maps[i].get(t)
                if lst:
                    for j in lst:
                        if inter[j][idx] == t:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})

                for i in range(idx_new):
                    t = inter[i][idx]
                    if t not in inter_maps[i]:
                        inter_maps[i][t] = []
                    inter_maps[i][t].append(idx_new)
                    if t not in inter_maps[idx_new]:
                        inter_maps[idx_new][t] = []
                    inter_maps[idx_new][t].append(i)

                if len(F) >= max_target:
                    break

        if len(F) > best_size:
            best_size = len(F)
            best_family = F[:]

    if time.time() > deadline - 5:
        # Convert and return
        result = []
        for mask in best_family:
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(i + 1)
            result.append(sorted(elems))
        return result

    # Phase 2: Local search - try to add more sets or swap
    # Represent family as a set of indices
    F_set = set(best_family)
    F_list = list(F_set)
    F_indices = [all_sets.index(m) for m in F_list]  # This is slow, let's use a dict

    # Better: keep track by index
    in_family = [False] * total
    for idx in F_indices:
        in_family[idx] = True

    # Try to add sets not in family
    for idx in range(total):
        if time.time() > deadline - 2:
            break
        if in_family[idx]:
            continue
        m = all_sets[idx]
        conflict = False
        for i in F_indices:
            t = inter[i][idx]
            # Check if any j in F has inter[i][j] == t and inter[j][idx] == t
            # We need inter_maps for this. Let's rebuild.
            pass

    # This is getting complicated. Let me use a cleaner local search.
    # Rebuild inter_maps for the current family
    def build_inter_maps(fam_indices):
        """fam_indices: list of indices into all_sets"""
        im = [{} for _ in fam_indices]
        for a in range(len(fam_indices)):
            for b in range(a + 1, len(fam_indices)):
                t = inter[fam_indices[a]][fam_indices[b]]
                if t not in im[a]:
                    im[a][t] = []
                im[a][t].append(b)
                if t not in im[b]:
                    im[b][t] = []
                im[b][t].append(a)
        return im

    def check_conflict(idx, fam_indices, im):
        """Check if adding idx to fam_indices creates a sunflower"""
        for i in range(len(fam_indices)):
            t = inter[fam_indices[i]][idx]
            lst = im[i].get(t)
            if lst:
                for j in lst:
                    if inter[fam_indices[j]][idx] == t:
                        return True
        return False

    # Local search: try adding one set at a time
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        im = build_inter_maps(F_indices)
        for idx in range(total):
            if time.time() > deadline - 2:
                break
            if in_family[idx]:
                continue
            if not check_conflict(idx, F_indices, im):
                # Add it
                F_indices.append(idx)
                in_family[idx] = True
                improved = True
                break  # Rebuild inter_maps and restart

    # Try swaps: remove one, add two
    if time.time() < deadline - 3:
        for remove_idx in list(F_indices):
            if time.time() > deadline - 2:
                break
            if len(F_indices) <= 1:
                continue
            # Temporarily remove
            in_family[remove_idx] = False
            F_indices.remove(remove_idx)
            im = build_inter_maps(F_indices)

            # Try to add two sets
            added = []
            for add1 in range(total):
                if in_family[add1]:
                    continue
                if check_conflict(add1, F_indices, im):
                    continue
                F_indices.append(add1)
                in_family[add1] = True
                added.append(add1)
                im = build_inter_maps(F_indices)
                for add2 in range(total):
                    if in_family[add2] or add2 == add1:
                        continue
                    if check_conflict(add2, F_indices, im):
                        continue
                    F_indices.append(add2)
                    in_family[add2] = True
                    added.append(add2)
                    break
                break

            if len(added) < 2:
                # Rollback
                for a in added:
                    in_family[a] = False
                    F_indices.remove(a)
                in_family[remove_idx] = True
                F_indices.append(remove_idx)

    # Final result
    result = []
    for idx in F_indices:
        mask = all_sets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
