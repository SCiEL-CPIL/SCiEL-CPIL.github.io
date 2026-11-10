"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 75  # 15s safety margin under 90s kill
    N = 128
    # Precompute GCD matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Build hyperedges: triples (i,j,k) with equal pairwise gcds
    # For speed, use bitmasks: vertices 1..128 -> bits 0..127
    # A hyperedge is a frozenset of 3 vertex indices (0-based)
    # We need to find max independent set in this 3-uniform hypergraph.

    # Precompute for each pair (i,j) the set of k such that {gcd(i,j),gcd(i,k),gcd(j,k)} are all equal
    # Actually: the condition is gcd(i,j) == gcd(i,k) == gcd(j,k)
    # Let g = gcd(i,j). Then we need gcd(i,k) == g and gcd(j,k) == g.

    # Build conflict structure: for each vertex v, list of hyperedges containing v
    # And for each hyperedge, the 3 vertices.

    # To make this tractable, let's precompute all bad triples.
    # There are C(128,3) = 341,376 triples. Checking each is feasible.

    # Represent each vertex by its index 0..127 (vertex i corresponds to number i+1)
    # Precompute: for each pair (a,b) (0-based), the set of c such that (a,b,c) is a bad triple.
    # bad_triples[a][b] = set of c values

    # Actually for branch-and-bound, we need:
    # - For a set S of chosen vertices, check if adding v creates a bad triple.
    # - A bad triple is formed when we add v if there exist a,b in S such that (a,b,v) is bad.

    # Precompute: conflicts[v] = set of pairs (a,b) with a<b, a!=v, b!=v such that (a,b,v) is bad
    # Then when we have selected set S, adding v is OK iff no pair (a,b) in conflicts[v] has both a,b in S.

    # Let's precompute all bad triples as a set of frozensets for quick lookup.
    # But that's 341K entries - too many to store as frozensets efficiently.

    # Better: for each vertex v, store the set of other vertices u such that there exists w
    # making a bad triple. Actually we need the full hyperedge structure.

    # Alternative approach: precompute for each pair (a,b) the bitmask of c's that form a bad triple.
    # pair_bad_mask[a][b] = bitmask of c (0-based) where gcd(a+1,b+1)==gcd(a+1,c+1)==gcd(b+1,c+1)

    # For branch and bound with 128 vertices, we need efficient conflict checking.
    # When we tentatively add vertex v to selected set S:
    #   For each pair (a,b) in S, check if (a,b,v) is a bad triple.
    # This is O(|S|^2) per vertex addition, which is too slow for deep search.

    # Better: precompute for each v, the set of "forbidden pairs" - pairs (a,b) that would
    # form a bad triple with v. Then adding v is valid iff S contains no such forbidden pair.
    # We can maintain for each v a count of how many forbidden pairs are "active" (both in S).

    # Let's use a different representation:
    # For each vertex v, precompute bad_pairs[v] = list of (a,b) with a<b, a!=v, b!=v,
    # such that gcd(v+1,a+1)==gcd(v+1,b+1)==gcd(a+1,b+1)

    # Precompute bad_pairs
    bad_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd(a+1, b+1)
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd(a+1, c+1) == ga and gcd(b+1, c+1) == ga:
                    # (a,b,c) is a bad triple
                    bad_pairs[c].append((a,b))
                    bad_pairs[a].append((b,c))
                    bad_pairs[b].append((a,c))

    # For branch-and-bound, we need to check if adding v to S creates any bad triple.
    # This happens iff there exist a,b in S such that (a,b,v) is bad, i.e., (a,b) is in bad_pairs[v].
    # So we need: does S contain any pair from bad_pairs[v]?

    # Efficient check: maintain for each v, a counter of how many pairs in bad_pairs[v] have both elements in S.
    # When we add x to S, for each v != x, we need to increment the counter for v if (x,w) is in bad_pairs[v] for some w in S.
    # This is complex. Let's use a simpler but still efficient approach.

    # Alternative: Use the fact that we're looking for maximum independent set.
    # Use a greedy + local search approach with multiple restarts, which the cell finding showed works well (score 1.5).

    # Given the complexity of exact B&B on 128-vertex 3-uniform hypergraph, let's use:
    # 1. Multiple randomized greedy passes with different orderings
    # 2. Local improvement (try to add/remove/swap vertices)
    # 3. Deadline-guarded

    def is_valid_subset(subset):
        """Check if subset has no bad triple."""
        s = set(subset)
        for v in s:
            for (a, b) in bad_pairs[v]:
                if a in s and b in s:
                    return False
        return True

    # Precompute for each vertex v, a list of (partner, other) pairs
    # where (v, partner, other) would be a bad triple.
    # i.e., bad_pairs[v] contains (a,b) meaning (v,a,b) is bad.
    # For efficient blocked-count maintenance:
    # When we add vertex x to S, for each v not in S, the blocked count for v
    # increases by the number of (a,b) in bad_pairs[v] such that a==x and b in S,
    # or b==x and a in S.
    # So we need: for each pair (x, v), how many partners p such that (x, p) is in bad_pairs[v] and p is in S.
    # Equivalently: for each v, and each x in S, count how many p in S such that (x,p) in bad_pairs[v].
    # 
    # Simpler: precompute for each ordered pair (x, v) with x != v, the list of p
    # such that (x, p) appears in bad_pairs[v]. Then when x is added, for each v,
    # blocked[v] += number of p in that list that are also in S.
    # This is still expensive. Let's use a different approach.
    #
    # For each v, precompute: neighbors_in_bad[v] = set of all u such that there exists w
    # with (u, w) in bad_pairs[v]. Actually we need pair-level info.
    #
    # Let me precompute: for each v, bad_pair_set[v] = set of frozenset({a,b}) for (a,b) in bad_pairs[v]
    # And for faster lookup: for each v, and for each u, the list of w such that (u,w) in bad_pairs[v]
    # i.e., bad_pair_lookup[v][u] = [w1, w2, ...] where (u, wi) in bad_pairs[v]

    # Precompute lookup structure
    # bad_pair_lookup[v][u] = list of w such that (u, w) is in bad_pairs[v] (i.e., (v,u,w) is a bad triple)
    bad_pair_lookup = [{} for _ in range(N)]
    for v in range(N):
        for (a, b) in bad_pairs[v]:
            if a not in bad_pair_lookup[v]:
                bad_pair_lookup[v][a] = []
            bad_pair_lookup[v][a].append(b)
            if b not in bad_pair_lookup[v]:
                bad_pair_lookup[v][b] = []
            bad_pair_lookup[v][b].append(a)

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid, using blocked counts."""
        selected = []
        selected_set = set()
        # blocked[v] = number of bad pairs (a,b) in bad_pairs[v] with both a,b in selected_set
        blocked = [0] * N
        for v in order:
            if v in selected_set:
                continue
            if blocked[v] == 0:
                selected.append(v)
                selected_set.add(v)
                # Update blocked counts for all other vertices
                for u in range(N):
                    if u == v or u in selected_set:
                        continue
                    # How many w in selected_set such that (v, w) is in bad_pairs[u]?
                    # i.e., w is in bad_pair_lookup[u][v] and w is in selected_set
                    if v in bad_pair_lookup[u]:
                        for w in bad_pair_lookup[u][v]:
                            if w in selected_set:
                                blocked[u] += 1
        return selected

    def local_improve(subset):
        """Try to improve subset by adding vertices that can fit, using blocked counts."""
        s = set(subset)
        # Compute initial blocked counts
        blocked = [0] * N
        for v in s:
            for u in range(N):
                if u == v or u in s:
                    continue
                if v in bad_pair_lookup[u]:
                    for w in bad_pair_lookup[u][v]:
                        if w in s:
                            blocked[u] += 1
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in s:
                    continue
                if blocked[v] == 0:
                    s.add(v)
                    improved = True
                    # Update blocked for others
                    for u in range(N):
                        if u == v or u in s:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s:
                                    blocked[u] += 1
        return list(s)

    def swap_improve(subset, rng):
        """Try swap: remove one random element, then greedily re-add."""
        s = set(subset)
        if not s:
            return list(s)
        # Compute blocked counts
        blocked = [0] * N
        for v in s:
            for u in range(N):
                if u == v or u in s:
                    continue
                if v in bad_pair_lookup[u]:
                    for w in bad_pair_lookup[u][v]:
                        if w in s:
                            blocked[u] += 1
        # Try removing each element and see if we can gain
        best_size = len(s)
        best_set = None
        elements = list(s)
        rng.shuffle(elements)
        for rem in elements:
            if time.time() > deadline:
                break
            # Remove rem
            s2 = s - {rem}
            # Recompute blocked for s2 (expensive but needed)
            blocked2 = [0] * N
            for v in s2:
                for u in range(N):
                    if u == v or u in s2:
                        continue
                    if v in bad_pair_lookup[u]:
                        for w in bad_pair_lookup[u][v]:
                            if w in s2:
                                blocked2[u] += 1
            # Greedily add
            for v in range(N):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = s2
        if best_set is not None:
            return list(best_set)
        return list(s)

    best = []
    random.seed(42)
    rng = random.Random(12345)

    # Try multiple random orderings
    for trial in range(300):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Also try sorted orderings and other heuristics
    for order in [list(range(N)), list(range(N-1, -1, -1)),
                  sorted(range(N), key=lambda x: len(bad_pairs[x])),
                  sorted(range(N), key=lambda x: -len(bad_pairs[x]))]:
        if time.time() > deadline:
            break
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Swap-improvement on best found so far
    for trial in range(50):
        if time.time() > deadline:
            break
        if not best:
            break
        improved = swap_improve(best, rng)
        if len(improved) > len(best):
            best = improved
        else:
            break

    # Additional randomized local search with perturbation
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve(candidate)
        # Do a few rounds of random perturbation + swap
        for _ in range(3):
            if time.time() > deadline:
                break
            if not candidate:
                break
            candidate = swap_improve(candidate, rng)
        if len(candidate) > len(best):
            best = candidate

    if not best:
        # Fallback: single greedy pass
        best = greedy_construction(list(range(N)))
        best = local_improve(best)

    # Convert back to 1-based
    return [v + 1 for v in sorted(best)]
# EVOLVE-BLOCK-END
