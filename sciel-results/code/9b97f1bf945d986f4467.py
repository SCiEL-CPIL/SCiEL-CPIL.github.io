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

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid."""
        selected = []
        selected_set = set()
        for v in order:
            # Check if adding v is valid
            valid = True
            for (a, b) in bad_pairs[v]:
                if a in selected_set and b in selected_set:
                    valid = False
                    break
            if valid:
                selected.append(v)
                selected_set.add(v)
        return selected

    def local_improve(subset):
        """Try to improve subset by adding vertices that can fit."""
        s = set(subset)
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in s:
                    continue
                valid = True
                for (a, b) in bad_pairs[v]:
                    if a in s and b in s:
                        valid = False
                        break
                if valid:
                    s.add(v)
                    improved = True
        return list(s)

    best = []
    random.seed(42)

    # Try multiple random orderings
    for trial in range(500):
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

    # Additional randomized local search
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        candidate = greedy_construction(order)
        # Do a few rounds of random perturbation
        for _ in range(5):
            if time.time() > deadline:
                break
            if not candidate:
                break
            # Remove a random element
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
            candidate = local_improve(candidate)
        if len(candidate) > len(best):
            best = candidate

    if not best:
        # Fallback: single greedy pass
        best = greedy_construction(list(range(N)))
        best = local_improve(best)

    # Convert back to 1-based
    return [v + 1 for v in sorted(best)]
# EVOLVE-BLOCK-END
