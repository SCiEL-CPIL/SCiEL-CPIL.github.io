import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill

    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330

    # Precompute compatibility: two 4-sets are compatible if |S ∩ T| <= 2
    masks = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(subsets):
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks[i] = m

    # Popcount table for values 0 to 2^11-1
    popcount_table = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_table[i] = bin(i).count('1')

    # Compatibility matrix: compatible[i,j] = True if |S_i ∩ S_j| <= 2
    m1 = masks[:, np.newaxis]
    m2 = masks[np.newaxis, :]
    and_matrix = m1 & m2
    popcount_matrix = popcount_table[and_matrix]
    compatible = (popcount_matrix <= 2)  # (n, n) boolean

    # Conflict graph: edge if NOT compatible
    # We want maximum clique in the compatibility graph (equivalently, max independent set in conflict graph)
    # Actually we want a set of mutually compatible subsets, which is a clique in the compatibility graph

    # For branch-and-bound with coloring, we work on the compatibility graph
    # adj[i] = set of j such that compatible[i,j] and j != i
    # Use bitsets for efficiency: adj_bits[i] is a bitmask of compatible neighbors

    adj_bits = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        # compatible[i, :] gives which j are compatible with i
        # Need to mask to 330 bits - but we only have 64-bit ints
        # 330 > 64, so we need multiple words or use a different approach
        pass

    # Since n=330 > 64, we can't use a single uint64. Let's use a list of booleans or numpy arrays.
    # For branch and bound, we'll use the adjacency matrix directly.

    # Better approach: use the complement. We want max clique in compatibility graph.
    # Branch and bound with greedy coloring upper bound.

    # Represent adjacency as a list of sets or as a 2D boolean array (already have compatible)
    # compatible[i,j] = True means edge in compatibility graph (i != j)

    # Greedy coloring for upper bound:
    # Given a candidate set P, color it greedily to get an upper bound on clique size in P.

    def greedy_color_bound(candidates, compat):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns the number of colors used (upper bound on clique size).
        candidates: list/array of indices
        compat: the compatibility matrix (n x n boolean)
        """
        if len(candidates) == 0:
            return 0

        # Order candidates by degree in the induced subgraph (descending) for better coloring
        # Compute degree of each candidate within the candidate set
        deg = np.zeros(len(candidates), dtype=np.int32)
        for i, ci in enumerate(candidates):
            for j, cj in enumerate(candidates):
                if i != j and compat[ci, cj]:
                    deg[i] += 1

        # Order by descending degree
        order = np.argsort(-deg)
        ordered_candidates = [candidates[i] for i in order]

        # Greedy coloring: assign each vertex the smallest color not used by its neighbors
        color = {}  # vertex -> color
        next_color = 0
        for v in ordered_candidates:
            # Find colors used by already-colored neighbors
            used = set()
            for u in ordered_candidates:
                if u != v and u in color and compat[v, u]:
                    used.add(color[u])
            # Assign smallest available color
            c = 0
            while c in used:
                c += 1
            color[v] = c
            next_color = max(next_color, c + 1)

        return next_color

    # Branch and bound for maximum clique
    # We'll use a recursive approach with pruning

    best_clique = []
    best_size = 0

    # Initial greedy solution
    # Greedy: repeatedly pick the vertex with highest degree among remaining
    remaining = set(range(n))
    current_clique = []
    while remaining:
        # Pick vertex with max degree in remaining
        best_v = None
        best_deg = -1
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and compatible[v, u])
            if deg > best_deg:
                best_deg = deg
                best_v = v
        current_clique.append(best_v)
        remaining.discard(best_v)
        # Remove all non-neighbors of best_v from remaining
        remaining = {u for u in remaining if compatible[best_v, u]}

    best_size = len(current_clique)
    best_clique = current_clique[:]

    # Now do branch and bound starting from an empty set
    # Use a stack-based approach to avoid recursion depth issues

    # For efficiency, precompute the neighbor list for each vertex
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and compatible[i, j]:
                neighbors[i].add(j)

    def extend_clique(clique, candidates):
        """
        Try to extend the current clique using candidates.
        clique: list of vertices in current clique
        candidates: set of vertices that can be added (all compatible with all in clique)
        """
        nonlocal best_clique, best_size

        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return

        # Upper bound via greedy coloring
        cand_list = list(candidates)
        bound = greedy_color_bound(cand_list, compatible)
        if len(clique) + bound <= best_size:
            return

        # Branch on each candidate
        # Order by degree for better pruning
        deg = {v: sum(1 for u in candidates if u != v and compatible[v, u]) for v in candidates}
        ordered = sorted(candidates, key=lambda v: -deg[v])

        for v in ordered:
            if len(clique) + len(candidates) <= best_size:
                break
            new_clique = clique + [v]
            new_candidates = candidates & neighbors[v]
            extend_clique(new_clique, new_candidates)
            candidates.discard(v)

    # Start branch and bound
    # Initial candidates: all vertices
    initial_candidates = set(range(n))

    # To make it faster, start with a good initial ordering
    # Use the greedy solution as a guide

    # Actually, let's use a simpler iterative approach with multiple restarts
    # since full B&B might be too slow for n=330

    # Alternative: use a heuristic approach with multiple passes
    # 1. Greedy with different orderings
    # 2. Local search (swap, add, remove)

    # Let's do a combination:
    # 1. Multiple greedy initializations with different random orderings
    # 2. Local search refinement

    rng = np.random.default_rng(42)

    best_selected = np.zeros(n, dtype=bool)
    best_score = 0

    for trial in range(50):  # multiple trials
        if time.time() > deadline:
            break

        # Greedy initialization with random tie-breaking
        remaining = set(range(n))
        selected_set = set()

        while remaining:
            # Pick vertex with max degree among remaining, with random tie-breaking
            best_v = None
            best_deg = -1
            best_candidates = []
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and compatible[v, u])
                if deg > best_deg:
                    best_deg = deg
                    best_candidates = [v]
                elif deg == best_deg:
                    best_candidates.append(v)
            # Random choice among best candidates
            if best_candidates:
                best_v = rng.choice(best_candidates)
            else:
                break

            selected_set.add(best_v)
            remaining.discard(best_v)
            # Keep only neighbors of best_v
            remaining = {u for u in remaining if compatible[best_v, u]}

        # Local search refinement
        current_selected = set(selected_set)
        current_score = len(current_selected)

        for _ in range(1000):  # local search iterations
            if time.time() > deadline:
                break

            # Try to add a vertex
            unselected = [v for v in range(n) if v not in current_selected]
            if unselected:
                # Find vertices that are compatible with all in current_selected
                addable = [v for v in unselected if all(compatible[v, u] for u in current_selected)]
                if addable:
                    # Add one (random choice)
                    v = rng.choice(addable)
                    current_selected.add(v)
                    current_score += 1

            # Try swap: remove one, add one
            if current_selected:
                out_candidates = list(current_selected)
                in_candidates = [v for v in range(n) if v not in current_selected]
                if out_candidates and in_candidates:
                    # Find valid swaps
                    for _ in range(20):  # try a few random swaps
                        out_v = rng.choice(out_candidates)
                        in_v = rng.choice(in_candidates)
                        # Check if in_v is compatible with all in (current_selected - {out_v})
                        new_set = (current_selected - {out_v}) | {in_v}
                        valid = True
                        for a in new_set:
                            for b in new_set:
                                if a != b and not compatible[a, b]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            current_selected = new_set
                            current_score = len(current_selected)

        if current_score > best_score:
            best_score = current_score
            best_selected = np.zeros(n, dtype=bool)
            for v in current_selected:
                best_selected[v] = True

    # Build vectors from best selected subsets
    sel_indices = np.where(best_selected)[0]

    vectors = []
    for idx in sel_indices:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)

    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
