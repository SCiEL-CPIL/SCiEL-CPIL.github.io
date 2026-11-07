import time
import sys

def construct():
    n = 12
    m = 40
    total_masks = 1 << n

    # Build comparability as bitmasks for speed
    # comp[i] is a bitmask where bit j is set iff set i is comparable to set j
    comp = [0] * total_masks
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)

    # Precompute degree of each mask
    degree = [0] * total_masks
    for i in range(total_masks):
        degree[i] = bin(comp[i]).count('1')

    # Initial solution: greedy construction
    def greedy_construct():
        family = []
        # Start with a chain
        cur = 0
        family.append(0)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)

        fam_set = set(family)

        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in fam_set:
                    continue
                gain = 0
                for f in family:
                    if (s & ~f) == 0 or (f & ~s) == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            fam_set.add(best_mask)
        return family

    best_family = greedy_construct()
    best_score = 0
    for i in range(len(best_family)):
        for j in range(i + 1, len(best_family)):
            if (best_family[i] & ~best_family[j]) == 0 or (best_family[j] & ~best_family[i]) == 0:
                best_score += 1

    # Branch and bound for maximum clique in comparability graph
    # We want a clique of size m=40 in the comparability graph
    # The comparability graph has 4096 vertices; we need to find a 40-clique

    deadline = time.time() + 85  # safety margin

    # For B&B: we build a clique by adding vertices one at a time
    # Upper bound: current_clique_size + min(remaining_slots, sum of top (remaining_slots) degrees in candidate set)
    # More precisely: if we have a partial clique P and candidate set C (vertices comparable to all in P),
    # then |clique| <= |P| + min(m - |P|, number of vertices in C that can be added)
    # Tighter: |clique| <= |P| + (number of vertices in C) but we need them to form a clique among themselves
    # Simple upper bound: |P| + |C| (if all candidates were mutually comparable)
    # Better: use greedy coloring bound or degree sum

    # Let's use a simpler but effective approach:
    # At each node, candidates = vertices comparable to all in current clique
    # Upper bound = len(current_clique) + len(candidates)
    # This is valid because the final clique can have at most len(candidates) more vertices from C

    # To make this tighter, we can use: upper bound = len(P) + min(len(C), m - len(P))
    # But even better: we can compute a greedy coloring of the subgraph induced by C
    # and use the max color as an upper bound on clique size in C

    # For speed with 4096 vertices, let's use a simpler bound first and see if it's fast enough

    def count_comparables(family):
        cnt = 0
        L = len(family)
        for i in range(L):
            for j in range(i + 1, L):
                if (family[i] & ~family[j]) == 0 or (family[j] & ~family[i]) == 0:
                    cnt += 1
        return cnt

    # Local search polish
    def local_search(family, fam_set, score, deadline):
        L = len(family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(L):
                if time.time() > deadline:
                    break
                old_mask = family[i]
                old_comp = 0
                for j in range(L):
                    if j == i:
                        continue
                    if (old_mask & ~family[j]) == 0 or (family[j] & ~old_mask) == 0:
                        old_comp += 1
                best_delta = 0
                best_repl = None
                for s in range(total_masks):
                    if s in fam_set:
                        continue
                    new_comp = 0
                    for j in range(L):
                        if j == i:
                            continue
                        if (s & ~family[j]) == 0 or (family[j] & ~s) == 0:
                            new_comp += 1
                    delta = new_comp - old_comp
                    if delta > best_delta:
                        best_delta = delta
                        best_repl = s
                if best_repl is not None and best_delta > 0:
                    new_family = family[:]
                    new_family[i] = best_repl
                    new_fam_set = fam_set - {old_mask}
                    new_fam_set.add(best_repl)
                    new_score = score + best_delta
                    family = new_family
                    fam_set = new_fam_set
                    score = new_score
                    improved = True
                    break
        return family, fam_set, score

    # Try to improve with local search
    best_fam_set = set(best_family)
    best_family, best_fam_set, best_score = local_search(best_family, best_fam_set, best_score, deadline)

    # Branch and bound to find better solutions
    # We'll try to find a 40-clique with high density
    # Actually, we want to maximize the number of comparable pairs, which is the number of edges in the induced subgraph
    # So we want a 40-vertex subgraph with maximum edges, not just any 40-clique

    # The B&B approach: we're selecting 40 vertices to maximize the number of edges among them
    # This is a densest subgraph problem, which is NP-hard

    # Given the time constraint, let's focus on improving the local search with better initialization
    # and multiple restarts

    # Multiple random restarts with local search
    import random
    random.seed(42)

    best_overall = best_score
    best_overall_family = best_family[:]

    start_time = time.time()
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Random initialization
        family = []
        fam_set = set()
        # Start with a random chain-like structure
        cur = 0
        if random.random() < 0.5:
            family.append(0)
            fam_set.add(0)
            cur = 0
        for i in range(n):
            if random.random() < 0.7:
                cur |= (1 << i)
                family.append(cur)
                fam_set.add(cur)

        while len(family) < m:
            # Pick random candidate with bias toward comparable sets
            best_gain = -1
            best_mask = None
            candidates = []
            for s in range(total_masks):
                if s in fam_set:
                    continue
                gain = 0
                for f in family:
                    if (s & ~f) == 0 or (f & ~s) == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            fam_set.add(best_mask)

        # Local search
        score = count_comparables(family)
        family, fam_set, score = local_search(family, fam_set, score, deadline - 5)

        if score > best_overall:
            best_overall = score
            best_overall_family = family[:]

    # Final local search on the best solution
    best_fam_set = set(best_overall_family)
    best_overall_family, best_fam_set, best_overall = local_search(best_overall_family, best_fam_set, best_overall, deadline)

    # Convert to list of lists
    result = []
    for mask in best_overall_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)

    return result
