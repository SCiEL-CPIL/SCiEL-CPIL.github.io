import time
import random
from itertools import combinations

def _precompute():
    """Precompute all triples, all K(2,2,2) configs, and adjacency structures."""
    n = 16
    # All 560 triples
    triples = []
    triple_index = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
                triple_index[(a, b, c)] = len(triples) - 1
    m = len(triples)  # 560

    # For each triple, precompute the 3 pairs in it
    triple_pairs = []
    for t in triples:
        triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])

    # Find all K(2,2,2) configurations:
    # A K(2,2,2) is defined by 3 disjoint pairs of points: (a,a'), (b,b'), (c,c')
    # The 8 transversal triples are: {a,b,c}, {a,b,c'}, {a',b,c}, {a',b,c'},
    #                                {a,b',c}, {a,b',c'}, {a',b',c}, {a',b',c'}
    # All 8 must be present as triples for the config to be violated.

    configs = []  # list of 8-tuples of triple indices
    config_to_edges = []  # for each config, the 8 edge indices

    # Choose 3 disjoint pairs from 16 points
    # Pairs are of the form (a, a') where a != a'
    # We need 3 disjoint pairs covering 6 distinct points
    # For efficiency, iterate over combinations of 6 points, then all perfect matchings

    all_points = list(range(n))
    config_count = 0

    for six in combinations(all_points, 6):
        # All perfect matchings of 6 points into 3 pairs
        # There are 15 perfect matchings
        s = list(six)
        matchings = []
        # Recursive generation of perfect matchings
        def gen_matchings(pts, used, current):
            if not pts:
                matchings.append(current[:])
                return
            # Pick the first unused point
            first = None
            for i in range(len(pts)):
                if i not in used:
                    first = i
                    break
            if first is None:
                return
            used.add(first)
            for j in range(first + 1, len(pts)):
                if j not in used:
                    used.add(j)
                    current.append((pts[first], pts[j]))
                    gen_matchings(pts, used, current)
                    current.pop()
                    used.remove(j)
            used.remove(first)

        gen_matchings(s, set(), [])
        for matching in matchings:
            # matching is a list of 3 pairs
            (a, a2), (b, b2), (c, c2) = matching
            # The 8 transversal triples
            trans = [
                (a, b, c), (a, b, c2), (a2, b, c), (a2, b, c2),
                (a, b2, c), (a, b2, c2), (a2, b2, c), (a2, b2, c2)
            ]
            # Get indices
            indices = []
            valid = True
            for t in trans:
                if t in triple_index:
                    indices.append(triple_index[t])
                else:
                    valid = False
                    break
            if valid:
                configs.append(tuple(indices))
                config_to_edges.append(indices)
                config_count += 1

    K = len(configs)

    # For each edge (triple), which configs does it participate in?
    edge_to_configs = [[] for _ in range(m)]
    for ci in range(K):
        for ei in config_to_edges[ci]:
            edge_to_configs[ei].append(ci)

    return triples, triple_pairs, configs, config_to_edges, edge_to_configs, m, K

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin

    triples, triple_pairs, configs, config_to_edges, edge_to_configs, m, K = _precompute()

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return time.time() > deadline

    # Initial solution: all 560 triples
    present = [True] * m
    config_counts = [8] * K  # each config has all 8 edges present
    violated_count = K  # all configs violated initially

    # Phase 1: Greedy removal to eliminate all violations
    # For each violated config, we need to remove at least one of its 8 edges
    # Greedy: repeatedly find the edge that participates in the most violated configs, remove it

    violated_configs = set(range(K))  # all violated initially

    while violated_configs and not check_deadline():
        # Find the edge that resolves the most violated configs
        best_edge = -1
        best_resolves = -1
        for e in range(m):
            if not present[e]:
                continue
            resolves = 0
            for ci in edge_to_configs[e]:
                if ci in violated_configs:
                    resolves += 1
            if resolves > best_resolves:
                best_resolves = resolves
                best_edge = e
            if best_resolves >= 8:  # can't do better
                break

        if best_edge == -1 or best_resolves == 0:
            break

        # Remove this edge
        present[best_edge] = False
        for ci in edge_to_configs[best_edge]:
            config_counts[ci] -= 1
            if config_counts[ci] < 8 and ci in violated_configs:
                # Check if it's no longer violated
                # A config is violated iff all 8 edges are present
                # If config_counts[ci] < 8, it's no longer violated
                # But wait: config_counts tracks the number of present edges in the config
                # Initially 8, we decrement when we remove an edge
                # A config is violated iff config_counts[ci] == 8
                violated_configs.discard(ci)
                violated_count -= 1

    # Phase 2: Add back safe triples
    absent_edges = [i for i in range(m) if not present[i]]
    random.shuffle(absent_edges)

    for e in absent_edges:
        if check_deadline():
            break
        # Check if adding this edge creates any violation
        safe = True
        for ci in edge_to_configs[e]:
            if config_counts[ci] == 7:
                # Adding this edge would make it 8, causing violation
                safe = False
                break
        if safe:
            present[e] = True
            for ci in edge_to_configs[e]:
                config_counts[ci] += 1

    # Phase 3: Local improvement - try swapping edges
    # For each absent edge, try adding it and removing one of its conflicting edges
    for e in absent_edges:
        if check_deadline():
            break
        if present[e]:
            continue
        # Find configs that would be violated if we add e
        conflicting_configs = []
        for ci in edge_to_configs[e]:
            if config_counts[ci] == 7:
                conflicting_configs.append(ci)
        if not conflicting_configs:
            # Can add directly
            present[e] = True
            for ci in edge_to_configs[e]:
                config_counts[ci] += 1
            continue
        # Try to resolve conflicts by removing other edges
        for ci in conflicting_configs:
            # Find another edge in this config that we could remove
            # Prefer removing an edge that participates in few other configs
            best_remove = -1
            best_cost = float('inf')
            for other_e in config_to_edges[ci]:
                if other_e == e or not present[other_e]:
                    continue
                # Cost: how many other configs does this edge participate in
                cost = len(edge_to_configs[other_e])
                if cost < best_cost:
                    best_cost = cost
                    best_remove = other_e
            if best_remove != -1:
                # Remove best_remove, add e
                present[best_remove] = False
                for ci2 in edge_to_configs[best_remove]:
                    config_counts[ci2] -= 1
                present[e] = True
                for ci2 in edge_to_configs[e]:
                    config_counts[ci2] += 1
                break

    # Final validity check and repair
    for ci in range(K):
        if config_counts[ci] == 8:
            # Remove one edge from this config
            for e in config_to_edges[ci]:
                if present[e]:
                    present[e] = False
                    for ci2 in edge_to_configs[e]:
                        config_counts[ci2] -= 1
                    break

    # Build result
    result = [list(triples[i]) for i in range(m) if present[i]]
    return result
