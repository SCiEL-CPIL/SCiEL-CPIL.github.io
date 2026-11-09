import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 82  # 8s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))

    total_edges = len(edge_set)

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    def time_left():
        return deadline - time.time()

    def expand_biclique(L, R, remaining):
        """Expand biclique to maximal size."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            if time_left() < 0.3:
                break
            changed = False
            # Try to add to L
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for r in R:
                    if make_edge(w, r) not in remaining:
                        valid = False
                        break
                if valid:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True

            # Try to add to R
            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for l in L:
                    if make_edge(w, l) not in remaining:
                        valid = False
                        break
                if valid:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        return L, R

    def find_biclique_from_edge(u, v, remaining):
        """Seed biclique with u on left, v on right, then expand."""
        L, R = expand_biclique({u}, {v}, remaining)
        if not L or not R:
            return None
        # Verify
        for l in L:
            for r in R:
                if l != r and make_edge(l, r) not in remaining:
                    return None
        return (L, R)

    def find_best_biclique(remaining, rng=None):
        """Find the largest biclique in the remaining edge set."""
        if not remaining:
            return None

        best_biclique = None
        best_size = 0

        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for (u, v) in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        # Get edges sorted by combined degree (descending)
        edge_list = list(remaining)
        edge_list.sort(key=lambda e: (-(len(rem_adj[e[0]]) + len(rem_adj[e[1]])), e[0], e[1]))

        # Try a limited number of seed edges
        max_seeds = 60
        if rng:
            # Shuffle and take first max_seeds
            rng.shuffle(edge_list)
            edge_list = edge_list[:max_seeds]
        else:
            edge_list = edge_list[:max_seeds]

        for (u, v) in edge_list:
            if time_left() < 0.5:
                break

            # Try u left, v right
            bc = find_biclique_from_edge(u, v, remaining)
            if bc is not None:
                L, R = bc
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)

            # Try v left, u right
            bc2 = find_biclique_from_edge(v, u, remaining)
            if bc2 is not None:
                L, R = bc2
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)

        return best_biclique

    def greedy_partition(remaining, rng=None, max_seeds=60):
        """Greedy: repeatedly find largest biclique, remove its edges."""
        bicliques = []
        rem = set(remaining)

        while rem:
            if time_left() < 1.0:
                break

            best_bc = find_best_biclique(rem, rng)

            if best_bc is None:
                # Fallback: single edge
                e = next(iter(rem))
                best_bc = ({e[0]}, {e[1]})

            L, R = best_bc
            bicliques.append((frozenset(L), frozenset(R)))

            # Remove covered edges
            to_remove = set()
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in rem:
                            to_remove.add(e)
            rem -= to_remove

        # Handle remaining edges
        for e in rem:
            bicliques.append((frozenset({e[0]}), frozenset({e[1]})))

        return bicliques

    def try_merge(bicliques, edge_set):
        """Try to merge pairs of bicliques. Returns merged list or None."""
        n_bics = len(bicliques)
        for i in range(n_bics):
            for j in range(i + 1, n_bics):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]

                candidates = []

                # Option 1: Union both sides
                newL = L1 | L2
                newR = R1 | R2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Option 2: Cross merge
                newL = L1 | R2
                newR = R1 | L2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Option 3: L2 subset of L1
                if L2 <= L1:
                    newR = R1 | R2
                    if not (L1 & newR):
                        candidates.append((L1, newR))

                # Option 4: R2 subset of R1
                if R2 <= R1:
                    newL = L1 | L2
                    if not (newL & R1):
                        candidates.append((newL, R1))

                # Option 5: L1 subset of L2
                if L1 <= L2:
                    newR = R1 | R2
                    if not (L2 & newR):
                        candidates.append((L2, newR))

                # Option 6: R1 subset of R2
                if R1 <= R2:
                    newL = L1 | L2
                    if not (newL & R2):
                        candidates.append((newL, R2))

                for newL, newR in candidates:
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue

                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        result = list(bicliques)
                        result.pop(j)
                        result.pop(i)
                        result.append((frozenset(newL), frozenset(newR)))
                        return result

        return None

    def try_vertex_reassign(bicliques, edge_set):
        """Try to move a vertex from one biclique to another to enable a merge."""
        n_bics = len(bicliques)
        
        for i in range(n_bics):
            L1, R1 = bicliques[i]
            
            # Try moving each vertex in L1 to another biclique
            for v in list(L1):
                for j in range(n_bics):
                    if j == i:
                        continue
                    L2, R2 = bicliques[j]
                    
                    # Try moving v from L1 to L2
                    new_L1 = L1 - {v}
                    new_L2 = L2 | {v}
                    if new_L1 and not (new_L1 & R1) and not (new_L2 & R2):
                        # Check validity
                        valid1 = True
                        for l in new_L1:
                            for r in R1:
                                if make_edge(l, r) not in edge_set:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        valid2 = True
                        for l in new_L2:
                            for r in R2:
                                if make_edge(l, r) not in edge_set:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        
                        if valid1 and valid2:
                            result = list(bicliques)
                            result[i] = (frozenset(new_L1), frozenset(R1))
                            result[j] = (frozenset(new_L2), frozenset(R2))
                            # Try to merge after reassignment
                            merged = try_merge(result, edge_set)
                            if merged is not None:
                                return merged
                    
                    # Try moving v from L1 to R2
                    new_L1 = L1 - {v}
                    new_R2 = R2 | {v}
                    if new_L1 and not (new_L1 & R1) and not (L2 & new_R2):
                        valid1 = True
                        for l in new_L1:
                            for r in R1:
                                if make_edge(l, r) not in edge_set:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        valid2 = True
                        for l in L2:
                            for r in new_R2:
                                if make_edge(l, r) not in edge_set:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        
                        if valid1 and valid2:
                            result = list(bicliques)
                            result[i] = (frozenset(new_L1), frozenset(R1))
                            result[j] = (frozenset(L2), frozenset(new_R2))
                            merged = try_merge(result, edge_set)
                            if merged is not None:
                                return merged
            
            # Try moving each vertex in R1 to another biclique
            for v in list(R1):
                for j in range(n_bics):
                    if j == i:
                        continue
                    L2, R2 = bicliques[j]
                    
                    # Try moving v from R1 to R2
                    new_R1 = R1 - {v}
                    new_R2 = R2 | {v}
                    if new_R1 and not (L1 & new_R1) and not (L2 & new_R2):
                        valid1 = True
                        for l in L1:
                            for r in new_R1:
                                if make_edge(l, r) not in edge_set:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        valid2 = True
                        for l in L2:
                            for r in new_R2:
                                if make_edge(l, r) not in edge_set:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        
                        if valid1 and valid2:
                            result = list(bicliques)
                            result[i] = (frozenset(L1), frozenset(new_R1))
                            result[j] = (frozenset(L2), frozenset(new_R2))
                            merged = try_merge(result, edge_set)
                            if merged is not None:
                                return result if merged is None else merged
                    
                    # Try moving v from R1 to L2
                    new_R1 = R1 - {v}
                    new_L2 = L2 | {v}
                    if new_R1 and not (L1 & new_R1) and not (new_L2 & R2):
                        valid1 = True
                        for l in L1:
                            for r in new_R1:
                                if make_edge(l, r) not in edge_set:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        valid2 = True
                        for l in new_L2:
                            for r in R2:
                                if make_edge(l, r) not in edge_set:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        
                        if valid1 and valid2:
                            result = list(bicliques)
                            result[i] = (frozenset(L1), frozenset(new_R1))
                            result[j] = (frozenset(new_L2), frozenset(R2))
                            merged = try_merge(result, edge_set)
                            if merged is not None:
                                return merged

        return None

    def validate_result(result, edge_set):
        """Validate the result."""
        covered_edges = set()
        for bc in result:
            if not isinstance(bc, list) or len(bc) != 2:
                return False
            left, right = bc
            if not isinstance(left, list) or not isinstance(right, list):
                return False
            if len(left) == 0 or len(right) == 0:
                return False
            if set(left) & set(right):
                return False
            for l in left:
                if not isinstance(l, int) or isinstance(l, bool):
                    return False
                for r in right:
                    if not isinstance(r, int) or isinstance(r, bool):
                        return False
                    if l == r:
                        return False
                    e = make_edge(l, r)
                    if e not in edge_set:
                        return False
                    if e in covered_edges:
                        return False
                    covered_edges.add(e)
        return covered_edges == edge_set

    # Run multiple strategies
    best_result = None
    best_count = total_edges + 1

    # Strategy 1: Deterministic greedy
    bicliques = greedy_partition(edge_set, rng=None, max_seeds=60)
    if len(bicliques) < best_count:
        best_count = len(bicliques)
        best_result = list(bicliques)

    # Strategy 2-8: Randomized runs with different seeds and seed counts
    for trial in range(7):
        if time_left() < 5.0:
            break
        
        rng_trial = random.Random(42 + trial * 7)
        max_seeds = 40 + (trial % 3) * 20  # Vary seed count
        
        bicliques_t = greedy_partition(edge_set, rng=rng_trial, max_seeds=max_seeds)
        
        if len(bicliques_t) < best_count:
            best_count = len(bicliques_t)
            best_result = list(bicliques_t)

    # Try to improve via merging and vertex reassignment
    improved = True
    while improved and time_left() > 3.0:
        improved = False
        
        # Try direct merge
        merged = try_merge(best_result, edge_set)
        if merged is not None:
            best_result = merged
            best_count = len(merged)
            improved = True
            continue
        
        # Try vertex reassignment
        reassigned = try_vertex_reassign(best_result, edge_set)
        if reassigned is not None:
            best_result = reassigned
            best_count = len(reassigned)
            improved = True

    # Final conversion to required format
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    # Validate the result
    if not validate_result(result, edge_set):
        # Fallback: one biclique per edge
        result = []
        for u, v in sorted(edge_set):
            result.append([[u], [v]])

    return result
