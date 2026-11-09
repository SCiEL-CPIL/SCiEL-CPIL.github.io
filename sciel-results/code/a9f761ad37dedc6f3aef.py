import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)

    # Edge list with indices and edge -> vertex incidence
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    edges_sorted = sorted(edge_set)
    m = len(edges_sorted)
    edge_idx = {}
    for i, e in enumerate(edges_sorted):
        edge_idx[e] = i

    # Vertex adjacency as sets
    adj_set = [set() for _ in range(n)]
    for u, v in edges_sorted:
        adj_set[u].add(v)
        adj_set[v].add(u)

    # Edge set as a frozenset for fast ops
    all_edges = frozenset(edges_sorted)
    all_edges_set = set(edges_sorted)

    def edges_of_biclique(L, R):
        """Return the set of edges (as tuples) fully inside L x R."""
        es = set()
        for u in L:
            for v in R:
                if u < v:
                    if (u, v) in all_edges_set:
                        es.add((u, v))
                else:
                    if (v, u) in all_edges_set:
                        es.add((v, u))
        return es

    def expand_biclique(L, R, remaining_edges):
        """Expand L and R such that all edges between L and R are in remaining_edges.
        Returns (L, R) as lists."""
        L = set(L)
        R = set(R)
        # Ensure no self-loop
        L -= R
        # Iteratively add vertices that can extend without violating
        changed = True
        while changed:
            changed = False
            # Try to add each candidate to L or R
            for w in range(n):
                if w in L or w in R:
                    continue
                # Check if w can be added to L: all R vertices must be connected to w in remaining_edges
                can_add_L = True
                for v in R:
                    if w < v:
                        if (w, v) not in remaining_edges:
                            can_add_L = False
                            break
                    else:
                        if (v, w) not in remaining_edges:
                            can_add_L = False
                            break
                if can_add_L and R:
                    L.add(w)
                    changed = True
                    continue
                # Check if w can be added to R
                can_add_R = True
                for u in L:
                    if u < w:
                        if (u, w) not in remaining_edges:
                            can_add_R = False
                            break
                    else:
                        if (w, u) not in remaining_edges:
                            can_add_R = False
                            break
                if can_add_R and L:
                    R.add(w)
                    changed = True
        # Now verify all L-R pairs are in remaining_edges
        for u in L:
            for v in R:
                if u < v:
                    if (u, v) not in remaining_edges:
                        # Remove v from R
                        R.discard(v)
                else:
                    if (v, u) not in remaining_edges:
                        R.discard(v)
        # Clean L and R
        L_list = sorted(L)
        R_list = sorted(R)
        return L_list, R_list

    # Generate a diverse set of candidate bicliques
    def generate_candidates(seed_edges, num_candidates=200):
        """Generate candidate bicliques from the seed graph."""
        candidates = []
        seen = set()
        
        # Strategy 1: Expand from each edge
        for i in range(min(num_candidates, len(edges_sorted))):
            if i % 2 == 0:
                break
            e = edges_sorted[i]
            L, R = expand_biclique([e[0]], [e[1]], all_edges_set)
            if L and R:
                key = (frozenset(L), frozenset(R))
                if key not in seen:
                    seen.add(key)
                    candidates.append((L, R))
            else:
                # Try swapped
                L, R = expand_biclique([e[1]], [e[0]], all_edges_set)
                if L and R:
                    key = (frozenset(L), frozenset(R))
                    if key not in seen:
                        seen.add(key)
                        candidates.append((L, R))
        
        # Strategy 2: Expand from random pairs of adjacent vertices
        for i in range(min(num_candidates, 50)):
            e = edges_sorted[i * 3 % m]
            L, R = expand_biclique([e[0]], [e[1]], all_edges_set)
            if L and R:
                key = (frozenset(L), frozenset(R))
                if key not in seen:
                    seen.add(key)
                    candidates.append((L, R))
        
        # Strategy 3: Single-vertex bicliques (star)
        for v in range(n):
            if adj_set[v]:
                L = [v]
                R = sorted(adj_set[v])
                # Check all edges exist
                valid = True
                for u in R:
                    if v < u:
                        if (v, u) not in all_edges_set:
                            valid = False
                            break
                    else:
                        if (u, v) not in all_edges_set:
                            valid = False
                            break
                if valid:
                    key = (frozenset(L), frozenset(R))
                    if key not in seen:
                        seen.add(key)
                        candidates.append((L, R))
        
        return candidates

    def greedy_set_cover(candidates, all_edges_list):
        """Greedy set cover: repeatedly pick the candidate covering the most uncovered edges."""
        uncovered = set(all_edges_list)
        selected = []
        
        while uncovered:
            best = None
            best_count = -1
            for L, R in candidates:
                covered = edges_of_biclique(L, R)
                new_cover = len(covered & uncovered)
                if new_cover > best_count:
                    best_count = new_cover
                    best = (L, R)
            if best is None or best_count <= 0:
                # Fallback: pick a single edge
                e = next(iter(uncovered))
                selected.append(([e[0]], [e[1]]))
                uncovered.discard(e)
            else:
                L, R = best
                covered = edges_of_biclique(L, R)
                selected.append((L, R))
                uncovered -= covered
        return selected

    def is_valid_biclique_partition(bicliques):
        """Check that the partition is valid: covers all edges exactly once, all cross-edges exist."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    if u == v:
                        return False
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return False
                    if e not in all_edges_set:
                        return False
                    covered.add(e)
        return covered == all_edges_set

    # Generate candidates
    start_time = time.time()
    deadline = start_time + 70  # Safety margin

    candidates = generate_candidates(all_edges_set, 300)
    
    # Initial greedy solution
    solution = greedy_set_cover(candidates, edges_sorted)
    
    # Local search: try to improve by replacing bicliques
    def local_search(solution, candidates):
        """Try 2-exchange: replace two bicliques with one (merge) or replace one with two."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try merging pairs
            for i in range(len(solution)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(solution)):
                    L1, R1 = solution[i]
                    L2, R2 = solution[j]
                    # Try merged
                    L_new = list(set(L1 + L2))
                    R_new = list(set(R1 + R2))
                    # Check if valid
                    valid = True
                    for u in L_new:
                        for v in R_new:
                            if u == v:
                                valid = False
                                break
                            e = (min(u, v), max(u, v))
                            if e not in all_edges_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Check coverage: the merged biclique must cover exactly the union of edges
                        edges_i = edges_of_biclique(L1, R1)
                        edges_j = edges_of_biclique(L2, R2)
                        edges_new = edges_of_biclique(L_new, R_new)
                        if edges_new == (edges_i | edges_j):
                            # Do the merge
                            new_solution = solution[:i] + solution[i+1:j] + solution[j+1:]
                            new_solution.insert(i, (L_new, R_new))
                            if is_valid_biclique_partition(new_solution):
                                solution = new_solution
                                improved = True
                                break
                if improved:
                    break
        return solution

    solution = local_search(solution, candidates)
    
    # Additional local search: try to replace a biclique with a smaller set
    def local_search2(solution, candidates):
        """Try to split one biclique into two, or replace one biclique with a different one."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(solution)):
                if time.time() > deadline:
                    break
                L_i, R_i = solution[i]
                edges_i = edges_of_biclique(L_i, R_i)
                # Try to cover edges_i with two candidate bicliques
                best_pair = None
                best_count = -1
                for L_a, R_a in candidates:
                    covered_a = edges_of_biclique(L_a, R_a)
                    if covered_a & edges_i != covered_a:
                        continue  # Must be subset
                    remaining = edges_i - covered_a
                    if not remaining:
                        continue  # Already covered by one
                    for L_b, R_b in candidates:
                        covered_b = edges_of_biclique(L_b, R_b)
                        if covered_b & remaining != covered_b:
                            continue
                        if covered_a | covered_b == edges_i:
                            count = 2
                            if count < best_count:
                                best_count = count
                                best_pair = ((L_a, R_a), (L_b, R_b))
                # This is O(n^2) per biclique, which is expensive.
                # Skip for now.
        return solution

    # Convert to required format
    final_result = []
    for L, R in solution:
        l = [int(x) for x in L]
        r = [int(x) for x in R]
        final_result.append([l, r])
    
    return final_result
