"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build host adjacency bitmask
    host_adj = [0] * n
    for u, v in edges:
        host_adj[u] |= (1 << v)
        host_adj[v] |= (1 << u)

    # Total edge count
    m = len(edges)

    def make_mask(vertices):
        mask = 0
        for v in vertices:
            mask |= (1 << v)
        return mask

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    # --- Phase 1: Generate candidate maximal bicliques ---
    # We'll use a greedy approach to find a set of large bicliques that cover all edges.
    # Then we'll use a set cover heuristic to pick the minimum subset.
    
    # First, let's find a greedy partition as in the current program, but store it as a set of bicliques.
    remaining_edges = set(frozenset((u, v)) for u, v in edges)
    
    def find_max_biclique():
        """Find a large biclique in remaining_edges using bitmask ops."""
        if not remaining_edges:
            return None

        # Compute degrees in remaining graph
        deg = [0] * n
        for e in remaining_edges:
            u, v = tuple(e)
            deg[u] += 1
            deg[v] += 1

        # Pick vertex with max degree
        anchor = max(range(n), key=lambda i: deg[i])
        if deg[anchor] == 0:
            return None

        # Right side starts as neighbors of anchor in remaining graph
        right_mask = 0
        for e in remaining_edges:
            u, v = tuple(e)
            if u == anchor:
                right_mask |= (1 << v)
            elif v == anchor:
                right_mask |= (1 << u)

        left_mask = (1 << anchor)

        # Greedy expansion: try to add vertices to left that connect to all right
        # And vertices to right that connect to all left
        changed = True
        while changed:
            changed = False
            # Try adding to left
            for v in range(n):
                if (left_mask >> v) & 1:
                    continue
                if (right_mask >> v) & 1:
                    continue
                # Check if v connects to all in right_mask within remaining_edges
                ok = True
                r = right_mask
                while r:
                    w = (r & -r).bit_length() - 1
                    r &= r - 1
                    if frozenset((v, w)) not in remaining_edges:
                        ok = False
                        break
                if ok:
                    left_mask |= (1 << v)
                    changed = True
            # Try adding to right
            for v in range(n):
                if (left_mask >> v) & 1:
                    continue
                if (right_mask >> v) & 1:
                    continue
                ok = True
                l = left_mask
                while l:
                    w = (l & -l).bit_length() - 1
                    l &= l - 1
                    if frozenset((v, w)) not in remaining_edges:
                        ok = False
                        break
                if ok:
                    right_mask |= (1 << v)
                    changed = True

        if left_mask and right_mask:
            return (left_mask, right_mask)
        return None

    greedy_bicliques = []
    while remaining_edges and (time.time() - t0) < deadline:
        bc = find_max_biclique()
        if bc is None:
            # Fallback: single edges
            for e in list(remaining_edges):
                u, v = tuple(e)
                greedy_bicliques.append((1 << u, 1 << v))
                remaining_edges.discard(e)
            break

        left_mask, right_mask = bc
        # Verify and remove
        valid = True
        for u in range(n):
            if not ((left_mask >> u) & 1):
                continue
            r = right_mask
            while r:
                v = (r & -r).bit_length() - 1
                r &= r - 1
                if frozenset((u, v)) not in remaining_edges:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            for e in list(remaining_edges):
                u, v = tuple(e)
                greedy_bicliques.append((1 << u, 1 << v))
                remaining_edges.discard(e)
            break

        for u in range(n):
            if not ((left_mask >> u) & 1):
                continue
            r = right_mask
            while r:
                v = (r & -r).bit_length() - 1
                r &= r - 1
                remaining_edges.discard(frozenset((u, v)))

        greedy_bicliques.append((left_mask, right_mask))

    # Final fallback
    for e in remaining_edges:
        u, v = tuple(e)
        greedy_bicliques.append((1 << u, 1 << v))

    # --- Phase 2: Set Cover with Randomized Local Search ---
    # We have a set of bicliques that cover all edges. We want to find the minimum subset.
    # This is a Set Cover problem, which is NP-hard. We'll use a greedy heuristic first.
    
    # Precompute edge sets for each biclique
    def get_edge_set(bc):
        lm, rm = bc
        es = set()
        for u in range(n):
            if not ((lm >> u) & 1):
                continue
            r = rm
            while r:
                v = (r & -r).bit_length() - 1
                r &= r - 1
                es.add(frozenset((u, v)))
        return es

    # Build the set cover instance
    all_edges = set(frozenset((u, v)) for u, v in edges)
    biclique_edge_sets = [get_edge_set(bc) for bc in greedy_bicliques]
    
    # Greedy Set Cover
    selected = []
    uncovered = set(all_edges)
    
    while uncovered and (time.time() - t0) < deadline:
        best_idx = -1
        best_count = 0
        for i, es in enumerate(biclique_edge_sets):
            if i in selected:
                continue
            count = len(es & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        uncovered -= biclique_edge_sets[best_idx]

    # Initial solution from greedy set cover
    current_solution = selected
    current_count = len(current_solution)
    
    # Local Search: Try to swap one biclique for another to reduce count
    # Or try to remove a biclique if its edges are covered by others in the solution
    
    def is_valid_solution(sol):
        covered = set()
        for idx in sol:
            covered |= biclique_edge_sets[idx]
        return covered == all_edges

    # Verify initial solution
    if not is_valid_solution(current_solution):
        # Fallback to greedy_bicliques
        return [[mask_to_list(lm), mask_to_list(rm)] for lm, rm in greedy_bicliques]

    # Local search to improve
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        # Try removing each biclique
        for i in range(len(current_solution)):
            new_sol = current_solution[:i] + current_solution[i+1:]
            if is_valid_solution(new_sol):
                current_solution = new_sol
                current_count -= 1
                improved = True
                break
        
        # If no removal possible, try swapping
        if not improved:
            for i in range(len(current_solution)):
                for j in range(len(greedy_bicliques)):
                    if j in current_solution:
                        continue
                    new_sol = current_solution[:i] + current_solution[i+1:] + [j]
                    if is_valid_solution(new_sol) and len(new_sol) < current_count:
                        current_solution = new_sol
                        current_count = len(new_sol)
                        improved = True
                        break
                if improved:
                    break

    # Build final result from current_solution
    result = []
    for idx in current_solution:
        lm, rm = greedy_bicliques[idx]
        left = mask_to_list(lm)
        right = mask_to_list(rm)
        if left and right:
            result.append([left, right])

    # Type check: ensure Python ints
    for bc in result:
        bc[0] = [int(x) for x in bc[0]]
        bc[1] = [int(x) for x in bc[1]]

    return result
# EVOLVE-BLOCK-END
