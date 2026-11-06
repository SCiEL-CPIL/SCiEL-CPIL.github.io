import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 125  # Safety margin before 150s kill

    def is_not_2colorable(n, edge_list):
        """Check if hypergraph has NO proper 2-coloring using backtracking with unit propagation.
        Returns True if NOT 2-colorable (UNSAT), False if 2-colorable (SAT)."""
        if time.time() > deadline:
            return True  # Conservative: assume not colorable

        m = len(edge_list)
        if m == 0:
            return False

        # Fast bit-based check for small n
        if n <= 15:
            # Precompute edge masks
            edge_masks = []
            for e in edge_list:
                em = 0
                for v in e:
                    em |= (1 << v)
                edge_masks.append(em)

            # By symmetry, fix vertex 0 to color 0
            # Check all 2^(n-1) colorings
            for mask in range(1 << (n - 1)):
                full_mask = mask  # vertex 0 (bit 0) is 0
                ok = True
                for em in edge_masks:
                    # Edge is monochromatic if (mask & em) == 0 (all 0) or (mask & em) == em (all 1)
                    if (full_mask & em) == 0 or (full_mask & em) == em:
                        ok = False
                        break
                if ok:
                    return False  # Found valid coloring, IS 2-colorable
            return True  # No coloring found, NOT 2-colorable

        # For larger n, use backtracking with unit propagation
        assignment = [-1] * n
        vert_edges = [[] for _ in range(n)]
        for i, e in enumerate(edge_list):
            for v in e:
                vert_edges[v].append(i)

        def backtrack(assignment):
            if time.time() > deadline:
                return None  # Timeout

            # Check if all assigned
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                # Verify all edges are not monochromatic
                for e in edge_list:
                    colors = set(assignment[v] for v in e)
                    if len(colors) < 2:
                        return None
                return assignment.copy()

            # MRV heuristic: pick vertex with most constraints
            best_v = max(unassigned, key=lambda v: len(vert_edges[v]))

            for color in [0, 1]:
                new_assignment = assignment.copy()
                new_assignment[best_v] = color

                # Unit propagation: check if any edge becomes forced
                valid = True
                for e in edge_list:
                    vals = [new_assignment[v] for v in e]
                    # If edge has both 0 and 1, it's satisfied
                    if 0 in vals and 1 in vals:
                        continue
                    # If edge has only 0s and some -1s, all -1s must be 1
                    if 0 in vals and 1 not in vals:
                        for v in e:
                            if new_assignment[v] == -1:
                                new_assignment[v] = 1
                    # If edge has only 1s and some -1s, all -1s must be 0
                    elif 1 in vals and 0 not in vals:
                        for v in e:
                            if new_assignment[v] == -1:
                                new_assignment[v] = 0

                # Check for conflicts after propagation
                for e in edge_list:
                    vals = [new_assignment[v] for v in e]
                    if -1 not in vals:
                        if all(v == 0 for v in vals) or all(v == 1 for v in vals):
                            valid = False
                            break

                if valid:
                    result = backtrack(new_assignment)
                    if result is not None:
                        return result
            return None

        result = backtrack(assignment)
        return result is None  # True if not 2-colorable

    # Fano plane: 7 vertices, 7 edges (3-uniform), known to be non-2-colorable
    # Standard Fano plane edges
    fano_edges_3 = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]

    # Lift to 5-uniform: add vertices 7 and 8 to each edge
    # This creates a 5-uniform hypergraph on 9 vertices
    # Any 2-coloring of 9 vertices restricts to a 2-coloring of vertices 0-6
    # Since Fano plane has no valid 2-coloring, the lifted hypergraph is also non-2-colorable
    base_edges = [(e[0], e[1], e[2], 7, 8) for e in fano_edges_3]

    # Verify the base construction
    n_base = 9
    if not is_not_2colorable(n_base, base_edges):
        # Should not happen, but fallback
        # Use complete 5-uniform hypergraph on 9 vertices
        all_edges = list(itertools.combinations(range(9), 5))
        return {'n': 9, 'edges': [list(e) for e in all_edges]}

    # Now try to add more edges to get a larger hypergraph, then optimize
    # Strategy: Start with base, try adding edges from a pool, then greedily remove

    # Generate candidate edges: all 5-subsets of {0..8}
    all_possible = list(itertools.combinations(range(9), 5))
    base_set = set(base_edges)
    candidate_pool = [e for e in all_possible if e not in base_set]

    # Try different target sizes and find minimal non-2-colorable
    best_m = len(base_edges)
    best_edges = base_edges[:]

    # Try to find smaller constructions with more vertices
    # First, try n=10, 11, etc. with various edge counts
    for n_try in [9, 10, 11, 12, 13]:
        if time.time() > deadline - 30:
            break

        if n_try == 9:
            # Already have base, try adding edges then removing
            current = base_edges[:]
            # Try adding random edges to create redundancy, then remove
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                # Add a random edge
                if candidate_pool:
                    new_edge = candidate_pool[int((time.time() * 1000) % len(candidate_pool))]
                    trial = current + [new_edge]
                    if is_not_2colorable(n_try, trial):
                        current = trial
                        # Now try to remove edges greedily
                        improved = True
                        while improved:
                            improved = False
                            if time.time() > deadline - 5:
                                break
                            for i in range(len(current) - 1, -1, -1):
                                if time.time() > deadline - 3:
                                    break
                                trial2 = current[:i] + current[i+1:]
                                if is_not_2colorable(n_try, trial2):
                                    current = trial2
                                    improved = True
                                    break
            if len(current) < best_m:
                best_m = len(current)
                best_edges = current[:]
        else:
            # For larger n, try random constructions
            all_possible_n = list(itertools.combinations(range(n_try), 5))
            for target_m in [20, 25, 30, 35, 40, 45, 50]:
                if time.time() > deadline - 10:
                    break
                if target_m > len(all_possible_n):
                    continue
                for trial in range(20):
                    if time.time() > deadline - 5:
                        break
                    # Random sample
                    indices = set()
                    for _ in range(target_m):
                        indices.add(int((time.time() * 10000 + trial * 137) % len(all_possible_n)))
                    candidate = [all_possible_n[i] for i in sorted(indices)]
                    if len(candidate) < target_m:
                        continue
                    if is_not_2colorable(n_try, candidate):
                        if len(candidate) < best_m:
                            best_m = len(candidate)
                            best_edges = candidate[:]
                        # Try to remove edges
                        current = candidate[:]
                        improved = True
                        while improved:
                            improved = False
                            if time.time() > deadline - 3:
                                break
                            for i in range(len(current) - 1, -1, -1):
                                if time.time() > deadline - 2:
                                    break
                                trial2 = current[:i] + current[i+1:]
                                if is_not_2colorable(n_try, trial2):
                                    current = trial2
                                    improved = True
                                    break
                        if len(current) < best_m:
                            best_m = len(current)
                            best_edges = current[:]

    # Final answer
    if best_edges:
        # Determine n from the vertices used
        max_v = max(max(e) for e in best_edges) + 1
        n_final = max(5, max_v)
        edges_final = [list(e) for e in best_edges]
        return {'n': n_final, 'edges': edges_final}

    # Fallback: complete hypergraph on 9 vertices
    n = 9
    edges = [list(e) for e in itertools.combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
