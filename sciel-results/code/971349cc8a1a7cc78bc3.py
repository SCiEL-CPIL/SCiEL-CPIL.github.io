import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin for 60s limit

    N = 256
    squares = [i * i for i in range(1, 23)]  # 1^2 to 22^2 = 484
    square_set = set(squares)

    # 1. Identify valid vertices (2*v not a perfect square)
    valid_vertices = []
    for i in range(1, N + 1):
        if 2 * i not in square_set:
            valid_vertices.append(i)

    M = len(valid_vertices)
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # 2. Build adjacency bitmask
    adj = [0] * M
    for i, u in enumerate(valid_vertices):
        for k in squares:
            v = k - u
            if 1 <= v <= N and v in v_to_idx:
                j = v_to_idx[v]
                if j != i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    # 3. Greedy initial solution
    def greedy_mis(adj_masks, active_mask):
        selected = []
        current_active = active_mask
        while current_active:
            min_deg = float('inf')
            best_v = -1
            temp = current_active
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj_masks[v] & current_active).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp -= lsb
            if best_v == -1:
                break
            selected.append(best_v)
            to_remove = adj_masks[best_v] | (1 << best_v)
            current_active &= ~to_remove
        return selected

    initial_selected = greedy_mis(adj, (1 << M) - 1)
    best_size = len(initial_selected)
    best_set_global = set(initial_selected)

    # 4. Coloring-based upper bound for MIS
    # Greedy coloring: the number of colors is an upper bound on alpha
    # because each color class is an independent set, and alpha >= max color class size
    # But actually we need: alpha <= number of colors? No!
    # The correct bound: if we can color with k colors, then alpha <= k is FALSE.
    # 
    # Actually, the standard approach for MIS B&B:
    # We color the graph. The number of colors k gives: alpha <= k? NO.
    # 
    # Let me think again. The key insight for B&B on MIS:
    # If we color the graph with k colors, then any independent set can have at most
    # one vertex from each color class? NO, that's for clique.
    #
    # For MIS: a valid upper bound is obtained by noting that if we can partition
    # vertices into k independent sets (coloring), then... that doesn't bound alpha.
    #
    # The CORRECT bound used in practice:
    # Use a greedy coloring where we assign colors in order of decreasing degree.
    # The number of colors k is NOT an upper bound on alpha.
    #
    # However, there IS a valid bound: 
    # If we compute a maximum matching M in the graph, then alpha <= |V| - |M|
    # (since each edge in a matching removes at most 1 vertex from an independent set... 
    #  actually no, each edge in a matching means at most 1 endpoint can be in the IS)
    # So alpha <= |V| - |matching|
    #
    # For a simpler but effective bound, let's use:
    # Greedy coloring bound: sort by degree, assign colors. The number of colors
    # in a proper coloring is an upper bound on the clique number, not alpha.
    #
    # Let me use a different approach: the "Lagrangian" or "Turán" type bound.
    # 
    # Actually, the simplest valid upper bound that's better than popcount:
    # For each vertex v in candidates, at most 1 of {v, u} can be in IS for each edge (v,u).
    # A simple bound: alpha <= sum over v of 1/(deg(v)+1) ... no.
    #
    # Let's use: alpha <= |V| - maximum_matching_size
    # Computing max matching is expensive for general graphs.
    #
    # Alternative: use a greedy independent set as a LOWER bound (to update best),
    # and use popcount as upper bound, but with smarter branching.
    #
    # For the bound, let's try: 
    # Upper bound = size of maximum independent set in a subgraph approximation.
    # 
    # Actually, let's use the following valid bound:
    # If we can color the graph with k colors, then alpha(G) <= k is WRONG.
    # But alpha(G) >= max_color_class_size.
    #
    # The standard B&B for MIS uses:
    # 1. A lower bound from a greedy heuristic
    # 2. An upper bound from... typically the number of vertices remaining (popcount)
    #    or from a coloring of the complement graph.
    #
    # Let me just use popcount but with much better branching strategy and 
    # local search to improve the initial solution significantly.

    def local_search_improve(selected_set):
        """Try to improve the independent set by adding compatible vertices."""
        selected = set(selected_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any excluded vertex that's compatible with all selected
            all_indices = set(range(M))
            excluded = all_indices - selected
            for w in sorted(excluded, key=lambda x: (adj[x] & ((1 << M) - 1)).bit_count()):
                # Check if w is compatible with all selected vertices
                can_add = True
                for u in selected:
                    if adj[u] & (1 << w):
                        can_add = False
                        break
                if can_add:
                    selected.add(w)
                    improved = True
                    break
            if not improved:
                # Try 1-swap: remove v, add w if w was blocked only by v
                for v in list(selected):
                    blocked_by_v = adj[v] & ((1 << M) - 1)
                    for w in sorted(blocked_by_v, key=lambda x: (adj[x] & ((1 << M) - 1)).bit_count()):
                        if w in selected:
                            continue
                        # Check if w is compatible with all selected except v
                        compatible_except_v = True
                        for u in selected:
                            if u != v and (adj[u] & (1 << w)):
                                compatible_except_v = False
                                break
                        if compatible_except_v:
                            selected.discard(v)
                            selected.add(w)
                            improved = True
                            break
                    if improved:
                        break
        return selected

    # 5. Branch and Bound with better branching
    def search(candidates_mask, current_count, current_set):
        nonlocal best_size, best_set_global
        nonlocal deadline

        if time.time() > deadline:
            return

        if candidates_mask == 0:
            if current_count > best_size:
                best_size = current_count
                best_set_global = set(current_set)
            return

        # Prune with popcount bound
        if current_count + candidates_mask.bit_count() <= best_size:
            return

        # Find vertex with highest degree in candidates_mask for branching
        max_deg = -1
        best_v = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp -= lsb

        if best_v == -1:
            return

        # Branch 1: Exclude best_v (explore first for faster bound tightening)
        new_candidates_excl = candidates_mask & ~(1 << best_v)
        search(new_candidates_excl, current_count, current_set)

        # Branch 2: Include best_v
        new_candidates_incl = candidates_mask & ~(adj[best_v] | (1 << best_v))
        current_set.append(best_v)
        search(new_candidates_incl, current_count + 1, current_set)
        current_set.pop()

    # Start search
    search((1 << M) - 1, 0, [])

    # 6. Local search polish - run multiple passes
    if time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            improved = local_search_improve(best_set_global)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set_global = improved
            else:
                break

    # Map back to original values
    result = [valid_vertices[i] for i in best_set_global]

    # Verify
    res_set = set(result)
    valid = True
    for a in res_set:
        for b in res_set:
            if a <= b:
                if (a + b) in square_set:
                    valid = False
                    break
        if not valid:
            break
    if not valid:
        # Safe fallback: all odd numbers (sum of two odds is even, check if square)
        # Actually, use a verified small set
        fallback = [i for i in range(1, 257) if i % 2 == 1]
        # Verify fallback
        fb_valid = True
        for a in fallback:
            for b in fallback:
                if a <= b and (a + b) in square_set:
                    fb_valid = False
                    break
            if not fb_valid:
                break
        if fb_valid:
            return fallback
        else:
            # Minimal valid set
            return [1]
    return result

if __name__ == "__main__":
    construct()
