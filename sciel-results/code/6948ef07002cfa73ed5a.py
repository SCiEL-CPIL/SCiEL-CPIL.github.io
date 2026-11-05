"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128

    # Pre-compute lcms
    lcms = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcms[(i, j)] = lcm(i, j)

    # Build conflict graph: edge (u,v) if exists w != u,v with lcm(u,v)=lcm(u,w)=lcm(v,w)
    # Use bitmasks for speed
    conflict_mask = [0] * (N + 1)  # conflict_mask[v] is a bitmask of neighbors of v

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcms[(a, b)]
            for c in range(b + 1, N + 1):
                if lcms[(a, c)] == lab and lcms[(b, c)] == lab:
                    # {a,b,c} is a bad triple
                    conflict_mask[a] |= (1 << b) | (1 << c)
                    conflict_mask[b] |= (1 << a) | (1 << c)
                    conflict_mask[c] |= (1 << a) | (1 << b)

    # Precompute neighbor lists
    neighbors = [0] * (N + 1)
    for v in range(1, N + 1):
        neighbors[v] = conflict_mask[v]

    # Greedy coloring of a candidate set (bitmask) to get upper bound
    # Returns number of colors needed (each color class is a valid set)
    def greedy_color_bound(cand_mask):
        """Upper bound on max independent set via greedy coloring."""
        colors = 0
        remaining = cand_mask
        while remaining:
            colors += 1
            # Greedy: pick vertices in order, assign to color if no conflict with same-color
            color_set = 0
            for v in range(1, N + 1):
                bit = 1 << v
                if remaining & bit:
                    if not (conflict_mask[v] & color_set):
                        color_set |= bit
                        remaining &= ~bit
            if not remaining:
                break
        return colors

    # Greedy independent set
    def greedy_independent_set(cand_mask, order=None):
        if order is None:
            order = list(range(1, N + 1))
        selected = 0
        for v in order:
            bit = 1 << v
            if cand_mask & bit and not (conflict_mask[v] & selected):
                selected |= bit
        return selected

    # Try multiple greedy orderings to get a good initial solution
    best_solution = 0
    best_size = 0

    # Degree-based greedy (ascending degree first)
    order_deg = sorted(range(1, N + 1), key=lambda v: bin(conflict_mask[v]).count('1'))
    s = greedy_independent_set((1 << (N + 1)) - 2, order_deg)  # bits 1..N
    if bin(s).count('1') > best_size:
        best_size = bin(s).count('1')
        best_solution = s

    # Randomized greedy with multiple restarts
    random.seed(42)
    all_mask = (1 << (N + 1)) - 2  # bits 1 through N set

    if time.time() < deadline - 15:
        for restart in range(300):
            if time.time() > deadline - 15:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Weighted sort: low degree first with randomness
            weight = 0.3 + random.random() * 0.5
            order.sort(key=lambda v: (bin(conflict_mask[v]).count('1') * weight + random.random() * (1 - weight)))
            s = greedy_independent_set(all_mask, order)
            sz = bin(s).count('1')
            if sz > best_size:
                best_size = sz
                best_solution = s

    # Local search: try to add more elements
    def try_add_more(selected_mask):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in range(1, N + 1):
                bit = 1 << v
                if not (selected_mask & bit) and not (conflict_mask[v] & selected_mask):
                    selected_mask |= bit
                    improved = True
        return selected_mask

    best_solution = try_add_more(best_solution)
    best_size = bin(best_solution).count('1')

    # Local search: 1-swap improvements
    def local_search_1swap(selected_mask):
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            selected_list = [v for v in range(1, N + 1) if selected_mask & (1 << v)]
            unselected = [v for v in range(1, N + 1) if not (selected_mask & (1 << v))]

            best_swap_mask = None
            best_swap_size = best_size

            for v_out in selected_list:
                if time.time() > deadline - 1:
                    break
                temp_mask = selected_mask & ~(1 << v_out)
                for v_in in unselected:
                    bit_in = 1 << v_in
                    if not (conflict_mask[v_in] & temp_mask):
                        new_mask = temp_mask | bit_in
                        new_mask = try_add_more(new_mask)
                        sz = bin(new_mask).count('1')
                        if sz > best_swap_size:
                            best_swap_size = sz
                            best_swap_mask = new_mask
            if best_swap_mask is not None:
                selected_mask = best_swap_mask
                best_size = best_swap_size
                improved = True
        return selected_mask

    if time.time() < deadline - 10:
        best_solution = local_search_1swap(best_solution)
        best_solution = try_add_more(best_solution)
        best_size = bin(best_solution).count('1')

    # Branch and Bound with 2-coloring upper bound
    # State: (cand_mask, current_mask, current_size)
    # cand_mask: vertices still to decide
    # We branch on a vertex: include it or exclude it
    
    # Precompute: for each vertex, its "forbidden" set given current selection
    # A vertex v can be added to current_mask iff conflict_mask[v] & current_mask == 0

    # B&B using the coloring bound
    def branch_and_bound(cand_mask, current_mask, current_size, best_ref):
        """Returns (best_mask, best_size) found in this subtree."""
        if time.time() > deadline - 1:
            return current_mask, current_size

        # If no candidates left
        if cand_mask == 0:
            if current_size > best_ref[0]:
                return current_mask, current_size
            return current_mask, current_size

        # Upper bound: current_size + greedy_color_bound(cand_mask)
        # But we can tighten: only vertices compatible with current_mask can be added
        # Filter cand_mask to only those compatible
        compatible = 0
        for v in range(1, N + 1):
            bit = 1 << v
            if cand_mask & bit and not (conflict_mask[v] & current_mask):
                compatible |= bit

        if compatible == 0:
            if current_size > best_ref[0]:
                return current_mask, current_size
            return current_mask, current_size

        # Upper bound via coloring
        ub = current_size + greedy_color_bound(compatible)
        if ub <= best_ref[0]:
            return current_mask, current_size

        # Choose branching vertex: pick one with maximum degree in the induced subgraph
        # (or just the first one for simplicity, but max degree is better)
        branch_v = -1
        max_deg = -1
        for v in range(1, N + 1):
            bit = 1 << v
            if compatible & bit:
                deg = bin(conflict_mask[v] & compatible).count('1')
                if deg > max_deg:
                    max_deg = deg
                    branch_v = v

        if branch_v == -1:
            return current_mask, current_size

        bit_b = 1 << branch_v

        # Branch 1: Include branch_v
        new_current = current_mask | bit_b
        new_cand = compatible & ~bit_b
        # Remove from new_cand all vertices conflicting with branch_v
        new_cand &= ~conflict_mask[branch_v]
        mask1, size1 = branch_and_bound(new_cand, new_current, current_size + 1, best_ref)
        if size1 > best_ref[0]:
            best_ref[0] = size1
            best_ref[1] = mask1

        # Branch 2: Exclude branch_v
        new_cand2 = compatible & ~bit_b
        mask2, size2 = branch_and_bound(new_cand2, current_mask, current_size, best_ref)
        if size2 > best_ref[0]:
            best_ref[0] = size2
            best_ref[1] = mask2

        return best_ref[1], best_ref[0]

    if time.time() < deadline - 5:
        best_ref = [best_size, best_solution]
        # Start B&B with all vertices as candidates
        _, bb_size = branch_and_bound(all_mask, 0, 0, best_ref)
        if best_ref[0] > best_size:
            best_size = best_ref[0]
            best_solution = best_ref[1]

    # Final: ensure non-empty and return
    if bin(best_solution).count('1') == 0:
        best_solution = 1 << 1  # just {1}

    result = [v for v in range(1, N + 1) if best_solution & (1 << v)]
    if not result:
        result = [1]

    return result

# EVOLVE-BLOCK-END
