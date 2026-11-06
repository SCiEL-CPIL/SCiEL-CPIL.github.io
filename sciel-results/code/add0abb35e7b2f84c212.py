"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # Build conflict: a conflicts with b if a+b is a perfect square (a != b)
    # Self-conflict: 2a is a perfect square -> a is forbidden
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}

    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N and b != a:
                if b in num_to_idx:
                    j = num_to_idx[b]
                    adj[i] |= (1 << j)

    # Also verify: for each pair, only count once (symmetric)
    # adj is symmetric by construction since if a+b=sq then b+a=sq

    def greedy_coloring(order):
        """Color the subgraph induced by 'order' using greedy coloring.
        Returns (color_count, color_of_vertex) where color_of_vertex[i] is color of vertex at position i in order."""
        n = len(order)
        colors = [0] * n
        color_of = {}  # vertex -> color
        max_color = 0
        for idx, v in enumerate(order):
            # Find smallest color not used by neighbors of v that are in order
            neighbor_colors = set()
            for pos, u in enumerate(order):
                if pos >= idx:
                    break
                if (adj[v] >> u) & 1:
                    neighbor_colors.add(colors[pos])
            c = 1
            while c in neighbor_colors:
                c += 1
            colors[idx] = c
            color_of[v] = c
            if c > max_color:
                max_color = c
        return max_color, colors

    def branch_and_bound(P, current_size, depth=0):
        """Tomita-style B&B for maximum independent set.
        P: list of vertices (candidates)
        current_size: size of current independent set
        Returns (best_size, best_set)"""
        nonlocal deadline
        if time.time() > deadline:
            return current_size, set()

        if not P:
            return current_size, set()

        # Upper bound via greedy coloring
        # Order P by degree within P (ascending)
        def degree_in(v, P_set):
            cnt = 0
            for u in P_set:
                if u != v and ((adj[v] >> u) & 1):
                    cnt += 1
            return cnt

        P_set = set(P)
        P_sorted = sorted(P, key=lambda v: degree_in(v, P_set))
        color_count, colors = greedy_coloring(P_sorted)

        if current_size + color_count <= best_size_found[0]:
            return current_size, set()

        # Branch: pick a vertex, try including it
        # Process in reverse color order (highest color first)
        for i in range(len(P_sorted) - 1, -1, -1):
            v = P_sorted[i]
            # Upper bound: current_size + (i + 1) (since colors[i] <= i+1)
            if current_size + i + 1 <= best_size_found[0]:
                return current_size, set()

            # Include v: new candidates are those in P_sorted[0:i] not adjacent to v
            new_P = [P_sorted[j] for j in range(i) if not ((adj[v] >> P_sorted[j]) & 1)]
            size, _ = branch_and_bound(new_P, current_size + 1, depth + 1)
            if size > best_size_found[0]:
                best_size_found[0] = size
                # Track best set via global
                best_set_found[0] = current_set.copy()
                best_set_found[0].add(v)

            # Exclude v: continue loop (implicitly)

        return best_size_found[0], best_set_found[0]

    # Global tracking for B&B
    best_size_found = [0]
    best_set_found = [set()]

    # Initial solution via greedy (min-degree order)
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    degree_order = sorted(range(M), key=lambda i: degrees[i])

    def greedy_init(order):
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    best_mask = greedy_init(degree_order)
    best_size = bin(best_mask).count('1')
    best_size_found[0] = best_size
    best_set_found[0] = set()
    for i in range(M):
        if (best_mask >> i) & 1:
            best_set_found[0].add(i)

    # Try a few random initializations to get a better starting point
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline - 30:
            break
        random_order = list(range(M))
        random.shuffle(random_order)
        mask = greedy_init(random_order)
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask
            best_size_found[0] = sz
            best_set_found[0] = set()
            for i in range(M):
                if (mask >> i) & 1:
                    best_set_found[0].add(i)

    # Run B&B from the best initial solution
    # P = all vertices not in the initial solution (we can add to it)
    # Actually, B&B should start from scratch or from the complement
    # Let's run B&B with P = all vertices, starting with current_size = 0
    # But that's expensive. Instead, use the initial solution as a lower bound.

    # Better approach: run B&B on the full graph with the initial solution as lower bound
    all_vertices = list(range(M))
    # Sort by degree for initial ordering
    all_vertices_sorted = sorted(all_vertices, key=lambda v: degrees[v])

    # Run B&B with a time limit
    result_size, result_set = branch_and_bound(all_vertices_sorted, 0)

    # If B&B didn't improve, try local search from best
    if result_size <= best_size:
        # Local search: swap-based improvement
        def local_search(current_mask, current_size):
            for _ in range(50000):
                if time.time() > deadline:
                    break
                improved = False
                sel_indices = [i for i in range(M) if (current_mask >> i) & 1]
                unselected = [i for i in range(M) if not (current_mask >> i) & 1]

                # Try adding a single element
                random.shuffle(unselected)
                for i in unselected:
                    if (adj[i] & current_mask) == 0:
                        current_mask |= (1 << i)
                        current_size += 1
                        improved = True
                        break

                if not improved:
                    # Try swap: remove one selected, add one unselected
                    if sel_indices and unselected:
                        random.shuffle(unselected)
                        for i in unselected:
                            # Find selected elements that conflict with i
                            conflicting = [j for j in sel_indices if (adj[i] >> j) & 1]
                            if len(conflicting) == 1:
                                # Swap: remove conflicting[0], add i
                                c = conflicting[0]
                                new_mask = (current_mask & ~(1 << c)) | (1 << i)
                                # Verify new mask is valid
                                valid = True
                                for a_idx in range(M):
                                    if (new_mask >> a_idx) & 1:
                                        if (adj[a_idx] & new_mask) & ~(1 << a_idx):
                                            # Check: adj[a_idx] & new_mask should only have a_idx itself? No, adj doesn't include self
                                            # adj[a_idx] has bits for neighbors. If any neighbor is in new_mask, conflict.
                                            # But we already know i is only conflicting with c, and c is removed.
                                            # Need to verify all pairs.
                                            pass
                                # Actually just check: for each new element, no conflicts
                                # Since we only removed c and added i, and i only conflicted with c among selected,
                                # the new set is valid.
                                current_mask = new_mask
                                # Size stays same
                                improved = True
                                break

                if not improved and random.random() < 0.1:
                    # Random restart component
                    pass

            return current_mask, current_size

        best_mask, best_size = local_search(best_mask, best_size)

    # Convert back to actual numbers
    if result_size > best_size:
        result = [valid_numbers[i] for i in result_set]
    else:
        result = [valid_numbers[i] for i in range(M) if (best_mask >> i) & 1]

    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]

    return sorted(result)
# EVOLVE-BLOCK-END
