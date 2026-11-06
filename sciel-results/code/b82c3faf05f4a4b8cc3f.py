"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]

    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build adjacency as bitmasks for fast operations
    # adj[i] is a bitmask of vertices adjacent to i
    adj = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    # Tomita's branch-and-bound algorithm for Maximum Independent Set
    # Uses greedy coloring for upper bounds
    best_size = [0]
    best_set = [set()]

    def color_sort(vertices):
        """Greedy coloring to get upper bound. Returns (colored_vertices, max_color)."""
        color_of = {}
        uncolored = set(vertices)
        color = 0
        ordered = []
        while uncolored:
            color += 1
            # Find maximal independent set in uncolored
            available = set(uncolored)
            while available:
                # Pick vertex with fewest neighbors in available
                best_v = None
                best_deg = float('inf')
                for v in available:
                    deg = len(adj[v] & (1 << num_valid - 1) & 0)  # placeholder
                    # Count neighbors in available
                    cnt = 0
                    for u in available:
                        if u != v and (adj[v] >> u) & 1:
                            cnt += 1
                    if cnt < best_deg:
                        best_deg = cnt
                        best_v = v
                if best_v is None:
                    break
                color_of[best_v] = color
                ordered.append(best_v)
                uncolored.discard(best_v)
                available.discard(best_v)
                # Remove neighbors of best_v from available
                to_remove = set()
                for u in available:
                    if (adj[best_v] >> u) & 1:
                        to_remove.add(u)
                available -= to_remove
        return ordered, color

    def branch_and_bound(vertices, current_set, current_size):
        if time.time() > deadline:
            return

        if not vertices:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return

        # Upper bound: current_size + len(vertices)
        if current_size + len(vertices) <= best_size[0]:
            return

        # Color sort for tighter upper bound
        ordered, max_color = color_sort(vertices)
        if current_size + max_color <= best_size[0]:
            return

        # Branch on vertices in reverse color order
        for v in reversed(ordered):
            if time.time() > deadline:
                return
            if v not in vertices:
                continue
            # Include v
            new_vertices = vertices - {v}
            # Remove all neighbors of v
            neighbors = set()
            for u in new_vertices:
                if (adj[v] >> u) & 1:
                    neighbors.add(u)
            new_vertices -= neighbors
            branch_and_bound(new_vertices, current_set | {v}, current_size + 1)
            # Exclude v
            vertices.discard(v)

    # Run branch and bound
    all_vertices = set(range(num_valid))
    branch_and_bound(all_vertices, set(), 0)

    # Convert back to values
    if best_set[0]:
        result = sorted([valid_numbers[i] for i in best_set[0]])
    else:
        # Fallback
        result = sorted(valid_numbers)

    # Verify result
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if int(math.isqrt(s)) ** 2 == s:
                # Invalid, fallback
                result = sorted(valid_numbers)
                break

    return result if result else [1]
# EVOLVE-BLOCK-END
