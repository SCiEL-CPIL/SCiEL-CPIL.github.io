import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles
    four_cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # Three 4-cycles:
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]

        for cycle_edges in [e1, e2, e3]:
            four_cycles.append(cycle_edges)
            for e in cycle_edges:
                others = [x for x in cycle_edges if x != e]
                edge_to_cycles[e].append(others)

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        for cycle in four_cycles:
            if len(set(colors[e] for e in cycle)) == 4:
                return False
        return True

    def try_change(colors, edge, new_color):
        old_color = colors[edge]
        if old_color == new_color:
            return True
        for others in edge_to_cycles[edge]:
            cycle_colors = set()
            cycle_colors.add(new_color)
            for e in others:
                cycle_colors.add(colors[e])
            if len(cycle_colors) == 4:
                return False
        return True

    def get_valid_colors_for_edge(colors, edge):
        """Return list of colors that can be assigned to edge without violating constraints."""
        old_color = colors[edge]
        valid = []
        for c in range(MAX_COLOR):
            if c == old_color:
                continue
            valid_flag = True
            for others in edge_to_cycles[edge]:
                cycle_colors = set()
                cycle_colors.add(c)
                for e in others:
                    cycle_colors.add(colors[e])
                if len(cycle_colors) == 4:
                    valid_flag = False
                    break
            if valid_flag:
                valid.append(c)
        return valid

    # Modular arithmetic base coloring: c(i,j) = (i*a + j*b) mod p
    # This creates a non-hierarchical structure that may avoid the partition cap
    def modular_coloring(a, b, p):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = (i * a + j * b) % p
        return colors

    # Try several modular constructions and pick the best valid one
    best_colors = None
    best_count = 0
    random.seed(42)

    # Candidate modular parameters
    candidates = []
    for p in [17, 19, 23, 29, 31]:
        for a in range(1, p):
            for b in range(1, p):
                if a != b:
                    candidates.append((a, b, p))
    random.shuffle(candidates)

    # Also include transitive as fallback
    transitive = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            transitive[edge_idx(i, j)] = i

    # Evaluate candidates
    for (a, b, p) in candidates[:50]:
        if time.time() > deadline - 10:
            break
        colors = modular_coloring(a, b, p)
        if is_valid(colors):
            cnt = count_distinct(colors)
            if cnt > best_count:
                best_count = cnt
                best_colors = colors[:]

    # If no modular coloring was valid, use transitive
    if best_colors is None:
        best_colors = transitive[:]
        best_count = count_distinct(best_colors)

    # Phase 1: Greedy introduction of new colors on the best base
    used_colors = set(best_colors)
    unused_colors = [c for c in range(MAX_COLOR) if c not in used_colors]

    edges_order = list(range(NUM_EDGES))
    random.shuffle(edges_order)

    for edge in edges_order:
        if time.time() > deadline - 10:
            break
        if not unused_colors:
            break

        # Try each unused color
        for new_color in unused_colors:
            if try_change(best_colors, edge, new_color):
                best_colors[edge] = new_color
                unused_colors.remove(new_color)
                best_count += 1
                break

    # Phase 2: Local search to further improve
    for restart in range(30):
        if time.time() > deadline - 5:
            break

        current = best_colors[:]
        current_count = best_count
        used = set(current)
        unused = [c for c in range(MAX_COLOR) if c not in used]

        if not unused:
            break

        # Try to add more colors
        for step in range(100):
            if time.time() > deadline - 5:
                break
            if not unused:
                break

            edge = random.randint(0, NUM_EDGES - 1)
            improved = False
            random.shuffle(unused)
            for new_color in unused:
                if try_change(current, edge, new_color):
                    current[edge] = new_color
                    unused.remove(new_color)
                    current_count += 1
                    improved = True
                    break

            if improved and current_count > best_count:
                best_count = current_count
                best_colors = current[:]

    # Final verification
    if not is_valid(best_colors):
        # Fallback to transitive
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i

    return best_colors
