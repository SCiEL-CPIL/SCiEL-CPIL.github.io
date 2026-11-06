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

    # Strategy: Start with a base coloring, then use a more systematic approach
    # to introduce new colors while maintaining validity

    # Base: transitive coloring
    colors = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i

    best_colors = colors[:]
    best_count = count_distinct(best_colors)

    random.seed(12345)

    # Phase 1: Greedy introduction of new colors
    # For each edge, try to assign it a color not currently used
    for phase in range(20):
        if time.time() > deadline - 5:
            break

        current = best_colors[:]
        current_count = best_count
        used_colors = set(current)

        # Shuffle edges
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)

        for edge in edges:
            if time.time() > deadline - 5:
                break

            old_color = current[edge]
            # Try unused colors first
            unused = [c for c in range(MAX_COLOR) if c not in used_colors]
            random.shuffle(unused)

            for new_color in unused:
                if try_change(current, edge, new_color):
                    current[edge] = new_color
                    used_colors.add(new_color)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = current[:]
                    break

    # Phase 2: Local search with multiple restarts
    # Try to find colorings with more distinct colors by random restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break

        # Start from best or random valid coloring
        if restart % 3 == 0 and best_count > 16:
            current = best_colors[:]
        else:
            # Random valid coloring: assign colors greedily
            current = [0] * NUM_EDGES
            valid = True
            for edge in range(NUM_EDGES):
                # Try random colors until valid
                found = False
                for _ in range(20):
                    c = random.randint(0, MAX_COLOR - 1)
                    if try_change(current, edge, c):
                        current[edge] = c
                        found = True
                        break
                if not found:
                    # Fallback: use 0
                    current[edge] = 0
                    valid = False

        current_count = count_distinct(current)

        # Local search: try to increase distinct colors
        for step in range(200):
            if time.time() > deadline - 5:
                break

            used_colors = set(current)
            unused = [c for c in range(MAX_COLOR) if c not in used_colors]

            if not unused:
                break

            # Pick a random edge and try to give it an unused color
            edge = random.randint(0, NUM_EDGES - 1)
            old_color = current[edge]

            random.shuffle(unused)
            improved = False
            for new_color in unused:
                if try_change(current, edge, new_color):
                    current[edge] = new_color
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = current[:]
                    improved = True
                    break

            if not improved:
                # Try changing to any valid color to explore
                valid_colors = get_valid_colors_for_edge(current, edge)
                if valid_colors:
                    new_color = random.choice(valid_colors)
                    old = current[edge]
                    current[edge] = new_color
                    new_count = count_distinct(current)
                    # Accept if same or better (simulated annealing style)
                    if new_count >= current_count or random.random() < 0.1:
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = current[:]

    # Final verification
    if not is_valid(best_colors):
        # Fallback to transitive
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i

    return best_colors
