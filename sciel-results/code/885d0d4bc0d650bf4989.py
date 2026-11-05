import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Precompute all 4-cycles and edge indices
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Map edge index to (u, v)
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[idx] = (i, j)
            idx += 1

    # Precompute cycles
    all_cycles = []
    vertices = list(range(N))

    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1, e2, e3, e4 = get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)
        all_cycles.append((e1, e2, e3, e4))
        # Cycle 2: a-b-d-c-a
        e1, e2, e3, e4 = get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)
        all_cycles.append((e1, e2, e3, e4))
        # Cycle 3: a-c-b-d-a
        e1, e2, e3, e4 = get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)
        all_cycles.append((e1, e2, e3, e4))

    num_cycles = len(all_cycles)
    cycles_arr = np.array(all_cycles, dtype=np.int32)

    # Precompute for each edge, the cycles it belongs to and its position in them
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for c_idx in range(num_cycles):
        e1, e2, e3, e4 = all_cycles[c_idx]
        edge_to_cycles[e1].append((c_idx, 0))
        edge_to_cycles[e2].append((c_idx, 1))
        edge_to_cycles[e3].append((c_idx, 2))
        edge_to_cycles[e4].append((c_idx, 3))

    # 2. Fast evaluation
    def evaluate(colors):
        num_unique = len(np.unique(colors))
        c0 = colors[cycles_arr[:, 0]]
        c1 = colors[cycles_arr[:, 1]]
        c2 = colors[cycles_arr[:, 2]]
        c3 = colors[cycles_arr[:, 3]]
        distinct_mask = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        num_violations = np.sum(distinct_mask)
        return num_unique, num_violations

    # 3. Incremental violation tracking
    # For each cycle, track the 4 colors
    def is_rainbow(c0, c1, c2, c3):
        return len({c0, c1, c2, c3}) == 4

    # 4. Base Construction: Disjoint Triangles
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_colors = current_colors.copy()
    best_score = 1
    best_violations = 0

    triangles = []
    remaining = list(range(N))
    while len(remaining) >= 3:
        t = (remaining.pop(), remaining.pop(), remaining.pop())
        triangles.append(t)

    color_id = 1
    for u, v, w in triangles:
        e_uv = get_idx(u, v)
        e_vw = get_idx(v, w)
        e_wu = get_idx(w, u)
        current_colors[e_uv] = color_id
        current_colors[e_vw] = color_id + 1
        current_colors[e_wu] = color_id + 2
        color_id += 3

    curr_score, curr_violations = evaluate(current_colors)
    if curr_violations == 0:
        best_score = curr_score
        best_colors = current_colors.copy()
        best_violations = 0
    else:
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        best_colors = current_colors.copy()
        best_score = 1
        best_violations = 0
        curr_score = 1
        curr_violations = 0

    # 5. Incremental local search with repair
    rng = random.Random(42)
    T = 10.0
    cooling = 0.9995
    T_min = 0.01

    edge_indices = list(range(NUM_EDGES))

    # Track which cycles are currently violated
    def get_violated_cycles(colors):
        violated = set()
        for c_idx in range(num_cycles):
            e1, e2, e3, e4 = all_cycles[c_idx]
            c0, c1, c2, c3 = colors[e1], colors[e2], colors[e3], colors[e4]
            if len({c0, c1, c2, c3}) == 4:
                violated.add(c_idx)
        return violated

    curr_violated = get_violated_cycles(current_colors)

    def try_recolor(e, new_c, colors, violated_set):
        """Try recoloring edge e to new_c. Returns (success, new_violated_set)"""
        old_c = colors[e]
        if old_c == new_c:
            return False, violated_set

        # Check if this change would fix or break cycles
        # Cycles containing edge e
        affected_cycles = []
        for c_idx, pos in edge_to_cycles[e]:
            e1, e2, e3, e4 = all_cycles[c_idx]
            c0, c1, c2, c3 = colors[e1], colors[e2], colors[e3], colors[e4]
            # Update the color at position pos
            new_colors = [c0, c1, c2, c3]
            new_colors[pos] = new_c
            was_violated = c_idx in violated_set
            is_now_violated = len(set(new_colors)) == 4

            if was_violated and not is_now_violated:
                # Fixed a violation - good
                pass
            elif not was_violated and is_now_violated:
                # Created a violation - bad
                return False, violated_set
            elif was_violated and is_now_violated:
                # Still violated - neutral
                pass
            else:
                # Still not violated - neutral
                pass

            affected_cycles.append((c_idx, was_violated, is_now_violated))

        # Apply the change
        new_violated = set(violated_set)
        for c_idx, was_v, is_now_v in affected_cycles:
            if was_v and not is_now_v:
                new_violated.discard(c_idx)
            elif not was_v and is_now_v:
                new_violated.add(c_idx)

        colors[e] = new_c
        return True, new_violated

    # Phase 1: Repair - fix all violations first
    repair_deadline = start_time + 20.0
    while curr_violations > 0 and time.time() < repair_deadline:
        # Pick a random violated cycle
        if not curr_violated:
            break
        c_idx = rng.choice(list(curr_violated))
        e1, e2, e3, e4 = all_cycles[c_idx]
        c0, c1, c2, c3 = current_colors[e1], current_colors[e2], current_colors[e3], current_colors[e4]

        # Try to make two edges in this cycle share a color
        # Option 1: set e1's color to match e2
        # Option 2: set e2's color to match e1
        # Option 3: set e3's color to match e4
        # Option 4: set e4's color to match e3
        options = [
            (e1, c1),  # make e1 = e2's color
            (e2, c0),  # make e2 = e1's color
            (e3, c3),  # make e3 = e4's color
            (e4, c2),  # make e4 = e3's color
            (e1, c3),  # make e1 = e4's color
            (e4, c0),  # make e4 = e1's color
            (e2, c3),  # make e2 = e4's color
            (e3, c0),  # make e3 = e1's color
        ]
        rng.shuffle(options)

        repaired = False
        for e, target_c in options:
            if current_colors[e] == target_c:
                continue
            # Try this recoloring
            test_colors = current_colors.copy()
            test_violated = set(curr_violated)
            success, new_violated = try_recolor(e, target_c, test_colors, test_violated)
            if success:
                # Check if we reduced total violations
                if len(new_violated) < len(curr_violated):
                    current_colors = test_colors
                    curr_violated = new_violated
                    curr_violations = len(curr_violated)
                    repaired = True
                    break

        if not repaired:
            # Try random recoloring of one edge in the violated cycle
            e = rng.choice([e1, e2, e3, e4])
            new_c = rng.randint(0, MAX_COLORS - 1)
            test_colors = current_colors.copy()
            test_violated = set(curr_violated)
            success, new_violated = try_recolor(e, new_c, test_colors, test_violated)
            if success and len(new_violated) <= len(curr_violated):
                current_colors = test_colors
                curr_violated = new_violated
                curr_violations = len(curr_violated)

        # Periodic full re-evaluation to sync state
        if time.time() > repair_deadline - 0.1:
            break

    # Full re-evaluation
    curr_score, curr_violations = evaluate(current_colors)
    curr_violated = get_violated_cycles(current_colors)

    if curr_violations == 0 and curr_score > best_score:
        best_score = curr_score
        best_colors = current_colors.copy()

    # Phase 2: Maximize colors using incremental SA
    T = 5.0
    cooling = 0.9998
    T_min = 0.01
    last_improvement = time.time()
    no_improve_count = 0

    while time.time() < deadline - 2.0:
        # Pick a random edge
        e = rng.choice(edge_indices)
        old_c = current_colors[e]

        # Pick a new color - bias towards unused colors
        if rng.random() < 0.6:
            used = set(current_colors)
            unused = [c for c in range(MAX_COLORS) if c not in used]
            if unused:
                new_c = rng.choice(unused)
            else:
                new_c = rng.randint(0, MAX_COLORS - 1)
        else:
            new_c = rng.randint(0, MAX_COLORS - 1)

        if new_c == old_c:
            continue

        # Try the recoloring incrementally
        test_colors = current_colors.copy()
        test_violated = set(curr_violated)
        success, new_violated = try_recolor(e, new_c, test_colors, test_violated)

        if not success:
            continue

        new_violations = len(new_violated)
        new_score = len(np.unique(test_colors))

        # Check for new best
        if new_violations == 0 and new_score > best_score:
            best_score = new_score
            best_colors = test_colors.copy()
            last_improvement = time.time()
            no_improve_count = 0

        # SA acceptance
        old_fitness = curr_score - 20.0 * curr_violations
        new_fitness = new_score - 20.0 * new_violations
        delta = new_fitness - old_fitness

        if delta >= 0:
            accept = True
        else:
            if T > T_min:
                accept = rng.random() < np.exp(delta / T)
            else:
                accept = False

        if accept:
            current_colors = test_colors
            curr_violated = new_violated
            curr_violations = new_violations
            curr_score = new_score
            no_improve_count += 1
        else:
            no_improve_count += 1

        T = max(T_min, T * cooling)

        # If stuck for too long without improvement, perturb
        if no_improve_count > 5000:
            no_improve_count = 0
            # Perturb: randomly recolor a few edges
            for _ in range(5):
                pe = rng.choice(edge_indices)
                pc = rng.randint(0, MAX_COLORS - 1)
                current_colors[pe] = pc
            curr_score, curr_violations = evaluate(current_colors)
            curr_violated = get_violated_cycles(current_colors)
            T = 5.0  # Reset temperature

    # Phase 3: Final polish - try to add more colors to best solution
    if time.time() < deadline - 1.0:
        polish_colors = best_colors.copy()
        polish_violated = get_violated_cycles(polish_colors)
        polish_score = len(np.unique(polish_colors))
        polish_deadline = deadline - 0.5

        while time.time() < polish_deadline:
            # Try to recolor an edge to an unused color
            used = set(polish_colors)
            unused = [c for c in range(MAX_COLORS) if c not in used]
            if not unused:
                break

            e = rng.choice(edge_indices)
            new_c = rng.choice(unused)
            if polish_colors[e] == new_c:
                continue

            test_colors = polish_colors.copy()
            test_violated = set(polish_violated)
            success, new_violated = try_recolor(e, new_c, test_colors, test_violated)

            if success and len(new_violated) == 0:
                new_score = len(np.unique(test_colors))
                if new_score > polish_score:
                    polish_colors = test_colors
                    polish_violated = new_violated
                    polish_score = new_score
                    if polish_score > best_score:
                        best_score = polish_score
                        best_colors = polish_colors.copy()

    # Final verification
    final_score, final_violations = evaluate(best_colors)
    if final_violations != 0:
        # Fallback to all zeros
        return [0] * NUM_EDGES

    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
