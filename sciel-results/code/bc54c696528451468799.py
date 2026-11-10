"""Edge-coloring of K_16 to maximize distinct colors with no rainbow C4."""
import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute edge indices
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1

    # Precompute all 4-cycles as a numpy array
    cycles_list = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles_list.append((edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(a,d)]))
                    # Cycle 2: a-b, b-d, d-c, c-a
                    cycles_list.append((edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(c,d)], edge_idx[(a,c)]))
                    # Cycle 3: a-c, c-b, b-d, d-a
                    cycles_list.append((edge_idx[(a,c)], edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(a,d)]))

    cycles_arr = np.array(cycles_list, dtype=np.int32)  # shape (num_cycles, 4)
    num_cycles = len(cycles_arr)

    # Precompute for each edge, which cycles it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for ei in cycles_arr[ci]:
            edge_to_cycles[ei].append(ci)

    def count_violations_np(colors_np):
        """Vectorized count of rainbow 4-cycles using numpy."""
        # colors_np: shape (120,)
        # For each cycle, get the 4 colors
        c_colors = colors_np[cycles_arr]  # shape (num_cycles, 4)
        # Check all pairs are different
        # c0!=c1, c0!=c2, c0!=c3, c1!=c2, c1!=c3, c2!=c3
        diff = (
            (c_colors[:, 0] != c_colors[:, 1]) &
            (c_colors[:, 0] != c_colors[:, 2]) &
            (c_colors[:, 0] != c_colors[:, 3]) &
            (c_colors[:, 1] != c_colors[:, 2]) &
            (c_colors[:, 1] != c_colors[:, 3]) &
            (c_colors[:, 2] != c_colors[:, 3])
        )
        return int(np.sum(diff))

    def count_violations(colors):
        """Count number of rainbow 4-cycles (list interface)."""
        return count_violations_np(np.array(colors, dtype=np.int32))

    def count_distinct_colors(colors):
        return len(set(colors))

    def entropy_penalty(colors):
        """Penalize low diversity: return -log(used_colors) to encourage diversity."""
        used = len(set(colors))
        if used <= 1:
            return 100.0
        return -np.log(used)

    rng = random.Random(42)
    best_colors = None
    best_score = -1
    best_violations = float('inf')

    # Simulated annealing with entropy-aware objective
    # Objective: minimize violations * penalty_weight + entropy_penalty
    # We want violations == 0 and maximize distinct colors

    for restart in range(30):
        if time.time() - start_time > DEADLINE:
            break

        # Initialize with a moderately diverse random coloring
        colors = np.array([rng.randint(0, MAX_COLOR - 1) for _ in range(NUM_EDGES)], dtype=np.int32)

        # Simulated annealing parameters
        T_start = 50.0
        T_end = 0.1
        sa_steps = 8000
        cooling_rate = (T_end / T_start) ** (1.0 / sa_steps)

        for step in range(sa_steps):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cooling_rate ** step)

            # Pick a random edge
            ei = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[ei]
            new_color = rng.randint(0, MAX_COLOR - 1)
            if new_color == old_color:
                continue

            # Compute delta violations
            colors[ei] = new_color
            new_viol = count_violations_np(colors)
            colors[ei] = old_color
            old_viol = count_violations_np(colors)
            delta_viol = new_viol - old_viol

            # Compute delta entropy penalty (approximate: just check if new color adds diversity)
            old_used = len(set(colors.tolist()))
            colors[ei] = new_color
            new_used = len(set(colors.tolist()))
            colors[ei] = old_color
            delta_entropy = entropy_penalty(colors.tolist() if new_color != old_color else colors.tolist()) - entropy_penalty(colors.tolist())
            # Simpler: if new color is not in old set, entropy improves (penalty decreases)
            old_set = set(colors.tolist())
            if new_color not in old_set:
                delta_entropy = -1.0  # improvement
            elif old_color in old_set and sum(1 for x in colors if x == old_color) == 1:
                delta_entropy = 1.0  # we're removing a unique color
            else:
                delta_entropy = 0.0

            # Total delta: violations weighted heavily + entropy
            delta_total = delta_viol * 10.0 + delta_entropy

            # Accept if delta < 0 or with probability exp(-delta/T)
            if delta_total < 0:
                colors[ei] = new_color
            else:
                if T > 0:
                    if rng.random() < np.exp(-delta_total / T):
                        colors[ei] = new_color

        # After SA, do local search to fix any remaining violations
        cur_viol = count_violations_np(colors)
        if cur_viol == 0:
            score = count_distinct_colors(colors.tolist())
            if score > best_score:
                best_score = score
                best_colors = colors.copy()
                best_violations = 0
        elif best_colors is None or cur_viol < best_violations:
            best_colors = colors.copy()
            best_violations = cur_viol

    # If we have a valid coloring, try to increase distinct colors with greedy + random
    if best_colors is not None and best_violations == 0:
        colors_list = best_colors.tolist()
        used_colors = set(colors_list)
        max_iter2 = 30000
        for it in range(max_iter2):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            for _ in range(20):
                ei = rng.randint(0, NUM_EDGES - 1)
                old_color = colors_list[ei]
                available = [c for c in range(MAX_COLOR) if c not in used_colors]
                if not available:
                    break
                nc = rng.choice(available)

                # Check if this change creates violations
                colors_np = np.array(colors_list, dtype=np.int32)
                colors_np[ei] = nc
                new_viol = count_violations_np(colors_np)
                if new_viol == 0:
                    colors_list[ei] = nc
                    used_colors.add(nc)
                    improved = True
                    break

            if not improved:
                # Try random changes that don't create violations (may or may not add new colors)
                any_change = False
                for _ in range(10):
                    ei = rng.randint(0, NUM_EDGES - 1)
                    old_color = colors_list[ei]
                    nc = rng.randint(0, MAX_COLOR - 1)
                    if nc == old_color:
                        continue
                    colors_np = np.array(colors_list, dtype=np.int32)
                    colors_np[ei] = nc
                    if count_violations_np(colors_np) == 0:
                        colors_list[ei] = nc
                        used_colors = set(colors_list)
                        any_change = True
                        break
                if not any_change:
                    break
        best_colors = np.array(colors_list, dtype=np.int32)
        best_score = len(set(best_colors.tolist()))
        best_violations = 0

    # Final validation
    if best_colors is None:
        return [0] * NUM_EDGES

    final_violations = count_violations_np(best_colors)
    if final_violations > 0:
        # Fallback: structured coloring
        colors2 = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(edges):
            colors2[idx] = (i + j) % 15
        if count_violations(colors2) == 0:
            return colors2
        else:
            return [0] * NUM_EDGES

    return best_colors.tolist()

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Distinct colors: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")
