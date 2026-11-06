import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles(n):
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def search_n_algebraic(n, time_budget, seed_offset=0):
        """
        Exploit the difference-based structure for small n.
        This is effective for n <= 11 where the space is small enough for
        efficient local search to converge, and acts as a fast fallback.
        """
        if time.time() > deadline - 5:
            return None

        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(math.isqrt(x)) + 1):
                if x % i == 0:
                    return False
            return True

        # For n > 11, this algebraic structure is too restrictive and we skip it
        # to save time for the general search which is more likely to find valid
        # colorings for larger n.
        if n > 11:
            return None

        m = n * (n - 1) // 2
        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        cycles_np = np.array(cycles, dtype=np.int32)

        # Build edge index to (i,j) mapping
        edge_to_ij = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_to_ij[idx] = (i, j)
                idx += 1

        # For each edge, compute its difference
        edge_diff = np.zeros(m, dtype=np.int32)
        for e, (i, j) in edge_to_ij.items():
            edge_diff[e] = j - i

        random.seed(42 + seed_offset)

        # diff_colors[d] = color for difference d (d in 1..n-1)
        diff_colors = np.array([random.randint(0, 3) for _ in range(n)], dtype=np.int8)

        def count_bad_from_diffs():
            cycle_diffs = edge_diff[cycles_np]
            cycle_edge_colors = diff_colors[cycle_diffs]
            mono = (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 1]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 2]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 3]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 4]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 5])
            return int(mono.sum())

        bad_count = count_bad_from_diffs()
        if bad_count == 0:
            colors = np.zeros(m, dtype=np.int8)
            for e in range(m):
                d = edge_diff[e]
                colors[e] = diff_colors[d]
            return {'n': n, 'colors': colors.tolist()}

        best_diff_colors = diff_colors.copy()
        best_bad = bad_count

        max_iter = 200000
        start_iter_time = time.time()
        stuck_counter = 0
        kick_threshold = 5000

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            if stuck_counter >= kick_threshold:
                kick_size = random.randint(3, 8)
                for _ in range(kick_size):
                    d = random.randint(1, n - 1)
                    diff_colors[d] = random.randint(0, 3)
                bad_count = count_bad_from_diffs()
                stuck_counter = 0
                continue

            d = random.randint(1, n - 1)
            old_color = diff_colors[d]
            new_color = random.randint(0, 3)
            while new_color == old_color:
                new_color = random.randint(0, 3)

            diff_colors[d] = new_color
            new_bad = count_bad_from_diffs()

            if new_bad < bad_count:
                bad_count = new_bad
                stuck_counter = 0
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_diff_colors = diff_colors.copy()
                    if best_bad == 0:
                        colors = np.zeros(m, dtype=np.int8)
                        for e in range(m):
                            dd = edge_diff[e]
                            colors[e] = best_diff_colors[dd]
                        return {'n': n, 'colors': colors.tolist()}
            else:
                if random.random() < 0.15:
                    bad_count = new_bad
                    stuck_counter += 1
                else:
                    diff_colors[d] = old_color
                    stuck_counter += 1

        if best_bad == 0:
            colors = np.zeros(m, dtype=np.int8)
            for e in range(m):
                d = edge_diff[e]
                colors[e] = best_diff_colors[d]
            return {'n': n, 'colors': colors.tolist()}
        return None

    def search_n(n, time_budget, seed_offset=0):
        """
        General local search with delta updates.
        This is the primary engine for n >= 12.
        """
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        # Build edge-to-cycles mapping
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        # Represent cycles as numpy array for fast access
        cycles_np = np.array(cycles, dtype=np.int32)

        # Initial random coloring
        colors = np.array([random.randint(0, 3) for _ in range(m)], dtype=np.int8)

        def count_bad():
            c0 = cycles_np[:, 0]
            mono = (colors[c0] == colors[cycles_np[:, 1]]) & \
                   (colors[c0] == colors[cycles_np[:, 2]]) & \
                   (colors[c0] == colors[cycles_np[:, 3]]) & \
                   (colors[c0] == colors[cycles_np[:, 4]]) & \
                   (colors[c0] == colors[cycles_np[:, 5]])
            return int(mono.sum())

        bad_count = count_bad()
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        # Precompute for each edge, the list of (cycle_id, position_in_cycle)
        # Actually, we just need the cycle id to access the other 5 edges.
        # We can just use edge_to_cycles.

        def compute_delta(e, c_old, c_new):
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles_np[ci]
                # Check if was mono
                was_mono = True
                for k in range(6):
                    ed = ce[k]
                    if ed == e:
                        continue
                    if colors[ed] != c_old:
                        was_mono = False
                        break
                # Check if will be mono
                will_be_mono = True
                for k in range(6):
                    ed = ce[k]
                    if ed == e:
                        continue
                    if colors[ed] != c_new:
                        will_be_mono = False
                        break

                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            return delta

        max_iter = 500000
        start_iter_time = time.time()
        stuck_counter = 0
        kick_threshold = 2000
        T = 1.0
        cool_rate = 0.999

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            if stuck_counter >= kick_threshold:
                # Kick: randomize several edges
                kick_size = random.randint(5, 10)
                for _ in range(kick_size):
                    e = random.randint(0, m - 1)
                    colors[e] = random.randint(0, 3)
                bad_count = count_bad()
                stuck_counter = 0
                T = 1.0
                continue

            # Pick a random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)

            delta = compute_delta(e, c_old, c_new)

            accept = False
            if delta < 0:
                accept = True
            else:
                if delta > 0:
                    if delta / T > 50:
                        accept = False
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            accept = True
                else:
                    # delta == 0
                    accept = random.random() < 0.5

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors.tolist()}
                stuck_counter = 0
            else:
                stuck_counter += 1

            T *= cool_rate

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Targets to try
    # We want to maximize n.
    # Start with high n and go down.
    # If we find a valid coloring for n, we return it immediately?
    # No, we should try to find the *maximum* n.
    # But if we find n=17, that's likely the max.
    # If we find n=16, we should check if 17 is possible?
    # Given time constraints, finding a valid coloring for a high n is the goal.
    # The score is n/seed.
    # Current best known is 17.
    # We will try 17, 16, 15, 14, 13, 12, 11, 10...
    
    # Time allocation:
    # n=17: 40s
    # n=16: 30s
    # n=15: 20s
    # n=14: 15s
    # n=13: 10s
    # n=12: 5s
    # n=11: 3s
    # n=10: 2s
    # n=9: 1s
    # n=8: 1s
    # n=7: 1s
    # n=6: 1s
    # Total: ~129s

    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    time_alloc = {17: 40, 16: 30, 15: 20, 14: 15, 13: 10, 12: 5, 11: 3, 10: 2, 9: 1, 8: 1, 7: 1, 6: 1}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        
        # For n <= 11, try algebraic first (it's fast and might find a solution)
        if n <= 11:
            num_restarts_algebraic = 2
            algebraic_budget = tb * 0.5 / num_restarts_algebraic
            for r in range(num_restarts_algebraic):
                if time.time() > deadline - 8:
                    break
                result = search_n_algebraic(n, algebraic_budget, seed_offset=r * 1000 + n)
                if result:
                    best_result = result
                    return best_result
        
        # Then try general approach
        remaining_budget = tb
        if n <= 11:
            remaining_budget = tb * 0.5
        
        num_restarts = 1
        if n >= 15:
            num_restarts = 2 # Try two different random seeds for higher n
        elif n >= 12:
            num_restarts = 2
        else:
            num_restarts = 1
            
        restart_budget = remaining_budget / num_restarts
        for r in range(num_restarts):
            if time.time() > deadline - 8:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
