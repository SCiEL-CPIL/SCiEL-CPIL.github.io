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

    def search_n(n, time_budget, seed_offset=0):
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

        # Represent cycles as numpy array
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

        def compute_edge_weights(colors_arr):
            weights = np.zeros(m, dtype=np.int32)
            for ci in range(num_cycles):
                ce = cycles_np[ci]
                c0 = colors_arr[ce[0]]
                cnt = 0
                for k in range(1, 6):
                    if colors_arr[ce[k]] == c0:
                        cnt += 1
                if cnt >= 5:
                    for e in ce:
                        weights[e] += 1
            return weights

        max_iter = 300000
        start_iter_time = time.time()
        stuck_counter = 0
        kick_threshold = 5000

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            if stuck_counter >= kick_threshold:
                kick_size = random.randint(5, 10)
                for _ in range(kick_size):
                    e = random.randint(0, m - 1)
                    colors[e] = random.randint(0, 3)
                bad_count = count_bad()
                stuck_counter = 0
                continue

            weights = compute_edge_weights(colors)
            max_w = weights.max()

            if max_w == 0:
                e = random.randint(0, m - 1)
            else:
                top_edges = np.where(weights == max_w)[0]
                e = int(top_edges[random.randint(0, len(top_edges) - 1)])

            c_old = colors[e]
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)

            colors[e] = c_new
            new_bad = count_bad()

            if new_bad < bad_count:
                bad_count = new_bad
                stuck_counter = 0
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors.tolist()}
            else:
                if random.random() < 0.1:
                    bad_count = new_bad
                    stuck_counter += 1
                else:
                    colors[e] = c_old
                    stuck_counter += 1

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    def search_n_algebraic(n, time_budget, seed_offset=0):
        """
        Exploit F_p structure: color edge (i,j) by f((j-i) mod p) where p is prime >= n.
        For n=17, p=17. For n=16, p=17. For n=15, p=17.
        The search space is just 4^(p-1) possible mappings (differences 1..p-1).
        """
        if time.time() > deadline - 5:
            return None

        # Find smallest prime >= n
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(math.isqrt(x)) + 1):
                if x % i == 0:
                    return False
            return True

        p = n
        while not is_prime(p):
            p += 1

        # Number of differences: 1, 2, ..., p-1 (but for undirected, diff d and p-d are equivalent)
        # Actually in F_p, the edge (i,j) has difference (j-i) mod p.
        # For undirected graph, we can use min(diff, p-diff) as the "distance"
        # But let's just use (j-i) mod p directly and note that edge (i,j) and (j,i) have diff d and p-d.
        # Since the graph is undirected, we need f(d) = f(p-d) for consistency? No, we're coloring K_n not K_p.
        # For vertices 0..n-1 in F_p, edge (i,j) where i<j has difference (j-i) mod p which is in 1..n-1.
        # So we need to assign colors to differences 1, 2, ..., n-1.
        
        num_diffs = n - 1  # differences 1 to n-1
        m = n * (n - 1) // 2

        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        cycles_np = np.array(cycles, dtype=np.int32)

        # Precompute for each cycle, what differences appear
        # For edge (i,j) with i<j, diff = j-i
        # edge_index(i,j,n) = i*(n-2) - i*(i-1)//2 + j - 1
        # We need to map edge index to (i,j) then to diff
        
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

        # Now for each cycle, the 6 edges have certain differences
        # A cycle is monochromatic if all 6 edges have the same color
        # Edge e has color f(edge_diff[e])
        
        # We search over assignments of colors to differences 1..n-1
        # Total search space: 4^(n-1) which for n=17 is 4^16 = 4.3e9 - too big for brute force
        # But we can use local search on the difference assignment
        
        random.seed(42 + seed_offset)
        
        # diff_colors[d] = color for difference d (d in 1..n-1)
        diff_colors = np.array([random.randint(0, 3) for _ in range(n)], dtype=np.int8)
        # diff_colors[0] unused, diff_colors[1]..diff_colors[n-1] used

        def count_bad_from_diffs():
            # For each cycle, check if all 6 edges have same color
            # Edge e has color diff_colors[edge_diff[e]]
            # Cycle c has edges cycles_np[c, 0..5]
            # All same if diff_colors[edge_diff[c,0]] == diff_colors[edge_diff[c,1]] == ...
            
            # Vectorized: for each cycle, get the 6 differences
            cycle_diffs = edge_diff[cycles_np]  # shape (num_cycles, 6)
            
            # For each cycle, check if all 6 differences map to same color
            # Get colors for each edge in each cycle
            cycle_edge_colors = diff_colors[cycle_diffs]  # shape (num_cycles, 6)
            
            # Check if all 6 are equal
            mono = (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 1]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 2]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 3]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 4]) & \
                   (cycle_edge_colors[:, 0] == cycle_edge_colors[:, 5])
            
            return int(mono.sum())

        bad_count = count_bad_from_diffs()
        if bad_count == 0:
            # Convert to full coloring
            colors = np.zeros(m, dtype=np.int8)
            for e in range(m):
                d = edge_diff[e]
                colors[e] = diff_colors[d]
            return {'n': n, 'colors': colors.tolist()}

        best_diff_colors = diff_colors.copy()
        best_bad = bad_count

        # Local search: modify one difference's color at a time
        max_iter = 1000000
        start_iter_time = time.time()
        stuck_counter = 0
        kick_threshold = 10000

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            if stuck_counter >= kick_threshold:
                # Kick: randomize several differences
                kick_size = random.randint(3, 8)
                for _ in range(kick_size):
                    d = random.randint(1, n - 1)
                    diff_colors[d] = random.randint(0, 3)
                bad_count = count_bad_from_diffs()
                stuck_counter = 0
                continue

            # Pick a random difference to modify
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

    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    time_alloc = {17: 30, 16: 25, 15: 20, 14: 15, 13: 12, 12: 10, 11: 8, 10: 6, 9: 4, 8: 3, 7: 2, 6: 2}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        
        # Try algebraic approach first for n >= 15
        if n >= 15:
            num_restarts_algebraic = 3
            algebraic_budget = tb * 0.6 / num_restarts_algebraic
            for r in range(num_restarts_algebraic):
                if time.time() > deadline - 8:
                    break
                result = search_n_algebraic(n, algebraic_budget, seed_offset=r * 1000 + n)
                if result:
                    best_result = result
                    return best_result
        
        # Then try general approach
        remaining_budget = tb * 0.4
        num_restarts = 2 if n >= 15 else 1
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
