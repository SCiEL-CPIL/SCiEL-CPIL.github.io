import time
import math

EDGE_LIST = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24
ALL_MASK = (1 << N) - 1

# Precompute edge index for each pair
EDGE_INDEX = {}
for i, (u, v) in enumerate(EDGE_LIST):
    if u > v:
        u, v = v, u
    EDGE_INDEX[(u, v)] = i + 1  # 1-based

# Precompute a fast lookup table for edge indices (24x24)
EDGE_IDX_TABLE = [[0] * N for _ in range(N)]
for i, (u, v) in enumerate(EDGE_LIST):
    idx = i + 1
    EDGE_IDX_TABLE[u][v] = idx
    EDGE_IDX_TABLE[v][u] = idx

def get_edge_idx(u, v):
    return EDGE_IDX_TABLE[u][v]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def find_cycle(K, max_restarts=3):
        adj = [0] * N
        for i in range(K):
            u, v = EDGE_LIST[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Pre-check: every vertex must have degree >= 2
        for i in range(N):
            if bin(adj[i]).count('1') < 2:
                return None

        # Try multiple start vertices (min-degree first, then a few others)
        degs = [bin(adj[i]).count('1') for i in range(N)]
        sorted_vertices = sorted(range(N), key=lambda x: (degs[x], x))
        start_candidates = sorted_vertices[:max_restarts]

        path = [0] * N

        def get_unvisited_count(n, visited_mask):
            return bin(adj[n] & ~visited_mask).count('1')

        def dfs2(node, depth, visited_mask, start_vertex):
            if time.time() > deadline:
                return False

            if depth == N:
                if (adj[node] & (1 << start_vertex)) != 0:
                    return True
                return False

            unvisited = adj[node] & ~visited_mask
            if unvisited == 0:
                return False

            # Collect neighbors
            neighbors = []
            u = unvisited
            while u:
                lsb = u & -u
                n = lsb.bit_length() - 1
                neighbors.append(n)
                u ^= lsb

            # Warnsdorff: sort by number of unvisited neighbors (ascending)
            neighbors.sort(key=lambda n: get_unvisited_count(n, visited_mask))

            for n in neighbors:
                path[depth] = n
                if dfs2(n, depth + 1, visited_mask | (1 << n), start_vertex):
                    return True

            return False

        for sv in start_candidates:
            path[0] = sv
            visited = 1 << sv
            if dfs2(sv, 1, visited, sv):
                return path[:]
        return None

    def cycle_max_edge(cycle):
        """Compute the maximum edge index in the cycle."""
        max_idx = 0
        for i in range(N):
            u = cycle[i]
            v = cycle[(i + 1) % N]
            idx = EDGE_IDX_TABLE[u][v]
            if idx > max_idx:
                max_idx = idx
        return max_idx

    def cycle_edge_indices(cycle):
        """Return list of edge indices for the cycle."""
        return [EDGE_IDX_TABLE[cycle[i]][cycle[(i + 1) % N]] for i in range(N)]

    def soft_max(edge_indices, beta=2.0):
        """Log-sum-exp approximation of max, scaled by beta.
        As beta -> inf, this approaches the true max."""
        m = max(edge_indices)
        if m == 0:
            return 0.0
        # Use log-sum-exp for numerical stability
        exps = [math.exp(beta * (e - m)) for e in edge_indices]
        s = sum(exps)
        return m - math.log(s) / beta

    def two_three_opt_improve(cycle, time_limit):
        """Try to improve the cycle using 2-opt and 3-opt moves."""
        cycle = list(cycle)
        current_max = cycle_max_edge(cycle)
        current_soft = soft_max(cycle_edge_indices(cycle), beta=3.0)
        
        for _ in range(500):
            if time.time() > time_limit:
                break
            improved = False

            # 2-opt moves
            for i in range(N):
                if time.time() > time_limit:
                    break
                for j in range(i + 2, N):
                    if i == 0 and j == N - 1:
                        continue
                    # Edges removed: (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                    # Edges added:   (cycle[i], cycle[j]) and (cycle[i+1], cycle[j+1])
                    a, b = cycle[i], cycle[i + 1]
                    c, d = cycle[j], cycle[(j + 1) % N]
                    # Check we're not creating self-loops or duplicate edges
                    if a == c or a == d or b == c or b == d:
                        continue
                    old_e1 = EDGE_IDX_TABLE[a][b]
                    old_e2 = EDGE_IDX_TABLE[c][d]
                    new_e1 = EDGE_IDX_TABLE[a][c]
                    new_e2 = EDGE_IDX_TABLE[b][d]
                    new_max = max(old_e1 if old_e1 != new_e1 and old_e1 != new_e2 else 0,
                                  old_e2 if old_e2 != new_e1 and old_e2 != new_e2 else 0,
                                  new_e1, new_e2)
                    # Compute properly: all edges except the two removed
                    all_edges = cycle_edge_indices(cycle)
                    # Remove the two old edges and add two new ones
                    # Simpler: just compute new cycle max
                    new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                    nm = cycle_max_edge(new_cycle)
                    ns = soft_max(cycle_edge_indices(new_cycle), beta=3.0)
                    if nm < current_max or (nm == current_max and ns < current_soft - 0.1):
                        cycle = new_cycle
                        current_max = nm
                        current_soft = ns
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            # 3-opt moves: pick 3 edges to reconnect
            improved3 = False
            for i in range(N):
                if time.time() > time_limit or improved3:
                    break
                for j in range(i + 2, N):
                    if time.time() > time_limit or improved3:
                        break
                    if j == N - 1 and i == 0:
                        continue
                    for k in range(j + 1, N):
                        if time.time() > time_limit or improved3:
                            break
                        if i == 0 and k == N - 1:
                            continue
                        # Three edges to remove: (cycle[i],cycle[i+1]), (cycle[j],cycle[j+1]), (cycle[k],cycle[(k+1)%N])
                        # Try reconnecting: (cycle[i],cycle[j]), (cycle[i+1],cycle[k]), (cycle[j+1],cycle[(k+1)%N])
                        # or other permutations
                        a1, a2 = cycle[i], cycle[i + 1]
                        b1, b2 = cycle[j], cycle[(j + 1) % N]
                        c1, c2 = cycle[k], cycle[(k + 1) % N]
                        # Check no self-loops
                        if a1 == b1 or a1 == b2 or a1 == c1 or a1 == c2:
                            continue
                        if a2 == b1 or a2 == b2 or a2 == c1 or a2 == c2:
                            continue
                        if b1 == c1 or b1 == c2:
                            continue
                        if b2 == c1 or b2 == c2:
                            continue

                        # Build new cycle by reversing segments
                        # Original: ...a1-a2-...-b1-b2-...-c1-c2-...
                        # Option: a1-b1, a2-c1, b2-c2 (reverse middle segments)
                        # New: a1, b1, [b2..c1 reversed], c1, c2, [c2..a2 reversed]... 
                        # This is complex; use a simpler approach:
                        # Reverse segment [i+1..j], then reverse [j+1..k]
                        new_cycle = cycle[:]
                        # Reverse [i+1, j]
                        new_cycle[i+1:j+1] = new_cycle[i+1:j+1][::-1]
                        # Reverse [j+1, k]
                        new_cycle[j+1:k+1] = new_cycle[j+1:k+1][::-1]
                        nm = cycle_max_edge(new_cycle)
                        ns = soft_max(cycle_edge_indices(new_cycle), beta=3.0)
                        if nm < current_max or (nm == current_max and ns < current_soft - 0.1):
                            cycle = new_cycle
                            current_max = nm
                            current_soft = ns
                            improved3 = True
                            break
                        if improved3:
                            break
                    if improved3:
                        break
                if improved3:
                    break
            if improved3:
                continue

            if not improved and not improved3:
                break

        return cycle, current_max

    lo = N
    hi = len(EDGE_LIST)
    best_cycle = list(range(N))
    best_score = 10**9

    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        cycle = find_cycle(mid, max_restarts=4)
        if cycle:
            # Try to improve with 2-opt and 3-opt
            time_left = deadline - time.time()
            search_limit = time_left * 0.3
            cycle, score = two_three_opt_improve(cycle, search_limit)
            best_cycle = cycle
            best_score = score
            hi = mid - 1
        else:
            lo = mid + 1

    # Final polish: use remaining time for more local search
    if best_score < 10**9:
        time_left = deadline - time.time()
        if time_left > 1.0:
            best_cycle, best_score = two_three_opt_improve(best_cycle, time_left - 0.5)

    if best_score == 10**9:
        return list(range(N))

    return best_cycle
