"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Pinned graph data from seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency structure once
def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return adj

ADJ = build_adj()

# Precompute degrees
DEG = [len(adj) for adj in ADJ]

def path_is_valid(path):
    """Verify all consecutive pairs are edges and vertices are distinct."""
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i+1] not in ADJ[path[i]]:
            return False
    return True

def greedy_grow(start, adj, visited=None, strategy='least_remaining'):
    """Grow a path greedily from start vertex."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'least_remaining':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'most_remaining':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
        elif strategy == 'random':
            v = random.choice(list(candidates))
        else:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def try_extend_both_ends(path, adj):
    """Try to extend a path from both ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending from start
        cur = path[0]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Try extending from end
        cur = path[-1]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def try_insert_vertex(path, adj, vertex):
    """Try to insert a vertex into the path at a valid position."""
    if vertex in path:
        return None
    # Check if vertex can be inserted between any two consecutive vertices
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            new_path = path[:i+1] + [vertex] + path[i+1:]
            return new_path
    # Check if vertex can be added at either end
    if vertex in adj[path[0]]:
        return [vertex] + path
    if vertex in adj[path[-1]]:
        return path + [vertex]
    return None

def insert_all_possible(path, adj, max_rounds=200):
    """Repeatedly try to insert any unvisited vertex into the path."""
    best_path = list(path)
    seen = set(best_path)
    for _ in range(max_rounds):
        improved = False
        for v in range(N):
            if v in seen:
                continue
            new_path = try_insert_vertex(best_path, adj, v)
            if new_path is not None:
                best_path = new_path
                seen.add(v)
                improved = True
                break
        if not improved:
            break
    best_path = try_extend_both_ends(best_path, adj)
    return best_path

def local_search_improve(path, adj, max_iter=100):
    """Try to improve a path by inserting unvisited vertices and 2-opt reversals."""
    if len(path) < 2:
        return path
    best_path = list(path)
    best_len = len(best_path)
    seen = set(best_path)

    for _ in range(max_iter):
        improved = False
        # Try inserting each unvisited vertex
        for v in range(N):
            if v in seen:
                continue
            new_path = try_insert_vertex(best_path, adj, v)
            if new_path is not None:
                best_path = new_path
                best_len = len(best_path)
                seen.add(v)
                improved = True
                break
        if improved:
            continue
        # Try 2-opt: reverse a segment to create new extension opportunities
        found_2opt = False
        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                # Check if reversing segment [i..j] creates a valid path
                a = best_path[i - 1]
                b = best_path[j]
                c = best_path[i]
                d = best_path[j + 1]
                # After reversal: a-b-...-c-d becomes a-b-...-c-d
                # Original: a-...-c-...-b-...-d
                # Need: a connected to b, and c connected to d
                if b in adj[a] and c in adj[d]:
                    new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                    extended = insert_all_possible(new_path, adj, max_rounds=20)
                    if len(extended) > best_len:
                        best_path = extended
                        best_len = len(best_path)
                        seen = set(best_path)
                        improved = True
                        found_2opt = True
                        break
            if found_2opt:
                break
        if not improved:
            break

    best_path = try_extend_both_ends(best_path, adj)
    return best_path

def beam_search(start, adj, beam_width=20, max_depth=None):
    """Beam search for longest path starting from start with Warnsdorff ordering."""
    if max_depth is None:
        max_depth = N
    beam = [([start], {start})]
    best_path = [start]
    best_len = 1

    for depth in range(max_depth):
        if not beam:
            break
        new_beam = []
        for path, seen in beam:
            cur = path[-1]
            candidates = adj[cur] - seen
            if not candidates:
                continue
            # Warnsdorff: prefer neighbors with fewest onward options
            candidates = sorted(candidates, key=lambda w: (len(adj[w] - seen), w))
            for v in candidates:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)
        if not new_beam:
            break
        # Keep top beam_width by length, then by Warnsdorff score
        new_beam.sort(key=lambda x: (len(x[0]), -len(adj[x[0][-1]] - x[1])))
        beam = new_beam[:beam_width]

    return best_path

def random_walk_with_backtracking(start, adj, num_tries=100):
    """Multiple random walks with occasional backtracking."""
    best_path = []
    for t in range(num_tries):
        rng = random.Random(start * 1000 + t)
        path = [start]
        seen = {start}
        for _ in range(N):
            cur = path[-1]
            candidates = list(adj[cur] - seen)
            if not candidates:
                break
            weights = [len(adj[v] - seen) + 1 for v in candidates]
            v = rng.choices(candidates, weights=weights)[0]
            path.append(v)
            seen.add(v)
        path = try_extend_both_ends(path, adj)
        if len(path) > len(best_path):
            best_path = list(path)
    return best_path

def dfs_longest_path(start, adj, time_limit=3.0):
    """Warnsdorff-guided DFS with backtracking for longest path."""
    start_time = time.time()
    best_path = [start]
    best_len = 1
    timeout = [False]

    def dfs(path, seen):
        nonlocal best_path, best_len
        if timeout[0]:
            return
        if time.time() - start_time > time_limit:
            timeout[0] = True
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        cur = path[-1]
        nbrs = list(adj[cur] - seen)
        if not nbrs:
            return
        # Warnsdorff heuristic: visit neighbor with fewest unvisited neighbors first
        nbrs.sort(key=lambda w: (len(adj[w] - seen), w))
        for v in nbrs:
            if timeout[0]:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.discard(v)

    dfs([start], {start})
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill

    best_path = []
    best_len = 0

    def consider(path):
        nonlocal best_path, best_len
        if path and len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    # Phase 1: Warnsdorff DFS from all vertices (most powerful single technique)
    for start in range(N):
        if time.time() > deadline - 10:
            break
        path = dfs_longest_path(start, ADJ, time_limit=0.8)
        path = insert_all_possible(path, ADJ, max_rounds=100)
        consider(path)

    # Phase 2: Greedy from all starts + insertion
    for start in range(N):
        if time.time() > deadline - 10:
            break
        for strategy in ['least_remaining', 'most_remaining']:
            path = greedy_grow(start, ADJ, strategy=strategy)
            path = try_extend_both_ends(path, ADJ)
            path = insert_all_possible(path, ADJ, max_rounds=100)
            consider(path)

    # Phase 3: Local search with 2-opt on best
    if time.time() < deadline - 5 and best_len > 0:
        improved = local_search_improve(best_path, ADJ, max_iter=300)
        consider(improved)

    # Phase 4: Beam search from promising starts
    if time.time() < deadline - 5:
        # Use endpoints of best path plus high-degree vertices
        promising_starts = [best_path[0], best_path[-1]]
        # Add high-degree vertices not already included
        for v in sorted(range(N), key=lambda x: -DEG[x]):
            if v not in promising_starts:
                promising_starts.append(v)
            if len(promising_starts) >= 15:
                break
        promising_starts = list(dict.fromkeys(promising_starts))
        for start in promising_starts:
            if time.time() > deadline - 5:
                break
            path = beam_search(start, ADJ, beam_width=50, max_depth=N)
            path = try_extend_both_ends(path, ADJ)
            path = insert_all_possible(path, ADJ, max_rounds=100)
            consider(path)

    # Phase 5: Random walks for diversity
    if time.time() < deadline - 3:
        for start in range(0, N, 3):
            if time.time() > deadline - 3:
                break
            path = random_walk_with_backtracking(start, ADJ, num_tries=6)
            path = try_extend_both_ends(path, ADJ)
            path = insert_all_possible(path, ADJ, max_rounds=50)
            consider(path)

    # Phase 6: Final local search polish
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, ADJ, max_iter=500)
        consider(improved)
        # One more round of insertion + extension
        final_path = insert_all_possible(improved, ADJ, max_rounds=200)
        final_path = try_extend_both_ends(final_path, ADJ)
        consider(final_path)

    # Ensure we have a valid nonempty path
    if not best_path:
        best_path = [0]

    # Final validation
    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]

    return best_path
# EVOLVE-BLOCK-END
