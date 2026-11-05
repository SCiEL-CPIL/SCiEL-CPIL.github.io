import random
import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin

    # Generate the pinned edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    r = random.Random(614)
    r.shuffle(edges)

    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx

    # Precompute adjacency lists sorted by edge arrival index
    adj_raw = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj_raw[u].append((v, idx))
        adj_raw[v].append((u, idx))

    for i in range(n):
        adj_raw[i].sort(key=lambda x: x[1])

    full_mask = (1 << n) - 1

    # Helper: Check if G_K has a Hamilton cycle and return it.
    def find_hamiltonian_cycle(K):
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        # Quick check: min degree
        for i in range(n):
            if adj_mask[i].bit_count() < 2:
                return None

        # Connectivity check
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            m = adj_mask[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return None

        # Iterative DFS with Warnsdorff's rule
        path = [0]
        visited_mask = 1

        # Stack stores (current_node, next_candidate_index, candidates_list)
        # We'll use a different approach: maintain the path and backtrack manually

        def get_candidates(u, visited_mask):
            """Get sorted candidates for vertex u using Warnsdorff's rule."""
            candidates = adj_mask[u] & (~visited_mask)
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj_mask[v] & ~visited_mask).bit_count()
                cand_list.append((deg, v))
            cand_list.sort()
            return [v for _, v in cand_list]

        # Iterative DFS
        path = [0]
        visited_mask = 1
        candidate_lists = [get_candidates(0, visited_mask)]
        next_indices = [0]

        while True:
            if time.time() > deadline:
                return None

            if len(path) == n:
                # Check if we can close the cycle
                if (adj_mask[path[-1]] & 1) != 0:
                    return path
                # Can't close, backtrack
                break

            u = path[-1]
            cands = candidate_lists[-1]
            next_idx = next_indices[-1]

            if next_idx < len(cands):
                v = cands[next_idx]
                next_indices[-1] += 1
                path.append(v)
                visited_mask |= (1 << v)
                candidate_lists.append(get_candidates(v, visited_mask))
                next_indices.append(0)
            else:
                # Backtrack
                path.pop()
                visited_mask ^= (1 << path[-1]) if len(path) > 0 else 0
                # Wait, this is wrong. Let me redo this.
                break

        # The iterative approach above has issues with backtracking. Let me use a cleaner recursive approach but with optimizations.

        # Actually, let me just use the recursive DFS but add more pruning.

        path = [0]

        def dfs(u, visited_mask):
            if time.time() > deadline:
                return False

            if len(path) == n:
                return (adj_mask[u] & 1) != 0

            candidates = adj_mask[u] & (~visited_mask)

            # Warnsdorff's rule: try neighbors with fewest options first
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (adj_mask[v] & ~visited_mask).bit_count()
                cand_list.append((deg, v))

            cand_list.sort()

            for deg, v in cand_list:
                # Pruning: if v is the last unvisited vertex, check if it can connect to start
                if (visited_mask | (1 << v)) == full_mask:
                    if (adj_mask[v] & 1) == 0:
                        continue

                path.append(v)
                if dfs(v, visited_mask | (1 << v)):
                    return True
                path.pop()

            return False

        if dfs(0, 1):
            return path
        return None

    # Binary search for minimum K
    low = n
    high = len(edges)
    k_min = len(edges)

    # Let's try to find a cycle for K = len(edges) first to have a fallback.
    fallback_cycle = find_hamiltonian_cycle(len(edges))
    if fallback_cycle is None:
        # Should not happen for complete graph
        return list(range(n))

    best_cycle = fallback_cycle
    best_K = len(edges)

    # Binary search with time management
    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        cycle = find_hamiltonian_cycle(mid)
        if cycle:
            k_min = mid
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1

    if best_cycle:
        return best_cycle

    return fallback_cycle

if __name__ == "__main__":
    pass
