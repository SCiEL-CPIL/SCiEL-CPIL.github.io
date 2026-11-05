import random
import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin for safety

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

        path = [0]

        def dfs(u, visited_mask):
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

    # We know K=276 (full graph) works.
    # Let's binary search.
    # To save time, we can start with a coarser search or use the knowledge that
    # the graph becomes dense quickly.
    # However, for n=24, binary search over 276 edges is ~8 steps.
    # Each step involves a DFS which might be slow.
    # Let's try to be smart:
    # 1. Check if K=n works? Unlikely.
    # 2. Check K=n+10, etc?
    # Let's just do binary search but with a time check.

    # Optimization: If we find a cycle for K, we can store it.
    # But we want the MINIMUM K.
    # The cycle found for K_min is the answer.

    # Let's try a linear search from a reasonable lower bound?
    # The minimum degree must be at least 2.
    # For K edges, average degree is 2K/n.
    # For min degree 2, we need roughly n edges? No, min degree 2 requires more.
    # Let's stick to binary search.

    # To speed up the binary search, we can cache the result of has_hamiltonian_cycle?
    # No, K changes.

    # Let's try to find a cycle for K = len(edges) first to have a fallback.
    # This guarantees we return *something* valid.
    fallback_cycle = find_hamiltonian_cycle(len(edges))
    if fallback_cycle is None:
        # Should not happen for complete graph
        return list(range(n))

    # Now binary search for the minimum K
    # We can narrow the range.
    # The graph is complete, so K_max = 276.
    # Let's try to find the smallest K.

    # We can try to increment K from a small value?
    # No, binary search is better.

    # Let's add a time check inside the binary search loop.
    # If we run out of time, return the best cycle found so far?
    # But we need the cycle for the *minimum* K to get the best score.
    # If we don't find the minimum K, we return a cycle for a larger K, which is worse.
    # So we must be careful.

    # Strategy:
    # 1. Find any valid cycle (done, fallback_cycle for K=276).
    # 2. Binary search for K_min.
    # 3. If time runs out during binary search, we might not know K_min.
    #    But we can return the fallback cycle? That gives score 276.
    #    Is there a better fallback?
    #    We can try to find a cycle for a smaller K, say K=100, K=50, etc.
    #    and keep the best one found.

    best_cycle = fallback_cycle
    best_K = len(edges)

    # Let's try some specific K values first to get a better fallback?
    # No, binary search is systematic.

    # Let's modify the binary search to update best_cycle if we find a valid one.
    # But binary search jumps around.
    # If has_hamiltonian_cycle(mid) is True, we know K_min <= mid.
    # We can try to find a cycle for mid. If we find one, update best_cycle.
    # Then high = mid - 1.
    # If False, low = mid + 1.

    # This way, if we time out, we have the best cycle found for the smallest K we tested successfully.

    # Let's implement this.

    def try_find_cycle(K):
        if time.time() > deadline:
            return None
        return find_hamiltonian_cycle(K)

    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        cycle = try_find_cycle(mid)
        if cycle:
            k_min = mid
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1

    # If we found a cycle, return it.
    # The cycle corresponds to best_K.
    # The score is the max edge index in the cycle.
    # Since the cycle is found in G_best_K, all its edges have index < best_K.
    # So the max index is < best_K.
    # The score is the 1-based arrival index of the largest edge.
    # So score = max(idx(e) for e in cycle) + 1.
    # This is <= best_K.

    if best_cycle:
        return best_cycle

    # Fallback: return the full graph cycle
    return fallback_cycle

if __name__ == "__main__":
    pass
