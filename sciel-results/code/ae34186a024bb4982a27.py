"""Combined: binary search on edge threshold + optimized backtracking with strong pruning."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']
FULL_MASK = (1 << N) - 1

# Precompute edge index
edge_index = {}
for idx, (u, v) in enumerate(EDGES, 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = idx


def _build_adj(k):
    """Build adjacency bitmask list for first k edges."""
    adj = [0] * N
    for i in range(min(k, len(EDGES))):
        u, v = EDGES[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _check_degree(adj):
    """Check all vertices have degree >= 2."""
    for i in range(N):
        if bin(adj[i]).count('1') < 2:
            return False
    return True


def _check_connected(adj):
    """Check graph is connected using BFS with bitmasks."""
    visited = 1  # start at 0
    frontier = 1
    while frontier:
        # Get all neighbors of frontier vertices
        nb = 0
        f = frontier
        while f:
            lsb = f & (-f)
            v = lsb.bit_length() - 1
            f ^= lsb
            nb |= adj[v]
        frontier = nb & ~visited
        visited |= frontier
    return visited == FULL_MASK


def _hamiltonian_backtrack(adj, deadline):
    """
    Find Hamilton cycle using backtracking with strong pruning.
    Returns path list or None.
    """
    # Precompute neighbor lists sorted by degree (Warnsdorff heuristic)
    deg = [bin(adj[i]).count('1') for i in range(N)]
    neighbor_lists = []
    for i in range(N):
        nbs = [j for j in range(N) if adj[i] & (1 << j)]
        nbs.sort(key=lambda x: deg[x])
        neighbor_lists.append(nbs)

    start = 0
    path = [start]
    visited_mask = 1 << start

    # Pruning: check if unvisited subgraph is connected (only among unvisited)
    def unvisited_connected(mask, adj_mask):
        """Check if vertices in adj_mask are connected via adj."""
        if adj_mask == 0:
            return True
        # BFS from lowest set bit
        lsb = adj_mask & (-adj_mask)
        v0 = lsb.bit_length() - 1
        visited = lsb
        frontier = lsb
        while frontier:
            nb = 0
            f = frontier
            while f:
                l = f & (-f)
                v = l.bit_length() - 1
                f ^= l
                nb |= adj[v]
            frontier = nb & adj_mask & ~visited
            visited |= frontier
        return visited == adj_mask

    def prune_check(u, mask):
        """
        Pruning check. Returns True if we can continue, False if dead end.
        u = current vertex, mask = visited mask.
        """
        unvisited = FULL_MASK ^ mask

        if unvisited == 0:
            # Must connect back to start
            return bool(adj[u] & (1 << start))

        # If only one unvisited vertex, it must connect to u and start
        if unvisited & (unvisited - 1) == 0:
            v = unvisited.bit_length() - 1
            return bool((adj[u] & (1 << v)) and (adj[v] & (1 << start)))

        # Every unvisited vertex must have at least 2 "available" neighbors
        # (neighbors that are unvisited, or are u, or are start)
        for v in range(N):
            if unvisited & (1 << v):
                # Available neighbors: unvisited neighbors + u + start
                avail = (adj[v] & unvisited)
                if u != start:
                    avail |= (adj[v] & (1 << u))
                avail |= (adj[v] & (1 << start))
                if bin(avail).count('1') < 2:
                    return False

        # Connectivity: unvisited vertices must be connected in the subgraph
        # induced by unvisited vertices, PLUS u must connect to the component,
        # AND start must connect to the component.
        # Simplified: check unvisited subgraph is connected, and both u and start
        # have at least one edge into it.
        if not unvisited_connected(unvisited, adj):
            return False
        # u must connect to at least one unvisited
        if not (adj[u] & unvisited):
            return False
        # start must connect to at least one unvisited
        if not (adj[start] & unvisited):
            return False

        return True

    def dfs(u, mask, path):
        if time.time() > deadline:
            return None

        if mask == FULL_MASK:
            if adj[u] & (1 << start):
                return path[:]
            return None

        if not prune_check(u, mask):
            return None

        # Try neighbors in Warnsdorff order
        for v in neighbor_lists[u]:
            if mask & (1 << v):
                continue
            path.append(v)
            res = dfs(v, mask | (1 << v), path)
            if res is not None:
                return res
            path.pop()

        return None

    return dfs(start, visited_mask, path)


def _exists_hamiltonian_cycle(k, deadline):
    """Check if a Hamilton cycle exists using only edges with index <= k."""
    adj = _build_adj(k)

    if not _check_degree(adj):
        return None
    if not _check_connected(adj):
        return None

    return _hamiltonian_backtrack(adj, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    max_idx = len(EDGES)

    # Binary search for smallest K with a Hamilton cycle
    low = N  # minimum edges needed
    high = max_idx
    best_cycle = None

    while low <= high:
        if time.time() > deadline - 2.0:
            break

        mid = (low + high) // 2
        remaining = deadline - time.time() - 1.0
        if remaining < 0.5:
            break

        cycle = _exists_hamiltonian_cycle(mid, deadline - 0.5)
        if cycle is not None:
            best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1

    if best_cycle is None:
        # Fallback: try with all edges
        if time.time() < deadline - 1.0:
            best_cycle = _exists_hamiltonian_cycle(max_idx, deadline - 0.5)

    if best_cycle is None:
        return list(range(N))

    return best_cycle

# EVOLVE-BLOCK-END
