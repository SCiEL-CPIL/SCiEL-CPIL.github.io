"""Triangle-free diameter-four augmentation for a 24-vertex path."""
import time
import random

P = {
    'n': 24,
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
              [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
              [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
    'diameter': 4
}


def _bfs_distances(adj, src):
    """Return list of distances from src; -1 if unreachable."""
    n = len(adj)
    dist = [-1] * n
    dist[src] = 0
    queue = [src]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _check_solution(added_edges):
    """
    Verify that base path + added_edges is:
      - triangle-free
      - connected
      - diameter <= 4
    Returns (ok, max_dist).
    """
    n = P['n']
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
    for u, v in added_edges:
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False, -1
        edge_set.add((a, b))
        adj[a].append(b)
        adj[b].append(a)

    # Triangle-free check: for each vertex, neighbors must not be adjacent to each other
    for u in range(n):
        nbrs = adj[u]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[i] in adj[nbrs[j]]:
                    return False, -1

    # Connectedness + diameter via BFS from 0
    dist0 = _bfs_distances(adj, 0)
    if any(d == -1 for d in dist0):
        return False, -1
    max_d = max(dist0)
    if max_d > 4:
        return False, max_d
    # For full diameter we need all pairs, but for a connected graph with max dist from one node
    # it's not sufficient. Let's check all pairs for safety.
    for s in range(n):
        d = _bfs_distances(adj, s)
        if any(x == -1 for x in d):
            return False, -1
        md = max(d)
        if md > 4:
            return False, md
    return True, max_d


def _candidate_edges():
    """All non-base edges (u,v) with u<v, u!=v."""
    base = set()
    for u, v in P['edges']:
        base.add((min(u, v), max(u, v)))
    cands = []
    for u in range(24):
        for v in range(u + 1, 24):
            if (u, v) not in base:
                cands.append((u, v))
    return cands


def _fast_check(added_edges):
    """
    Quick triangle-free + connectivity + diameter check.
    Uses the fact that the base graph is a path (bipartite).
    Added edges must go between opposite parity to maintain triangle-free
    (since path edges connect opposite parity, and any triangle would need
    two edges of same parity or a path shortcut creating a 3-cycle).
    
    Actually, triangle-free is more subtle. Let's just do the full check but
    with early termination.
    """
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Triangle check
    for u in range(n):
        for i in range(len(adj[u])):
            for j in range(i + 1, len(adj[u])):
                a, b = adj[u][i], adj[u][j]
                # Check if a and b are adjacent
                if a < b:
                    # Use a set for speed? For n=24 it's fine to just check
                    if b in adj[a]:
                        return False
                else:
                    if a in adj[b]:
                        return False

    # BFS from 0 for connectivity
    dist = [-1] * n
    dist[0] = 0
    q = [0]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    if any(d == -1 for d in dist):
        return False

    # Diameter: check all pairs
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    q.append(v)
        md = max(d)
        if md > 4:
            return False
    return True


def construct():
    start = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill

    # Known good solution: 11 edges
    # Strategy: bipartite edges only (even-odd) to ensure triangle-free
    # Let's use a known valid set and try to improve
    best = None
    best_count = 999

    # First, try the 11-edge solution from analysis
    # Edges: (1,14),(3,14),(5,14),(7,14),(9,14),(11,14),(13,14),(14,21),(2,23),(17,14),(19,14)
    # Wait, (14,21) is even-odd? 14 even, 21 odd. Yes.
    # But (1,14) is odd-even. Yes.
    # Let me just use a set and verify
    candidate_set = [
        [1, 14], [3, 14], [5, 14], [7, 14], [9, 14],
        [11, 14], [13, 14], [17, 14], [19, 14],
        [14, 21], [2, 23]
    ]
    if _fast_check(candidate_set):
        best = candidate_set
        best_count = len(candidate_set)

    # Try to find a smaller solution via simulated annealing
    # State: set of added edges
    # Move: add or remove one edge
    # Cost: number of edges + penalty if invalid

    all_cands = _candidate_edges()
    # For efficiency, precompute which candidate edges are even-odd (safe for triangle-free)
    # Actually, not all even-odd edges are safe, but same-parity edges are definitely problematic
    # Let's restrict to even-odd edges for the search to maintain triangle-free property
    # (since path is bipartite with parity, adding only cross-parity edges keeps it bipartite,
    #  hence triangle-free)
    safe_cands = [(u, v) for u, v in all_cands if (u % 2) != (v % 2)]

    # Simulated annealing
    if time.time() < deadline - 5:
        random.seed(42)
        # Start from best known or empty
        if best is not None:
            current = set(tuple(sorted(e)) for e in best)
        else:
            current = set()

        current_cost = len(current)
        if not _fast_check([list(e) for e in current]):
            current_cost = 100 + len(current)

        best_found = set(current)
        best_cost = current_cost

        T = 10.0
        cooling = 0.995
        min_T = 0.01

        while T > min_T and time.time() < deadline - 2:
            # Pick a random move: add or remove
            if len(current) < 20 and random.random() < 0.5:
                # Add a random safe edge
                u, v = random.choice(safe_cands)
                key = (min(u, v), max(u, v))
                if key not in current:
                    new_set = current | {key}
                    new_cost = len(new_set)
                    valid = _fast_check([list(e) for e in new_set])
                    if valid:
                        new_cost = len(new_set)
                    else:
                        new_cost = 100 + len(new_set)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                        current = new_set
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_found = set(current)
            else:
                # Remove a random edge
                if current:
                    to_remove = random.choice(list(current))
                    new_set = current - {to_remove}
                    new_cost = len(new_set)
                    valid = _fast_check([list(e) for e in new_set])
                    if valid:
                        new_cost = len(new_set)
                    else:
                        new_cost = 100 + len(new_set)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                        current = new_set
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_found = set(current)

            T *= cooling

        if best_cost < best_count:
            best = [list(e) for e in best_found]
            best_count = best_cost

    # Final polish: try to remove one edge at a time from best
    if best is not None and time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(len(best)):
                trial = best[:i] + best[i+1:]
                if _fast_check(trial):
                    best = trial
                    improved = True
                    break

    # Ensure types are correct
    result = []
    for e in best:
        result.append([int(e[0]), int(e[1])])
    return result


if __name__ == '__main__':
    res = construct()
    print(f"Added {len(res)} edges: {res}")
    ok, md = _check_solution(res)
    print(f"Valid: {ok}, max dist: {md}")
