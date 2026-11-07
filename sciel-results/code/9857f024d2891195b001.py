"""Solve dense odd-order graphs avoiding equal-degree path endpoints.

Validity: no path s-u-v-t (s!=t) has deg(s)==deg(t).
Equivalent: for every vertex u, all neighbors of u have distinct degrees.
Strategy: degree-partition construction + greedy edge addition + repair.
"""
import time
import random

N = 15
DEADLINE = 75.0  # safety margin under 90s kill


def _is_valid(adj, n):
    """Check: for every u, neighbors have distinct degrees."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        seen = set()
        for v in adj[u]:
            d = deg[v]
            if d in seen:
                return False
            seen.add(d)
    return True


def _violation_count(adj, n):
    """Count number of (u, pair-of-equal-degree-neighbors) violations."""
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for u in range(n):
        from collections import Counter
        c = Counter(deg[v] for v in adj[u])
        for d, cnt in c.items():
            if cnt > 1:
                count += cnt - 1
    return count


def _greedy_build(partition, n, seed):
    """Build a graph respecting: vertices in same partition group never adjacent.
    partition: list of group ids per vertex."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    # All candidate edges between different groups
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if partition[i] != partition[j]:
                candidates.append((i, j))
    rng.shuffle(candidates)
    for i, j in candidates:
        # Tentatively add
        adj[i].add(j)
        adj[j].add(i)
        if _is_valid(adj, n):
            pass
        else:
            adj[i].discard(j)
            adj[j].discard(i)
    return adj


def _repair(adj, n):
    """Remove edges to fix violations."""
    improved = True
    while improved:
        improved = False
        deg = [len(adj[i]) for i in range(n)]
        # Find violating vertex
        for u in range(n):
            from collections import Counter
            c = Counter(deg[v] for v in adj[u])
            if any(cnt > 1 for cnt in c.values()):
                # Remove one edge to a neighbor with duplicated degree
                seen = {}
                for v in sorted(adj[u], key=lambda x: -deg[x]):
                    d = deg[v]
                    if d in seen:
                        # Remove edge u-v (the later one)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        improved = True
                        break
                    seen[d] = v
                if improved:
                    break
    return adj


def _try_partition(partition, n, seed):
    adj = _greedy_build(partition, n, seed)
    adj = _repair(adj, n)
    if _is_valid(adj, n):
        return adj
    return None


def construct():
    start = time.time()
    best_adj = None
    best_count = -1

    # Seed program baseline: star on 14 edges
    star = [set() for _ in range(N)]
    for v in range(1, N):
        star[0].add(v)
        star[v].add(0)
    best_adj = star
    best_count = 14

    # Try various degree partitions
    # Partition = assignment of each vertex to a group; same group => no edges between them
    # Groups of equal size give symmetric degrees potential

    partitions = []

    # 5 groups of 3 (disjoint triangles candidate)
    partitions.append([i // 3 for i in range(N)])
    # 3 groups of 5
    partitions.append([i // 5 for i in range(N)])
    # 15 groups of 1 (no restriction - full search)
    partitions.append([i for i in range(N)])
    # 5 groups: sizes 3,3,3,3,3
    partitions.append([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4])
    # 3 groups: 5,5,5
    partitions.append([0]*5 + [1]*5 + [2]*5)
    # 7+8 split
    partitions.append([0]*7 + [1]*8)
    # 4 groups: 4,4,4,3
    partitions.append([0]*4 + [1]*4 + [2]*4 + [3]*3)
    # 2 groups: 7,8
    partitions.append([0]*7 + [1]*8)
    # 6 groups: 3,3,2,2,2,3
    partitions.append([0,0,0,1,1,2,2,3,3,4,4,5,5,5,5][:15])
    # Actually let me be more systematic
    # Random partitions
    for s in range(20):
        rng = random.Random(1000 + s)
        p = [rng.randint(0, 4) for _ in range(N)]
        partitions.append(p)

    seen_partitions = set()
    for p in partitions:
        key = tuple(p)
        if key in seen_partitions:
            continue
        seen_partitions.add(key)

        for seed in range(5):
            if time.time() - start > DEADLINE:
                break
            adj = _try_partition(p, N, seed)
            if adj is not None:
                cnt = sum(len(adj[i]) for i in range(N)) // 2
                if cnt > best_count:
                    best_count = cnt
                    best_adj = adj

    # Local search: try to add edges to best_adj
    if time.time() - start < DEADLINE - 5:
        adj = [set(s) for s in best_adj]
        improved = True
        while improved and time.time() - start < DEADLINE - 5:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    if j not in adj[i]:
                        adj[i].add(j)
                        adj[j].add(i)
                        if _is_valid(adj, N):
                            cnt = sum(len(adj[k]) for k in range(N)) // 2
                            if cnt > best_count:
                                best_count = cnt
                                best_adj = [set(s) for s in adj]
                                improved = True
                            else:
                                adj[i].discard(j)
                                adj[j].discard(i)
                        else:
                            adj[i].discard(j)
                            adj[j].discard(i)

    # Convert to list of [u,v] with proper types
    edges = []
    for i in range(N):
        for j in adj[i]:
            if i < j:
                edges.append([int(i), int(j)])

    # Final validity check
    adj_final = [set() for _ in range(N)]
    for e in edges:
        adj_final[e[0]].add(e[1])
        adj_final[e[1]].add(e[0])
    if not _is_valid(adj_final, N):
        # Fallback to star
        edges = [[0, v] for v in range(1, N)]

    return edges
