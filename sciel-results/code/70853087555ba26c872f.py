"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = set()
for e in P['edges']:
    BASE.add((e[0], e[1]))
    BASE.add((e[1], e[0]))

# Candidate edges: only those with |u-v| >= 3 (|u-v|=2 makes a triangle with the path)
CANDIDATES = []
for u in range(N):
    for v in range(u + 1, N):
        if (v - u) >= 3:
            CANDIDATES.append((u, v))

def _build_adj(added):
    adj = [set() for _ in range(N)]
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for (u, v) in added:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_valid(added):
    # no duplicate / existing edge
    seen = set()
    for (u, v) in added:
        key = (min(u, v), max(u, v))
        if key in BASE or key in seen:
            return False
        seen.add(key)
    adj = _build_adj(added)
    # triangle-free
    for u in range(N):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            a = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                b = neighbors[j]
                if b in adj[a]:
                    return False
    # connected
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not visited[y]:
                visited[y] = True
                count += 1
                stack.append(y)
    if count != N:
        return False
    # diameter <= 4
    for src in range(N):
        dist = [-1] * N
        dist[src] = 0
        stack = [src]
        while stack:
            x = stack.pop()
            if dist[x] >= 4:
                continue
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    stack.append(y)
        for d in dist:
            if d == -1 or d > 4:
                return False
    return True

def construct():
    deadline = time.time() + 70.0  # safety margin before 90s kill
    best = None
    best_len = 999999

    # Candidate pool for search: prioritize hub-like and long-range edges
    # Build a smaller, structured candidate set to make combinatorial search feasible
    # Include: all edges with |u-v| in {3,4,5} plus a few longer ones, plus hub spokes
    hub_candidates = []
    for hub in [11, 12, 10, 13]:
        for v in range(N):
            if v != hub and abs(v - hub) >= 3:
                hub_candidates.append((min(hub, v), max(hub, v)))
    # Also include edges of length 3 and 4 (shortcuts that help diameter a lot)
    short = [(u, v) for (u, v) in CANDIDATES if (v - u) in (3, 4)]
    pool = list(set(hub_candidates) | set(short))
    # Deduplicate and sort
    pool = sorted(set(pool))

    # Try increasing sizes; for each size, try combinations from the pool
    # To keep it fast, cap the pool size and the combination count
    pool = pool[:60]  # cap pool

    # First, try the known-good hub solution as a fallback
    hub_sol = []
    hub = 11
    for v in range(N):
        if v != hub and abs(v - hub) >= 3:
            hub_sol.append((min(hub, v), max(hub, v)))
    if _is_valid(hub_sol):
        best = hub_sol
        best_len = len(hub_sol)

    # Local search / greedy: start from a random/structured set and try to shrink
    # We'll do a simple iterative improvement: try removing each edge, if still valid, remove.
    def _shrink(s):
        s = list(s)
        changed = True
        while changed:
            changed = False
            for i in range(len(s)):
                trial = s[:i] + s[i+1:]
                if _is_valid(trial):
                    s = trial
                    changed = True
                    break
        return s

    # Generate several seed solutions and shrink them
    seeds = [hub_sol]
    # Add some structured seeds
    for hub in [10, 11, 12, 13]:
        sol = []
        for v in range(N):
            if v != hub and abs(v - hub) >= 3:
                sol.append((min(hub, v), max(hub, v)))
        seeds.append(sol)
    # A few random seeds
    import random
    rng = random.Random(42)
    for _ in range(20):
        k = rng.randint(6, 14)
        sol = rng.sample(pool, min(k, len(pool)))
        seeds.append(sol)

    for seed in seeds:
        if time.time() > deadline:
            break
        sh = _shrink(seed)
        if len(sh) < best_len:
            best = sh
            best_len = len(sh)

    # Combinatorial search for smaller sizes if time permits
    # Try sizes from best_len-1 down to 4
    for size in range(best_len - 1, 3, -1):
        if time.time() > deadline:
            break
        # Limit number of combinations tried
        found = False
        for combo in itertools.combinations(pool, size):
            if time.time() > deadline:
                break
            if _is_valid(list(combo)):
                best = list(combo)
                best_len = size
                found = True
                break
        if found:
            continue
        # If we didn't find at this size, stop (sizes are smaller, harder)
        break

    if best is None:
        # Fallback: return a known valid solution
        return [[u, v] for (u, v) in hub_sol]
    # Convert to list of [u, v] with Python ints
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
