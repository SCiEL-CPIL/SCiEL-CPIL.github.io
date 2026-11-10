"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_regular(adj, subset):
    """Check if the induced subgraph on subset is regular (all vertices same degree within subset)."""
    s = list(subset)
    k = len(s)
    if k < 6:
        return False
    idx = {v: i for i, v in enumerate(s)}
    degs = [0] * k
    for i in range(k):
        u = s[i]
        for j in range(i + 1, k):
            v = s[j]
            if adj[u][v]:
                degs[i] += 1
                degs[j] += 1
    d0 = degs[0]
    for d in degs[1:]:
        if d != d0:
            return False
    return True

def _count_conflicts(adj, n):
    """Count number of subsets of size >= 6 that induce a regular subgraph."""
    bad = 0
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            if _is_regular(adj, subset):
                bad += 1
    return bad

def _count_conflicts_fast(adj, n):
    """Count conflicts, only checking subsets of size 6 (since any larger regular
    subgraph contains a regular sub-subset of size 6? Not necessarily. But checking
    all sizes is expensive. Let's check size 6 primarily and a sample of larger.)"""
    bad = 0
    # Size 6 subsets
    for subset in combinations(range(n), 6):
        if _is_regular(adj, subset):
            bad += 1
    return bad

def _find_bad_subsets(adj, n, max_k=8):
    """Find all bad subsets up to size max_k."""
    bad = []
    for k in range(6, min(max_k, n + 1)):
        for subset in combinations(range(n), k):
            if _is_regular(adj, subset):
                bad.append(set(subset))
    return bad

def _local_search(adj, n, deadline, seed_rng):
    """Local search: flip edges to reduce number of bad subsets."""
    best_adj = [row[:] for row in adj]
    best_count = _count_conflicts(best_adj, n)
    if best_count == 0:
        return best_adj, 0

    current_adj = [row[:] for row in adj]
    current_count = best_count

    while time.time() < deadline:
        # Find a bad subset
        bad_subsets = _find_bad_subsets(current_adj, n, max_k=7)
        if not bad_subsets:
            # Verify with full check
            if _count_conflicts(current_adj, n) == 0:
                return current_adj, 0
            # There's a bad subset of size > 7; try flipping random edges
            for _ in range(20):
                u = seed_rng.randint(0, n - 1)
                v = seed_rng.randint(0, n - 1)
                if u == v:
                    continue
                current_adj[u][v] = not current_adj[u][v]
                current_adj[v][u] = current_adj[u][v]
                new_count = _count_conflicts(current_adj, n)
                if new_count < current_count:
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = [row[:] for row in current_adj]
                else:
                    current_adj[u][v] = not current_adj[u][v]
                    current_adj[v][u] = current_adj[u][v]
            continue

        # Pick a random bad subset
        bad_set = seed_rng.choice(bad_subsets)
        bad_list = list(bad_set)
        k = len(bad_list)

        # Try flipping each edge within the bad subset
        improved = False
        for i in range(k):
            for j in range(i + 1, k):
                u = bad_list[i]
                v = bad_list[j]
                # Flip edge
                current_adj[u][v] = not current_adj[u][v]
                current_adj[v][u] = current_adj[u][v]
                new_count = _count_conflicts(current_adj, n)
                if new_count < current_count:
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = [row[:] for row in current_adj]
                    improved = True
                    break
                else:
                    # Revert
                    current_adj[u][v] = not current_adj[u][v]
                    current_adj[v][u] = current_adj[u][v]
            if improved:
                break

        if not improved:
            # Try flipping random edges in the bad subset or globally
            for _ in range(10):
                if seed_rng.random() < 0.5:
                    # Random edge in bad subset
                    i = seed_rng.randint(0, k - 1)
                    j = seed_rng.randint(0, k - 1)
                    if i == j:
                        continue
                    u = bad_list[i]
                    v = bad_list[j]
                else:
                    u = seed_rng.randint(0, n - 1)
                    v = seed_rng.randint(0, n - 1)
                    if u == v:
                        continue
                current_adj[u][v] = not current_adj[u][v]
                current_adj[v][u] = current_adj[u][v]
                new_count = _count_conflicts(current_adj, n)
                if new_count < current_count:
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = [row[:] for row in current_adj]
                    break
                else:
                    current_adj[u][v] = not current_adj[u][v]
                    current_adj[v][u] = current_adj[u][v]

    return best_adj, best_count

def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    return edges

def _is_valid(adj, n):
    """Full validation: no subset of size >= 6 induces a regular subgraph."""
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            if _is_regular(adj, subset):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    rng = random.Random(42)

    best_n = 5
    best_edges = []

    # Try increasing n from 6 upward
    for n in range(6, 23):
        if time.time() > deadline:
            break

        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break

            # Random graph with edge probability ~0.5
            adj = [[0] * n for _ in range(n)]
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.5:
                        adj[u][v] = 1
                        adj[v][u] = 1

            # Quick check: count size-6 conflicts
            count6 = 0
            for subset in combinations(range(n), 6):
                if _is_regular(adj, subset):
                    count6 += 1
                    if count6 > 50:
                        break
            if count6 > 50:
                continue  # Too many conflicts, skip

            # Local search to repair
            adj_repaired, final_count = _local_search(adj, n, deadline, rng)

            if final_count == 0:
                # Verify fully
                if _is_valid(adj_repaired, n):
                    edges = _adj_to_edges(adj_repaired, n)
                    best_n = n
                    best_edges = edges
                    found = True
                    break

        if found:
            # Try to go higher
            continue
        else:
            # Could not find valid graph for this n; break
            break

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
