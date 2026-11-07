"""Triangle-free diameter-4 augmentation: exact search with branch-and-bound."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_D = P['diameter']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}


def _base_adj():
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    return adj


_BASE_ADJ = _base_adj()


def _bfs_dist(adj, s):
    dist = [-1] * _N
    dist[s] = 0
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def _diameter(adj):
    best = 0
    for s in range(_N):
        dist = _bfs_dist(adj, s)
        if -1 in dist:
            return 10**9
        m = max(dist)
        if m > best:
            best = m
    return best


def _diameter_violations(adj):
    """Return list of (u,v) pairs with dist > _D."""
    violations = []
    for s in range(_N):
        dist = _bfs_dist(adj, s)
        for v in range(s + 1, _N):
            if dist[v] > _D:
                violations.append((s, v))
    return violations


def _is_triangle_free(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    return True


def _valid(added):
    if not _is_triangle_free(added):
        return False
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return _diameter(adj) <= _D


def _candidate_edges():
    """All non-base unordered pairs."""
    return [(u, v) for u in range(_N) for v in range(u + 1, _N) if (u, v) not in _BASE_SET]


def _conflict_map(candidates):
    """For each candidate edge, the set of other candidates that would form a triangle."""
    conf = {e: set() for e in candidates}
    for i in range(len(candidates)):
        u1, v1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            u2, v2 = candidates[j]
            # Check all shared-vertex cases
            if u1 == u2 and v1 in _BASE_ADJ[v2]:
                conf[candidates[i]].add(candidates[j])
                conf[candidates[j]].add(candidates[i])
            if u1 == v2 and v1 in _BASE_ADJ[u2]:
                conf[candidates[i]].add(candidates[j])
                conf[candidates[j]].add(candidates[i])
            if v1 == u2 and u1 in _BASE_ADJ[v2]:
                conf[candidates[i]].add(candidates[j])
                conf[candidates[j]].add(candidates[i])
            if v1 == v2 and u1 in _BASE_ADJ[u2]:
                conf[candidates[i]].add(candidates[j])
                conf[candidates[j]].add(candidates[i])
    return conf


def _edge_covers(added, edge, violations):
    """Check which violations are fixed by adding this edge to current added set."""
    u, v = edge
    # Build adjacency with added + this edge
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    adj[u].add(v)
    adj[v].add(u)
    
    covered = 0
    for (a, b) in violations:
        # Quick check: does the new edge help?
        # dist(a,b) in new graph <= 4?
        # Use the fact that the only new path is through (u,v)
        # dist_new(a,b) = min(dist_old(a,b), dist_old(a,u)+1+dist_old(v,b), dist_old(a,v)+1+dist_old(u,b))
        # But dist_old is in graph with `added` (without this edge)
        # We don't have dist_old precomputed for all pairs, so let's just check directly
        # Actually, let's compute BFS from a in the new graph
        pass
    
    # Simpler: just check if each violation pair now has dist <= 4
    # This is O(|violations| * N) which is fine for small violations
    for (a, b) in violations:
        # BFS from a
        dist = [-1] * _N
        dist[a] = 0
        dq = deque([a])
        while dq:
            x = dq.popleft()
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    dq.append(y)
        if dist[b] <= _D:
            covered += 1
    return covered


def _search(added, candidates, conf, deadline, best_found):
    """Branch and bound: try to find a valid solution with <= best_found edges."""
    if time.time() > deadline:
        return None
    
    # Check if current added is already valid
    if _valid(added):
        return list(added)
    
    if not candidates:
        return None
    
    # Get current violations
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    
    violations = _diameter_violations(adj)
    if not violations:
        return list(added)
    
    # If we already have >= best_found edges, we can't improve
    if len(added) >= best_found:
        return None
    
    # Score each candidate by how many violations it covers
    scored = []
    for e in candidates:
        if any(s in conf[e] for s in set(added)):
            continue
        cov = _edge_covers(added, e, violations)
        if cov > 0:
            scored.append((cov, e))
    
    if not scored:
        return None
    
    # Sort by coverage descending
    scored.sort(key=lambda x: -x[0])
    
    # Try each candidate in order
    for cov, e in scored:
        if time.time() > deadline:
            return None
        
        new_added = added + [e]
        new_candidates = [c for c in candidates if c != e and not any(s in conf[c] for s in set(new_added))]
        
        result = _search(new_added, new_candidates, conf, deadline, best_found)
        if result is not None:
            return result
    
    return None


def _greedy_init():
    """Greedy set cover to get initial solution."""
    candidates = _candidate_edges()
    conf = _conflict_map(candidates)
    
    # Get initial violations in base path
    violations = _diameter_violations(_BASE_ADJ)
    
    added = []
    added_set = set()
    remaining_violations = set(violations)
    
    while remaining_violations:
        best_e = None
        best_cov = 0
        for e in candidates:
            if e in added_set:
                continue
            if any(s in conf[e] for s in added_set):
                continue
            cov = _edge_covers(added, e, list(remaining_violations))
            if cov > best_cov:
                best_cov = cov
                best_e = e
        if best_e is None:
            break
        added.append(best_e)
        added_set.add(best_e)
        # Update remaining violations
        adj = [set() for _ in range(_N)]
        for (a, b) in _BASE:
            adj[a].add(b)
            adj[b].add(a)
        for (a, b) in added:
            adj[a].add(b)
            adj[b].add(a)
        remaining_violations = set(_diameter_violations(adj))
    
    return added


def _try_remove(added):
    """Try to remove edges one at a time."""
    cur = list(added)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur


def _swap_search(added, deadline):
    """Try swapping one edge for another to reduce count."""
    cur = list(added)
    candidates = _candidate_edges()
    conf = _conflict_map(candidates)
    
    for _ in range(50):  # Limited iterations
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(cur)):
            # Try removing edge i
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
            # Try swapping edge i with some other edge
            for j in range(len(cur)):
                if j == i:
                    continue
            # Try adding a new edge that allows removing an existing one
            for new_e in candidates:
                if new_e in cur:
                    continue
                if any(s in conf[new_e] for s in set(cur)):
                    continue
                # Try: cur - cur[i] + new_e
                trial = cur[:i] + cur[i+1:] + [new_e]
                if _valid(trial) and len(trial) < len(cur):
                    cur = trial
                    improved = True
                    break
                # Or: cur + new_e - some other edge
                trial2 = cur + [new_e]
                if _valid(trial2):
                    # Now try to remove an edge
                    for k in range(len(trial2)):
                        trial3 = trial2[:k] + trial2[k+1:]
                        if _valid(trial3) and len(trial3) < len(cur):
                            cur = trial3
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    
    return cur


def construct():
    deadline = time.time() + 80.0
    
    # Get initial solution via greedy
    best = _greedy_init()
    if best is None or not _valid(best):
        # Fallback: use a known pattern
        best = [(i, i + 6) for i in range(18)]
        if not _valid(best):
            # Even more fallback
            best = [(i, i + 5) for i in range(19)]
            if not _valid(best):
                return [[u, v] for u in range(_N) for v in range(u + 1, _N)
                        if (u - v) % 2 and (u, v) not in _BASE_SET]
    
    # Try to improve via removal
    best = _try_remove(best)
    
    # Branch and bound to find smaller solutions
    candidates = _candidate_edges()
    conf = _conflict_map(candidates)
    
    # Try to find a solution with fewer edges
    for target_size in range(len(best) - 1, 0, -1):
        if time.time() > deadline:
            break
        # Try to find a solution of exactly target_size edges
        # Use greedy to get a starting point, then search
        result = _search([], candidates, conf, deadline, target_size)
        if result is not None:
            best = _try_remove(result)
            break
    
    # Also try swap search
    if time.time() < deadline:
        swapped = _swap_search(best, deadline)
        if len(swapped) < len(best):
            best = swapped
    
    # Final validation and type conversion
    if not _valid(best):
        # Emergency fallback
        fallback = [(i, i + 6) for i in range(18)]
        if _valid(fallback):
            best = fallback
        else:
            return [[u, v] for u in range(_N) for v in range(u + 1, _N)
                    if (u - v) % 2 and (u, v) not in _BASE_SET]
    
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
