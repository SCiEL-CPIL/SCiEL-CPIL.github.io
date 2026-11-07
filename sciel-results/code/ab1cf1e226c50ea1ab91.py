"""Densest graph on 15 vertices avoiding equal-degree path endpoints.

Strategy:
- Greedy construction with randomized edge ordering.
- Local search: try adding missing edges, and 1-for-1 swaps.
- Multiple restarts with different seeds.
- Use a fast incremental validity check.

The constraint: for every simple path s-u-v-t (3 edges), deg[s] != deg[t] in the full graph.
"""
import random
import time

# EVOLVE-BLOCK-START
N = 15


def _is_valid_path(adj, deg, s, u, v, t):
    """Check if path s-u-v-t violates the constraint (deg[s] == deg[t])."""
    return deg[s] == deg[t]


def _check_all_paths(adj, deg):
    """Check all simple 3-edge paths. Returns True if valid."""
    n = N
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True


def _try_add_edge(adj, deg, a, b):
    """Try to add edge (a,b). Returns True if valid and added."""
    n = N
    da, db = deg[a], deg[b]
    # Check all existing edges (u,v): if a in N(u)\{v} and b in N(v)\{u}, path a-u-v-b exists
    # Also: if b in N(u)\{v} and a in N(v)\{u}, path b-u-v-a exists
    # In both cases, we need deg[a] != deg[b]
    if da == db:
        # If degrees are equal, we need to ensure no path a-?-?-b or b-?-?-a exists
        # A path a-u-v-b exists if: a~u, u~v, v~b, a!=v, b!=u, a!=b (a!=b is given)
        # More precisely: u in N(a)\{b} ∩ N(v) where v in N(b)\{a}
        # Let's check: for each u in N(a)\{b}, for each v in N(b)\{a}, if u~v and u!=b and v!=a
        for u in adj[a]:
            if u == b:
                continue
            for v in adj[b]:
                if v == a:
                    continue
                if v in adj[u]:
                    # Path a-u-v-b exists
                    return False
    # If da != db, the edge is always safe to add (no violation possible)
    # Actually wait - we need to check: adding edge (a,b) creates new paths.
    # New paths involving (a,b) as one of the three edges:
    # 1. (a,b) is the middle edge: s-a-b-t where s in N(a)\{b}, t in N(b)\{a}
    #    Need deg[s] != deg[t] for all such s,t
    # 2. (a,b) is the first edge: a-b-u-t where u in N(b)\{a}, t in N(u)\{b}
    #    Need deg[a] != deg[t]
    # 3. (a,b) is the last edge: s-u-b-a where s in N(u)\{b}, u in N(b)\{a}
    #    Need deg[s] != deg[a]
    # 4. (a,b) is the first edge (reversed): b-a-u-t where u in N(a)\{b}, t in N(u)\{a}
    #    Need deg[b] != deg[t]
    # 5. (a,b) is the last edge (reversed): s-u-a-b where s in N(u)\{a}, u in N(a)\{b}
    #    Need deg[s] != deg[b]
    
    # Case 1: s-a-b-t
    for s in adj[a]:
        if s == b:
            continue
        for t in adj[b]:
            if t == a:
                continue
            if deg[s] == deg[t]:
                return False
    
    # Case 2: a-b-u-t (a is endpoint)
    for u in adj[b]:
        if u == a:
            continue
        for t in adj[u]:
            if t == b:
                continue
            if deg[a] == deg[t]:
                return False
    
    # Case 3: s-u-b-a (a is endpoint)
    for u in adj[b]:
        if u == a:
            continue
        for s in adj[u]:
            if s == b:
                continue
            if deg[s] == deg[a]:
                return False
    
    # Case 4: b-a-u-t (b is endpoint)
    for u in adj[a]:
        if u == b:
            continue
        for t in adj[u]:
            if t == a:
                continue
            if deg[b] == deg[t]:
                return False
    
    # Case 5: s-u-a-b (b is endpoint)
    for u in adj[a]:
        if u == b:
            continue
        for s in adj[u]:
            if s == a:
                continue
            if deg[s] == deg[b]:
                return False
    
    # All checks passed, add the edge
    adj[a].add(b)
    adj[b].add(a)
    deg[a] += 1
    deg[b] += 1
    return True


def _remove_edge(adj, deg, a, b):
    """Remove edge (a,b). Assumes it exists."""
    adj[a].discard(b)
    adj[b].discard(a)
    deg[a] -= 1
    deg[b] -= 1


def _greedy_construct(n, rng, deadline):
    """Greedy: try all edges in random order, add if valid."""
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    count = 0
    for (a, b) in edges:
        if time.time() > deadline:
            break
        if _try_add_edge(adj, deg, a, b):
            count += 1
    return adj, deg, count


def _local_search(adj, deg, rng, deadline, iters=500):
    """Local search: try adding missing edges; if stuck, try 1-for-1 swaps."""
    n = N
    count = sum(len(s) for s in adj) // 2
    best_adj = [set(s) for s in adj]
    best_deg = list(deg)
    best_count = count

    no_improve = 0
    for it in range(iters):
        if time.time() > deadline:
            break
        # Collect missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    missing.append((i, j))
        rng.shuffle(missing)
        improved = False
        for (a, b) in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, deg, a, b):
                count += 1
                improved = True
                if count > best_count:
                    best_count = count
                    best_adj = [set(s) for s in adj]
                    best_deg = list(deg)
                    no_improve = 0
                break
        if not improved:
            # Try 1-for-1 swaps
            all_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            rng.shuffle(all_edges)
            swap_found = False
            for (ra, rb) in all_edges:
                if time.time() > deadline:
                    break
                _remove_edge(adj, deg, ra, rb)
                for (a, b) in missing:
                    if (a == ra and b == rb) or (a == rb and b == ra):
                        continue
                    if _try_add_edge(adj, deg, a, b):
                        swap_found = True
                        break
                    else:
                        _remove_edge(adj, deg, a, b)
                if swap_found:
                    break
                else:
                    _add_back(adj, deg, ra, rb)
            if not swap_found:
                no_improve += 1
                if no_improve > 20:
                    break
            else:
                no_improve = 0

    return best_adj, best_deg, best_count


def _add_back(adj, deg, a, b):
    """Add edge back without validity check (for internal use after failed swap)."""
    adj[a].add(b)
    adj[b].add(a)
    deg[a] += 1
    deg[b] += 1


def _multi_start(n, rng, deadline):
    """Multiple greedy starts + local search refinement."""
    best_count = -1
    best_adj = None
    best_deg = None

    n_starts = 30
    for k in range(n_starts):
        if time.time() > deadline - 15.0:
            break
        adj, deg, cnt = _greedy_construct(n, rng, deadline - 15.0)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
            best_deg = list(deg)
        # Refine with local search
        adj2, deg2, cnt2 = _local_search(adj, deg, rng, deadline - 10.0, iters=300)
        if cnt2 > best_count:
            best_count = cnt2
            best_adj = [set(s) for s in adj2]
            best_deg = list(deg2)

    # Final polish
    if best_adj is not None and time.time() < deadline - 5.0:
        adj3, deg3, cnt3 = _local_search(best_adj, best_deg, rng, deadline - 2.0, iters=500)
        if cnt3 > best_count:
            best_count = cnt3
            best_adj = [set(s) for s in adj3]
            best_deg = list(deg3)

    if best_adj is None:
        best_adj = [set() for _ in range(n)]

    return best_adj


def construct():
    n = N
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    rng = random.Random(42)

    adj = _multi_start(n, rng, deadline)

    # Final verification
    deg = [len(adj[i]) for i in range(n)]
    if not _check_all_paths(adj, deg):
        # Fallback: rebuild with full check
        adj2 = [set() for _ in range(n)]
        deg2 = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng2 = random.Random(999)
        rng2.shuffle(edges)
        for (a, b) in edges:
            adj2[a].add(b)
            adj2[b].add(a)
            deg2[a] += 1
            deg2[b] += 1
            if not _check_all_paths(adj2, deg2):
                adj2[a].discard(b)
                adj2[b].discard(a)
                deg2[a] -= 1
                deg2[b] -= 1
        adj = adj2

    # Build edge list with proper types
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    return edges
# EVOLVE-BLOCK-END
