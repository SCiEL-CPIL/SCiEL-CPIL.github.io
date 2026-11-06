"""Rewritten program: stronger K3,3-free graph construction on 20 vertices."""
import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.88  # safety margin before 120s kill

def _has_k33(adj):
    """Check if graph has a K3,3 subgraph: 3 vertices with >= 3 common neighbours."""
    n = N
    for a in range(n):
        na = adj[a]
        for b in range(a + 1, n):
            nb = adj[b]
            common_ab = na & nb
            if len(common_ab) < 3:
                continue
            for c in range(b + 1, n):
                nc = adj[c]
                if len(common_ab & nc) >= 3:
                    return True
    return False

def _try_add_edge(adj, u, v):
    """Try adding edge (u,v). Returns True if graph stays K3,3-free."""
    if u == v or v in adj[u]:
        return False
    adj[u].add(v)
    adj[v].add(u)
    ok = not _has_k33(adj)
    if not ok:
        adj[u].discard(v)
        adj[v].discard(u)
    return ok

def _edge_swap(adj, u1, v1, u2, v2):
    """Remove (u1,v1), add (u2,v2). Returns True if valid."""
    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
        return False
    if v1 not in adj[u1]:
        return False
    if v2 in adj[u2]:
        return False
    adj[u1].discard(v1)
    adj[v1].discard(u1)
    adj[u2].add(v2)
    adj[v2].add(u2)
    if _has_k33(adj):
        adj[u2].discard(v2)
        adj[v2].discard(u2)
        adj[u1].add(v1)
        adj[v1].add(u1)
        return False
    return True

def _two_swap(adj, rem1, rem2, add1, add2):
    """Remove rem1, rem2; add add1, add2. Returns True if valid."""
    # Validate
    for (a, b) in [rem1, rem2]:
        if b not in adj[a]:
            return False
    for (a, b) in [add1, add2]:
        if b in adj[a]:
            return False
    # Check no overlap issues
    if rem1 == add1 or rem1 == add2 or rem2 == add1 or rem2 == add2:
        return False
    if rem1 == rem2 or add1 == add2:
        return False
    
    adj[rem1[0]].discard(rem1[1])
    adj[rem1[1]].discard(rem1[0])
    adj[rem2[0]].discard(rem2[1])
    adj[rem2[1]].discard(rem2[0])
    adj[add1[0]].add(add1[1])
    adj[add1[1]].add(add1[0])
    adj[add2[0]].add(add2[1])
    adj[add2[1]].add(add2[0])
    
    if _has_k33(adj):
        adj[add2[1]].discard(add2[0])
        adj[add2[0]].discard(add2[1])
        adj[add1[1]].discard(add1[0])
        adj[add1[0]].discard(add1[1])
        adj[rem2[1]].add(rem2[0])
        adj[rem2[0]].add(rem2[1])
        adj[rem1[1]].add(rem1[0])
        adj[rem1[0]].add(rem1[1])
        return False
    return True

def _make_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_count(adj):
    return sum(len(s) for s in adj) // 2

def _add_only_search(adj, deadline):
    """Repeatedly try to add any missing edge."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        random.shuffle(pairs)
        for u, v in pairs:
            if _try_add_edge(adj, u, v):
                improved = True
            if time.time() > deadline:
                break
    return adj

def _swap_search(adj, deadline):
    """Edge-swap local search."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        random.shuffle(existing)
        random.shuffle(missing)
        for u1, v1 in existing:
            if time.time() > deadline:
                break
            for u2, v2 in missing:
                if _edge_swap(adj, u1, v1, u2, v2):
                    improved = True
                    break
            if improved:
                break
    return adj

def _two_swap_search(adj, deadline):
    """Two-edge swap local search."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        random.shuffle(existing)
        random.shuffle(missing)
        for i in range(len(existing)):
            if time.time() > deadline:
                break
            u1, v1 = existing[i]
            for j in range(i + 1, len(existing)):
                u2, v2 = existing[j]
                for a1, b1 in missing:
                    if time.time() > deadline:
                        break
                    for a2, b2 in missing:
                        if (a1, b1) == (a2, b2):
                            continue
                        if _two_swap(adj, (u1, v1), (u2, v2), (a1, b1), (a2, b2)):
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return adj

def _structured_initial(seed):
    """Create structured initial graphs.
    
    Strategy: partition 20 vertices into groups and use block structure.
    K3,3-free means no 3 vertices share 3+ common neighbours.
    
    One approach: use a bipartite-like structure where each side has limited
    common neighbours. Or use a random graph with controlled density.
    """
    adj = [set() for _ in range(N)]
    rng = random.Random(seed)
    
    # Try different structured approaches
    approach = seed % 5
    
    if approach == 0:
        # K2,18 seed
        for a in range(2):
            for b in range(2, 20):
                adj[a].add(b)
                adj[b].add(a)
    elif approach == 1:
        # Random graph with p=0.5, greedily remove K3,3 violations
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.5:
                    adj[u].add(v)
                    adj[v].add(u)
        # Remove edges to eliminate K3,3
        changed = True
        while changed:
            changed = False
            edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
            rng.shuffle(edges)
            for u, v in edges:
                adj[u].discard(v)
                adj[v].discard(u)
                if not _has_k33(adj):
                    changed = True
                    break
                adj[u].add(v)
                adj[v].add(u)
    elif approach == 2:
        # Start with complete bipartite K10,10 and remove edges
        for u in range(10):
            for v in range(10, 20):
                adj[u].add(v)
                adj[v].add(u)
        # Remove edges to break K3,3
        changed = True
        while changed and time.time() < deadline:
            changed = False
            edges = [(u, v) for u in range(10) for v in range(10, 20) if v in adj[u]]
            rng.shuffle(edges)
            for u, v in edges:
                adj[u].discard(v)
                adj[v].discard(u)
                if not _has_k33(adj):
                    changed = True
                    break
                adj[u].add(v)
                adj[v].add(u)
    elif approach == 3:
        # Random with p=0.4
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.4:
                    adj[u].add(v)
                    adj[v].add(u)
        changed = True
        while changed:
            changed = False
            edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
            rng.shuffle(edges)
            for u, v in edges:
                adj[u].discard(v)
                adj[v].discard(u)
                if not _has_k33(adj):
                    changed = True
                    break
                adj[u].add(v)
                adj[v].add(u)
    else:
        # Turan-like: split into 4 groups of 5, connect between groups
        groups = [list(range(i * 5, (i + 1) * 5)) for i in range(4)]
        for gi in range(4):
            for gj in range(gi + 1, 4):
                for u in groups[gi]:
                    for v in groups[gj]:
                        if rng.random() < 0.7:
                            adj[u].add(v)
                            adj[v].add(u)
        # Fix K3,3
        changed = True
        while changed:
            changed = False
            edges = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
            rng.shuffle(edges)
            for u, v in edges:
                adj[u].discard(v)
                adj[v].discard(u)
                if not _has_k33(adj):
                    changed = True
                    break
                adj[u].add(v)
                adj[v].add(u)
    
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    best_adj = None
    best_count = 0
    
    # Try multiple structured starts
    for seed in range(30):
        if time.time() > deadline:
            break
        adj = _structured_initial(seed)
        adj = _add_only_search(adj, deadline)
        adj = _swap_search(adj, deadline)
        if time.time() < deadline - 5:
            adj = _two_swap_search(adj, deadline)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    # If we still have time, do more random restarts
    if best_adj is None:
        adj = _structured_initial(0)
        adj = _add_only_search(adj, deadline)
        count = _edge_count(adj)
        best_count = count
        best_adj = [set(s) for s in adj]
    
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
