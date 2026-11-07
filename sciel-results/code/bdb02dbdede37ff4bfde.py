"""Packing edge-disjoint Hamiltonian cycles on a 21-vertex graph."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_ham_cycle(cycle, adj):
    n = len(cycle)
    if n < 3:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if v not in adj[u]:
            return False
    return True

def _edge_set_of_cycle(cycle):
    n = len(cycle)
    es = set()
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        es.add((min(u, v), max(u, v)))
    return es

def _find_ham_cycles_bfs(adj, n, limit=200, seed=None):
    """Find Hamiltonian cycles via randomized DFS with pruning."""
    rng = random.Random(seed)
    found = []
    seen = set()
    start = 0  # fix start to reduce symmetry
    # Precompute neighbor lists
    nbrs = [sorted(adj[v]) for v in range(n)]
    
    def dfs(path, used):
        if len(found) >= limit:
            return
        if len(path) == n:
            # Check closing edge
            last = path[-1]
            first = path[0]
            if first in adj[last]:
                # Canonicalize: rotate so min vertex is first, then pick direction
                cyc = list(path)
                # Rotate to start with smallest vertex
                min_idx = cyc.index(min(cyc))
                cyc = cyc[min_idx:] + cyc[:min_idx]
                # Two directions
                rev = [cyc[0]] + cyc[1:][::-1]
                canon = tuple(cyc) if cyc < rev else tuple(rev)
                if canon not in seen:
                    seen.add(canon)
                    found.append(list(canon))
            return
        cur = path[-1]
        # Neighbors not yet used
        cands = [v for v in nbrs[cur] if not used[v]]
        rng.shuffle(cands)
        for v in cands:
            # Pruning: if v is last unvisited, must connect back to start
            remaining = [u for u in range(n) if not used[u] and u != v]
            if remaining:
                # v must be able to continue; also check that remaining vertices
                # can eventually reach start
                pass
            used[v] = True
            path.append(v)
            dfs(path, used)
            path.pop()
            used[v] = False
            if len(found) >= limit:
                return
    
    used = [False] * n
    used[start] = True
    dfs([start], used)
    return found

def _try_pack(cycles, adj, n, max_cycles=3):
    """Try to greedily select edge-disjoint cycles."""
    used_edges = set()
    selected = []
    # Sort cycles: prefer shorter "canonical" or just try all
    for cyc in cycles:
        es = _edge_set_of_cycle(cyc)
        if es & used_edges:
            continue
        used_edges |= es
        selected.append(list(cyc))
        if len(selected) >= max_cycles:
            break
    return selected

def _random_removal_search(adj, n, budget, seed=0):
    """Randomly remove edges, try to find Hamiltonian cycle in remaining graph."""
    rng = random.Random(seed)
    all_edges = [(a, b) for a, b in P['edges']]
    found = []
    seen = set()
    while budget > 0:
        budget -= 1
        # Randomly keep a subset of edges (try to keep ~ enough for a cycle)
        # A Hamiltonian cycle needs n edges; keep maybe n + some
        k = n + rng.randint(0, 10)
        kept = set(rng.sample(all_edges, k))
        # Build subgraph adj
        sub_adj = [set() for _ in range(n)]
        for a, b in kept:
            sub_adj[a].add(b)
            sub_adj[b].add(a)
        # Check each vertex has degree >= 2
        ok = True
        for v in range(n):
            if len(sub_adj[v]) < 2:
                ok = False
                break
        if not ok:
            continue
        # Try to find Hamiltonian cycle in subgraph
        nbrs = [sorted(sub_adj[v]) for v in range(n)]
        start = 0
        if len(sub_adj[start]) < 2:
            continue
        found_local = []
        seen_local = set()
        
        def dfs(path, used):
            if len(found_local) >= 5:
                return
            if len(path) == n:
                last = path[-1]
                first = path[0]
                if first in sub_adj[last]:
                    cyc = list(path)
                    min_idx = cyc.index(min(cyc))
                    cyc = cyc[min_idx:] + cyc[:min_idx]
                    rev = [cyc[0]] + cyc[1:][::-1]
                    canon = tuple(cyc) if cyc < rev else tuple(rev)
                    if canon not in seen_local:
                        seen_local.add(canon)
                        found_local.append(list(canon))
                return
            cur = path[-1]
            cands = [v for v in nbrs[cur] if not used[v]]
            rng.shuffle(cands)
            for v in cands:
                used[v] = True
                path.append(v)
                dfs(path, used)
                path.pop()
                used[v] = False
                if len(found_local) >= 5:
                    return
        
        used = [False] * n
        used[start] = True
        dfs([start], used)
        for cyc in found_local:
            canon = tuple(cyc)
            if canon not in seen:
                seen.add(canon)
                found.append(list(cyc))
        if len(found) >= 50:
            break
    return found

def _pack_with_backtracking(candidates, adj, n, max_cycles=3):
    """Backtracking search for max edge-disjoint cycles."""
    if not candidates:
        return []
    # Precompute edge sets
    edge_sets = [_edge_set_of_cycle(c) for c in candidates]
    
    best = []
    def bt(idx, used, current):
        nonlocal best
        if len(current) > len(best):
            best = list(current)
        if len(current) >= max_cycles:
            return
        for i in range(idx, len(candidates)):
            es = edge_sets[i]
            if es & used:
                continue
            new_used = used | es
            current.append(candidates[i])
            bt(i + 1, new_used, current)
            current.pop()
    
    bt(0, set(), [])
    return best

def construct():
    deadline = time.time() + 80  # safety margin under 90s
    n = P['n']
    adj = _build_adj()
    
    # Step 1: Find many Hamiltonian cycles via randomized DFS on full graph
    all_cycles = []
    seen = set()
    
    # Multiple seeds for diversity
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        cycles = _find_ham_cycles_bfs(adj, n, limit=50, seed=seed)
        for c in cycles:
            canon = tuple(c)
            if canon not in seen:
                seen.add(canon)
                all_cycles.append(c)
        if len(all_cycles) >= 100:
            break
    
    # Also try random edge removal to find more diverse cycles
    if time.time() < deadline - 15:
        more = _random_removal_search(adj, n, budget=2000, seed=42)
        for c in more:
            canon = tuple(c)
            if canon not in seen:
                seen.add(canon)
                all_cycles.append(c)
    
    if not all_cycles:
        # Fallback: the known Hamiltonian cycle
        return [list(range(n))]
    
    # Step 2: Try to pack 3 edge-disjoint cycles
    # First try greedy
    best_pack = _try_pack(all_cycles, adj, n, max_cycles=3)
    
    if len(best_pack) >= 3:
        # Verify
        used = set()
        valid = True
        for cyc in best_pack:
            if not _is_ham_cycle(cyc, adj):
                valid = False
                break
            es = _edge_set_of_cycle(cyc)
            if es & used:
                valid = False
                break
            used |= es
        if valid:
            return [list(cyc) for cyc in best_pack]
    
    # Step 3: Backtracking on a subset of candidates
    if time.time() < deadline - 5:
        # Use a manageable subset
        subset = all_cycles[:200]
        pack = _pack_with_backtracking(subset, adj, n, max_cycles=3)
        if len(pack) >= 3:
            used = set()
            valid = True
            for cyc in pack:
                if not _is_ham_cycle(cyc, adj):
                    valid = False
                    break
                es = _edge_set_of_cycle(cyc)
                if es & used:
                    valid = False
                    break
                used |= es
            if valid:
                return [list(cyc) for cyc in pack]
    
    # Step 4: Greedy with different orderings
    for order_seed in range(50):
        if time.time() > deadline - 2:
            break
        rng = random.Random(order_seed)
        shuffled = list(all_cycles)
        rng.shuffle(shuffled)
        pack = _try_pack(shuffled, adj, n, max_cycles=3)
        if len(pack) >= 3:
            used = set()
            valid = True
            for cyc in pack:
                if not _is_ham_cycle(cyc, adj):
                    valid = False
                    break
                es = _edge_set_of_cycle(cyc)
                if es & used:
                    valid = False
                    break
                used |= es
            if valid:
                return [list(cyc) for cyc in pack]
    
    # Step 5: Return best found (at least 1)
    if best_pack:
        # Verify the best pack
        used = set()
        valid_pack = []
        for cyc in best_pack:
            if not _is_ham_cycle(cyc, adj):
                continue
            es = _edge_set_of_cycle(cyc)
            if es & used:
                continue
            used |= es
            valid_pack.append(list(cyc))
        if valid_pack:
            return valid_pack
    
    # Ultimate fallback
    return [list(range(n))]
# EVOLVE-BLOCK-END
