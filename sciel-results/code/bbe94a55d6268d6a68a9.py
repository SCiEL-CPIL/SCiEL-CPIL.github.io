"""Conflict-graph max-clique with bitmask validity."""
import time
import random
import itertools
import sys

N = 15
DEADLINE = 55.0  # Safety margin under the 90s limit
sys.setrecursionlimit(100000)

def _adjacency(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _degrees(adj):
    return [bin(adj[i]).count('1') for i in range(N)]

def _is_valid(adj, deg):
    # For every edge (u, v) with u < v, check all simple 3-edge paths s-u-v-t
    for u in range(N):
        au = adj[u]
        for v in range(u + 1, N):
            if not (au >> v) & 1:
                continue
            av = adj[v]
            # s is neighbor of u, s != v
            for s in range(N):
                if s == v or s == u:
                    continue
                if not (au >> s) & 1:
                    continue
                ds = deg[s]
                # t is neighbor of v, t != u, t != s
                for t in range(N):
                    if t == u or t == s:
                        continue
                    if not (av >> t) & 1:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _build_conflict():
    """For every edge e=(u,v), compute the set of edges f that cannot coexist with e.
    e and f conflict iff there exists a simple 3-edge path s-a-b-t with a-b == e (or b-a == e)
    such that deg[s] == deg[t] in the FULL graph.
    Since degrees depend on the full edge set, we precompute for each edge e and each
    candidate 3-path through e, the condition on degrees.
    
    Actually the constraint is: in the final graph, for every 3-edge path s-u-v-t,
    deg[s] != deg[t]. So two edges e and f conflict if there's a 3-path using both
    where the endpoints would have equal degree. But degree equality depends on the
    full graph, so this isn't a fixed pairwise conflict.
    
    However, we can still use a conflict graph approach: precompute for each pair of
    edges whether they COULD form a 3-path (i.e., share a vertex in the right way),
    and then during search, check if the degrees would be equal.
    
    Alternative: just use the greedy + local search but with a much faster validity
    check and better heuristics.
    """
    # Precompute for each edge (u,v) the list of (s, t) pairs that form a 3-path s-u-v-t
    # where s is neighbor of u (s != v) and t is neighbor of v (t != u, t != s)
    # In the final graph, deg[s] must != deg[t]
    # Two edges e1=(u,v) and e2=(u',v') conflict if they form a 3-path with equal endpoints
    # But since degrees are global, we can't precompute this statically.
    # 
    # Instead: use the approach where we build the conflict graph dynamically.
    # For maximum clique: vertices are edges, two edges are connected if they CAN coexist.
    # Two edges CAN coexist if no 3-path through them has endpoints of equal degree.
    # But degree equality depends on the whole graph...
    #
    # Key insight: if we fix the degree sequence, then we CAN precompute conflicts.
    # So: try candidate degree sequences, build conflict graph for that sequence,
    # find max clique, verify.
    
    pass

def _conflict_for_degs(degs):
    """Given a fixed degree sequence, build the conflict graph.
    Two edges e=(u,v) and f=(a,b) conflict if there's a 3-edge simple path
    s-x-y-t where {x,y} is one of the edges and {s,t} are the other,
    such that deg[s] == deg[t].
    
    Actually more precisely: a 3-edge path is s-u-v-t where edges are (s,u),(u,v),(v,t).
    The constraint is deg[s] != deg[t].
    
    For two edges e and f to be in conflict, they must be part of some 3-path where
    the endpoints have equal degree.
    
    Let me think differently. In the final graph with degree sequence degs, 
    edge e=(u,v) is "bad" with respect to edge f if there exists a 3-path using
    both e and f where endpoints have equal degree.
    
    A 3-path s-u-v-t uses edges (s,u), (u,v), (v,t). So it uses 3 edges.
    Two edges e and f are in a 3-path together if:
    - They share a vertex: e=(u,v), f=(u,w) or f=(v,w), forming path w-u-v-? or ?-v-u-w
    - Wait, a 3-path has 3 edges. Two specific edges are in a 3-path if they are
      connected (share a vertex) and the path extends by one more edge at each end.
    
    Let me enumerate: for edge e=(u,v), a 3-path through e is s-u-v-t or s-v-u-t.
    The other two edges are (s,u) and (v,t) [or (s,v) and (u,t)].
    
    So edge e=(u,v) and edge f=(s,u) are in a 3-path together if there exists t
    adjacent to v (t != u, t != s) such that (v,t) is an edge. Then the 3-path is
    s-u-v-t and the constraint is deg[s] != deg[t].
    
    This is getting complex. Let me just use a different approach:
    For a given degree sequence, check if there exists a graph with that degree
    sequence that satisfies the constraint. Use backtracking with the degree
    sequence as a guide.
    """
    pass

def _greedy_add(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        trial = edges + [list(key)]
        if _valid(trial):
            edges = trial
            present.add(key)
    return edges

def _erdos_gallai(degs):
    """Check if degree sequence is graphical using Erdős–Gallai theorem."""
    n = len(degs)
    if sum(degs) % 2 != 0:
        return False
    if any(d < 0 or d > n - 1 for d in degs):
        return False
    s = sorted(degs, reverse=True)
    total = sum(s)
    for k in range(1, n + 1):
        left = sum(s[:k])
        right = k * (k - 1) + sum(min(d, k) for d in s[k:])
        if left > right:
            return False
    return True

def _try_degree_partition(target_degs, rng, t0):
    """Try to construct a graph with the given degree sequence that is valid."""
    # Generate candidate edges ordered by how they help satisfy degrees
    # Use a randomized greedy approach
    best = []
    best_len = 0
    
    for attempt in range(50):
        if time.time() - t0 > DEADLINE:
            break
        
        # Shuffle edge order
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(all_edges)
        
        # Greedy: add edge if it keeps the partial graph "degree-compatible"
        # We track current degrees and try to approach target_degs
        cur_degs = [0] * N
        edges = []
        adj = [0] * N
        
        for u, v in all_edges:
            if time.time() - t0 > DEADLINE:
                break
            # Check if adding this edge keeps us on track
            # Simple heuristic: prefer edges between vertices with lower current degree
            # relative to target
            if cur_degs[u] >= target_degs[u] or cur_degs[v] >= target_degs[v]:
                continue
            
            # Try adding
            trial_adj = adj[:]
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            trial_degs = cur_degs[:]
            trial_degs[u] += 1
            trial_degs[v] += 1
            
            if _is_valid(trial_adj, trial_degs):
                edges.append([u, v])
                adj = trial_adj
                cur_degs = trial_degs
        
        if len(edges) > best_len:
            best_len = len(edges)
            best = edges
    
    return best, best_len

def _fast_is_valid(adj, deg):
    """Faster validity check using precomputed neighbor lists."""
    # Precompute neighbor lists
    nbrs = [None] * N
    for i in range(N):
        nbrs[i] = [j for j in range(N) if (adj[i] >> j) & 1]
    
    for u in range(N):
        for v in nbrs[u]:
            if v <= u:
                continue
            # Check all s in nbrs[u] (s != v) and t in nbrs[v] (t != u, t != s)
            for s in nbrs[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in nbrs[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _max_clique_branch_and_bound(adj_mask, n_vertices, t0, deadline, current_best):
    """Branch and bound for maximum clique using bitmasks.
    adj_mask[i] is a bitmask of neighbors of vertex i.
    n_vertices is the number of vertices.
    Returns the size of the maximum clique found.
    """
    # Use a simple branch and bound with coloring upper bound
    # We'll use a greedy coloring to get upper bounds
    
    def color_bound(candidates):
        """Greedy coloring of the subgraph induced by candidates.
        Returns number of colors used (upper bound on clique size).
        """
        if not candidates:
            return 0
        # Greedy sequential coloring
        remaining = list(candidates)
        colors = 0
        while remaining:
            colors += 1
            # Pick a maximal independent set from remaining
            independent = []
            new_remaining = []
            for v in remaining:
                if not any((adj_mask[v] >> w) & 1 for w in independent):
                    independent.append(v)
                else:
                    new_remaining.append(v)
            remaining = new_remaining
        return colors
    
    best_clique = [0]
    
    def bb(candidates, current):
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > best_clique[0]:
                best_clique[0] = len(current)
            return
        
        # Upper bound: current size + color bound on candidates
        ub = len(current) + color_bound(candidates)
        if ub <= best_clique[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the candidate subgraph
        cand_list = list(candidates)
        # Sort by degree in candidate subgraph (descending)
        cand_list.sort(key=lambda v: bin(adj_mask[v] & ((1 << n_vertices) - 1)).count('1'), reverse=True)
        
        for v in cand_list:
            if time.time() > deadline:
                return
            # Include v
            new_current = current + [v]
            # New candidates: intersection of candidates with neighbors of v
            new_cands = [w for w in candidates if w != v and (adj_mask[v] >> w) & 1]
            bb(new_cands, new_current)
            
            # Exclude v
            candidates = [w for w in candidates if w != v]
    
    bb(list(range(n_vertices)), [])
    return best_clique[0]

def _try_clique_for_degs(degs, t0, deadline):
    """For a given degree sequence, build conflict graph and find max clique.
    Two edges conflict if they form a 3-path with endpoints of equal degree.
    """
    # Precompute: for each edge e=(u,v), which edges f can form a 3-path with e
    # such that the endpoints would have equal degree (given degs)
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    n_e = len(all_edges)
    edge_idx = {}
    for i, e in enumerate(all_edges):
        edge_idx[e] = i
    
    # conflict[i] = bitmask of edges that conflict with edge i
    conflict = [0] * n_e
    
    for i, (u, v) in enumerate(all_edges):
        # 3-paths through edge (u,v): s-u-v-t where s~u, t~v, s!=v, t!=u, t!=s
        # Also s-v-u-t where s~v, t~u, s!=u, t!=v, t!=s
        # For each such 3-path, the other two edges are (s,u) and (v,t) [or (s,v) and (u,t)]
        # Edge (u,v) conflicts with edge (s,u) if deg[s] == deg[t] for some valid t
        # Edge (u,v) conflicts with edge (v,t) if deg[s] == deg[t] for some valid s
        
        # Collect all 3-paths through (u,v)
        # Path type 1: s-u-v-t, edges are (s,u), (u,v), (v,t)
        #   s must be neighbor of u, s != v
        #   t must be neighbor of v, t != u, t != s
        for s in range(N):
            if s == v or s == u:
                continue
            for t in range(N):
                if t == u or t == s or t == v:
                    continue
                if deg[s] == deg[t]:
                    # Edge (u,v) conflicts with edge (s,u) and edge (v,t)
                    if (s, u) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(s, u)])
                    elif (u, s) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(u, s)])
                    if (v, t) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(v, t)])
                    elif (t, v) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(t, v)])
        
        # Path type 2: s-v-u-t, edges are (s,v), (v,u), (u,t)
        #   s must be neighbor of v, s != u
        #   t must be neighbor of u, t != v, t != s
        for s in range(N):
            if s == u or s == v:
                continue
            for t in range(N):
                if t == v or t == s or t == u:
                    continue
                if deg[s] == deg[t]:
                    # Edge (u,v) conflicts with edge (s,v) and edge (u,t)
                    if (s, v) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(s, v)])
                    elif (v, s) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(v, s)])
                    if (u, t) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(u, t)])
                    elif (t, u) in edge_idx:
                        conflict[i] |= (1 << edge_idx[(t, u)])
    
    # Build adjacency for the conflict graph's complement (compatibility graph)
    # Two edges are compatible if they don't conflict
    compat = [0] * n_e
    for i in range(n_e):
        for j in range(i + 1, n_e):
            if not (conflict[i] >> j) & 1 and not (conflict[j] >> i) & 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Now find maximum clique in the compatibility graph
    # But we also need the clique to be a valid graph (no two edges share both endpoints, etc.)
    # Actually, a clique in the compatibility graph is a set of edges where no two conflict.
    # But we also need the resulting graph to have the correct degree sequence!
    # This is the catch - the conflict graph approach with fixed degrees doesn't ensure
    # the degree sequence is met.
    
    # So this approach is flawed. Let me just use improved greedy + local search.
    return None

def construct():
    t0 = time.time()
    rng = random.Random(20240607)
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Strong baseline: K_{7,8} has 56 edges and is valid
    k78 = [[a, b] for a in range(7) for b in range(7, 15)]
    best = k78
    best_len = len(k78)
    
    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand
    
    # Phase 1: Greedy from K_{7,8} with multiple random orders
    for _ in range(30):
        if time.time() - t0 > DEADLINE:
            break
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy_add(order, k78)
        consider(cand)
    
    # Phase 2: Try different degree sequences
    candidate_degs = []
    for _ in range(300):
        if time.time() - t0 > DEADLINE:
            break
        degs = [8] * 7 + [7] * 8
        for _ in range(5):
            i = rng.randint(0, N - 1)
            if degs[i] < N - 1:
                degs[i] += 1
        for _ in range(3):
            i = rng.randint(0, N - 1)
            if degs[i] > 1:
                degs[i] -= 1
        if _erdos_gallai(degs):
            candidate_degs.append(degs)
    
    for degs in candidate_degs[:80]:
        if time.time() - t0 > DEADLINE:
            break
        if sum(degs) // 2 <= best_len:
            continue
        cand, cl = _try_degree_partition(degs, rng, t0)
        consider(cand)
    
    # Phase 3: Local search
    for _ in range(50):
        if time.time() - t0 > DEADLINE:
            break
        base = best[:]
        if not base:
            continue
        idx = rng.randint(0, len(base) - 1)
        reduced = base[:idx] + base[idx + 1:]
        
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy_add(order, reduced)
        consider(cand)
        
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy_add(order2, cand)
            consider(cand2)
    
    # Phase 4: Random restarts with repair
    for _ in range(40):
        if time.time() - t0 > DEADLINE:
            break
        
        adj = [0] * N
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.85:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        deg = _degrees(adj)
        for _ in range(150):
            if time.time() - t0 > DEADLINE:
                break
            if _fast_is_valid(adj, deg):
                break
            found = False
            for u in range(N):
                if found:
                    break
                au = adj[u]
                for v in range(u + 1, N):
                    if not (au >> v) & 1:
                        continue
                    av = adj[v]
                    for s in range(N):
                        if s == u or s == v or not (au >> s) & 1:
                            continue
                        for t in range(N):
                            if t == u or t == s or not (av >> t) & 1:
                                continue
                            if deg[s] == deg[t]:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                deg[u] -= 1
                                deg[v] -= 1
                                edges = [[u2, v2] for u2 in range(N) for v2 in range(u2 + 1, N) if (adj[u2] >> v2) & 1]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        consider(edges)
    
    # Phase 5: Intensive local improvement on best
    if time.time() - t0 < DEADLINE - 5:
        adj = _adjacency(best)
        deg = _degrees(adj)
        for _ in range(200):
            if time.time() - t0 > DEADLINE - 2:
                break
            if not _fast_is_valid(adj, deg):
                break
            # Try to add a random missing edge
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not (adj[u] >> v) & 1]
            if not missing:
                break
            u, v = rng.choice(missing)
            trial_adj = adj[:]
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            trial_deg = deg[:]
            trial_deg[u] += 1
            trial_deg[v] += 1
            if _fast_is_valid(trial_adj, trial_deg):
                adj = trial_adj
                deg = trial_deg
                best = [[u2, v2] for u2 in range(N) for v2 in range(u2 + 1, N) if (adj[u2] >> v2) & 1]
                best_len = len(best)
    
    result = [[int(u), int(v)] for u, v in best]
    return result
