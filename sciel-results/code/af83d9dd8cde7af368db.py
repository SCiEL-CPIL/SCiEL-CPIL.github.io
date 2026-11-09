"""K3,3-free graph on 20 vertices: maximize edges via randomized greedy + local search."""
import time
import random
import itertools

N = 20
# All pairs of vertices
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
# Map edge to index for fast lookup
EDGE_SET = set(ALL_PAIRS)

def _can_add(edge, adj, common3):
    """Check if adding edge (u,v) keeps the graph K3,3-free.
    
    A K3,3 is formed if there exist vertices a,b,c on one side and d,e,f on the other
    such that all 9 cross-edges exist. Adding edge (u,v) creates a K3,3 iff:
    - u and v share >= 2 common neighbors w1, w2, and one of {u,v} has >= 1 neighbor 
      that together with w1,w2 forms a K3,3.
    
    Actually, the standard check: adding edge (u,v) creates a K3,3 iff there exist
    3 vertices S and 3 vertices T such that all edges between S and T exist, and
    (u,v) is one of those edges. Without loss of generality, u is in S and v is in T.
    So we need: there exist a,b in S\{u} and c,d,e in T\{v} such that:
      - a,b are neighbors of u (already in graph)
      - c,d,e are neighbors of v (already in graph)  
      - a,b are neighbors of c,d,e (all 6 edges already in graph)
    
    Equivalently: let N_u = neighbors of u, N_v = neighbors of v.
    We need |{w in N_u ∩ N_v}| >= 2 (call these shared neighbors s1, s2)
    AND there exist 3 common neighbors of {s1, s2} that are all in N_v... 
    
    Wait, let me reconsider. The K3,3 has bipartition (S,T) with |S|=|T|=3.
    If we add edge (u,v), then either u in S, v in T or vice versa.
    
    Case: u in S, v in T.
    S = {u, a, b}, T = {v, c, d, e}... no, |T|=3 so T={v, c, d}.
    Edges needed: u-v (new), u-c, u-d, a-v, a-c, a-d, b-v, b-c, b-d.
    So a,b must be neighbors of u (in current graph).
    c,d must be neighbors of v (in current graph).
    a,b must be neighbors of c,d (all 4 edges in current graph).
    
    So: exists a,b in N(u) and c,d in N(v) such that a,b are both adjacent to c and d.
    i.e., exists c,d in N(v) such that |N(c) ∩ N(d) ∩ N(u)| >= 2.
    OR equivalently: exists a,b in N(u) such that |N(a) ∩ N(b) ∩ N(v)| >= 2.
    
    By symmetry, we can check: for each pair (c,d) in N(v), check if they have >=2 common neighbors in N(u).
    This is O(|N(v)|^2 * |N(u)|) which is fine for small graphs.
    """
    u, v = edge
    Nu = adj[u]
    Nv = adj[v]
    
    # Check: exists c,d in Nv such that |N(c) ∩ N(d) ∩ Nu| >= 2
    for c in Nv:
        for d in Nv:
            if c == d:
                continue
            # Count common neighbors of c and d that are in Nu
            cnt = 0
            for x in Nu:
                if x == c or x == d:
                    continue
                if x in adj[c] and x in adj[d]:
                    cnt += 1
                    if cnt >= 2:
                        return False
    return True

def _can_add_fast(edge, adj):
    """Faster check using bitmasks."""
    u, v = edge
    Nu = adj[u]
    Nv = adj[v]
    for c in Nv:
        for d in Nv:
            if c == d:
                continue
            # Check if c and d share at least 2 neighbors in Nu
            common = Nu & adj[c] & adj[d]
            if len(common) >= 2:
                return False
    return True

def _is_k33_free(adj):
    """Verify the entire graph is K3,3-free."""
    for u in range(N):
        for v in range(u+1, N):
            if v in adj[u]:
                # Check if adding this edge would have been valid given the rest
                # Simpler: just check all potential K3,3 subgraphs
                pass
    # Direct check: for every pair of disjoint 3-sets (S,T), check if all 9 edges exist
    for S in itertools.combinations(range(N), 3):
        for T in itertools.combinations(range(N), 3):
            if set(S) & set(T):
                continue
            ok = True
            for s in S:
                for t in T:
                    if t not in adj[s]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return False
    return True

def construct():
    deadline = time.time() + 140  # Safety margin
    best_edges = 0
    best_adj = None
    
    # Strategy: Randomized greedy construction + local improvement
    # Multiple restarts with different random orderings
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        random.seed(trial * 7919 + 1)
        
        # Shuffle edge order
        edges = ALL_PAIRS[:]
        random.shuffle(edges)
        
        adj = [set() for _ in range(N)]
        edge_count = 0
        added = set()
        
        # Greedy: add edges in random order if valid
        for e in edges:
            if time.time() > deadline:
                break
            u, v = e
            if v in adj[u]:
                continue
            if _can_add_fast(e, adj):
                adj[u].add(v)
                adj[v].add(u)
                added.add(e)
                edge_count += 1
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [set(s) for s in adj]
        
        # Local search: try 1-for-1 swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each missing edge
            for e in ALL_PAIRS:
                if time.time() > deadline:
                    break
                if e in added:
                    continue
                u, v = e
                if v in adj[u]:
                    continue
                if _can_add_fast(e, adj):
                    adj[u].add(v)
                    adj[v].add(u)
                    added.add(e)
                    edge_count += 1
                    improved = True
                    break  # Restart scan after change
            
            if not improved:
                # Try 1-for-1 swaps: remove edge e1, add edge e2
                for e1 in list(added):
                    if time.time() > deadline:
                        break
                    u1, v1 = e1
                    # Temporarily remove
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    added.discard(e1)
                    edge_count -= 1
                    
                    # Try to add a different edge
                    for e2 in ALL_PAIRS:
                        if e2 == e1 or e2 in added:
                            continue
                        u2, v2 = e2
                        if v2 in adj[u2]:
                            continue
                        if _can_add_fast(e2, adj):
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            added.add(e2)
                            edge_count += 1
                            improved = True
                            break
                        else:
                            # e2 not valid, but we already removed e1
                            # Keep e1 removed? No, revert
                            pass
                    
                    if not improved:
                        # Revert e1
                        adj[u1].add(v1)
                        adj[v1].add(u1)
                        added.add(e1)
                        edge_count += 1
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [set(s) for s in adj]
    
    # Build result from best_adj
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    result = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
