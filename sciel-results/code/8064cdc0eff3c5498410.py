import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 15
    A = list(range(7))
    B = list(range(7, 15))
    
    def compute_deg_adj(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj
    
    def is_valid(edge_set):
        deg, adj = compute_deg_adj(edge_set)
        
        # Precompute for each vertex: neighbors grouped by degree
        # For vertex v, deg_neighbors[v][d] = set of neighbors of v with degree d
        deg_neighbors = [defaultdict(set) for _ in range(n)]
        for v in range(n):
            for nb in adj[v]:
                deg_neighbors[v][deg[nb]].add(nb)
        
        # For each edge (u,v), check if there exist s in N(u)\{v}, t in N(v)\{u}
        # with s != t and deg[s] == deg[t]
        for (u, v) in edge_set:
            # Get neighbors of u (excluding v) grouped by degree
            nu_deg = {}
            for s in adj[u]:
                if s != v:
                    d = deg[s]
                    if d not in nu_deg:
                        nu_deg[d] = []
                    nu_deg[d].append(s)
            
            # Get neighbors of v (excluding u) grouped by degree
            nv_deg = {}
            for t in adj[v]:
                if t != u:
                    d = deg[t]
                    if d not in nv_deg:
                        nv_deg[d] = []
                    nv_deg[d].append(t)
            
            # Check for common degrees
            common_degs = set(nu_deg.keys()) & set(nv_deg.keys())
            for d in common_degs:
                ns = nu_deg[d]
                nt = nv_deg[d]
                # Need s != t
                if len(ns) >= 2 and len(nt) >= 2:
                    return False
                if len(ns) == 1 and len(nt) == 1:
                    if ns[0] != nt[0]:
                        return False
                if len(ns) >= 1 and len(nt) >= 1:
                    # More careful check
                    s_set = set(ns)
                    t_set = set(nt)
                    if len(s_set & t_set) < min(len(s_set), len(t_set)):
                        # There exists s in s_set and t in t_set with s != t
                        if len(s_set) + len(t_set) - len(s_set & t_set) > 1 or len(s_set & t_set) < min(len(s_set), len(t_set)):
                            # Simpler: if there's any s,t with s!=t
                            if len(s_set) > 0 and len(t_set) > 0:
                                if len(s_set & t_set) < min(len(s_set), len(t_set)):
                                    return False
                                elif len(s_set) == 1 and len(t_set) == 1:
                                    # Already handled above
                                    pass
        
        return True
    
    def is_valid_fast(edge_set, deg, adj):
        """Check validity given precomputed deg and adj."""
        for (u, v) in edge_set:
            # Quick check: for each degree d that appears in both N(u)\{v} and N(v)\{u}
            # we need all such neighbors to be the same vertex
            deg_u = {}
            for s in adj[u]:
                if s != v:
                    d = deg[s]
                    if d not in deg_u:
                        deg_u[d] = set()
                    deg_u[d].add(s)
            
            deg_v = {}
            for t in adj[v]:
                if t != u:
                    d = deg[t]
                    if d not in deg_v:
                        deg_v[d] = set()
                    deg_v[d].add(t)
            
            for d in set(deg_u.keys()) & set(deg_v.keys()):
                su = deg_u[d]
                tv = deg_v[d]
                # If there exist s in su, t in tv with s != t, invalid
                if len(su) + len(tv) > len(su & tv) + 1:
                    return False
                # More precisely: if su and tv are not identical singletons
                if len(su) == 1 and len(tv) == 1:
                    if su != tv:
                        return False
                elif len(su) > 1 or len(tv) > 1:
                    # If either has >1 element, we can pick different ones unless they're forced
                    # Actually if |su|>=1 and |tv|>=1 and su != tv as sets, or one has size >1
                    if su != tv:
                        return False
                    if len(su) > 1:
                        return False
        
        return True
    
    # Start with K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    # Generate all candidate intra-partition edges
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            candidates.append((B[i], B[j]))
    
    best_set = set(base_edges)
    best_count = len(base_edges)
    
    # Score an edge by how "safe" it is: prefer edges where endpoints have few same-degree neighbors
    def edge_score(e, deg, adj):
        u, v = e
        score = 0
        # Count potential conflicts
        for s in adj[u]:
            if s != v:
                for t in adj[v]:
                    if t != u and t != s and deg[s] == deg[t]:
                        score += 1
        return score
    
    # Multiple random restarts with greedy + local search
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        current_set = set(base_edges)
        deg, adj = compute_deg_adj(current_set)
        
        # Greedy phase: add edges in order of increasing conflict score
        remaining = list(candidates)
        improved = True
        max_greedy_rounds = 5
        for round_num in range(max_greedy_rounds):
            if time.time() > deadline - 3.0:
                break
            # Score all remaining edges
            scored = []
            for e in remaining:
                if e in current_set:
                    continue
                s = edge_score(e, deg, adj)
                scored.append((s, e))
            
            if not scored:
                break
            
            # Sort by score (fewer conflicts first)
            scored.sort(key=lambda x: x[0])
            
            added_this_round = False
            for s, e in scored:
                if time.time() > deadline - 2.0:
                    break
                test_set = set(current_set)
                test_set.add(e)
                if is_valid_fast(test_set, deg, adj):
                    current_set = test_set
                    deg, adj = compute_deg_adj(current_set)
                    remaining.remove(e)
                    added_this_round = True
                    break
            
            if not added_this_round:
                break
        
        # Local search: try adding any remaining edge
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            deg, adj = compute_deg_adj(current_set)
            for e in list(remaining):
                if time.time() > deadline - 1.0:
                    break
                test_set = set(current_set)
                test_set.add(e)
                if is_valid_fast(test_set, deg, adj):
                    current_set = test_set
                    remaining.remove(e)
                    improved = True
                    break
        
        if len(current_set) > best_count:
            best_set = current_set
            best_count = len(current_set)
        
        # Early termination if we've found a good solution
        if best_count >= 65:
            break
    
    # Convert to required format
    result = []
    for u, v in best_set:
        result.append([int(u), int(v)])
    
    return result
