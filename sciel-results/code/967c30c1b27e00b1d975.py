import random
import time

def _find_c6(adj, A, B):
    # A: list of vertices in part A
    # B: list of vertices in part B
    # adj: list of sets
    n_A = len(A)
    # Precompute indices for A
    # Iterate over triples in A
    for i in range(n_A):
        for j in range(i+1, n_A):
            for k in range(j+1, n_A):
                a1, a2, a3 = A[i], A[j], A[k]
                # S12 = N(a1) ∩ N(a2)
                # S23 = N(a2) ∩ N(a3)
                # S31 = N(a3) ∩ N(a1)
                
                # Optimization: if any is empty, skip
                if not adj[a1] or not adj[a2] or not adj[a3]:
                    continue
                    
                S12 = adj[a1].intersection(adj[a2])
                if not S12: continue
                S23 = adj[a2].intersection(adj[a3])
                if not S23: continue
                S31 = adj[a3].intersection(adj[a1])
                if not S31: continue
                
                # Pick one from each
                b1 = next(iter(S12))
                b2 = next(iter(S23))
                b3 = next(iter(S31))
                
                if b1 == b2 or b2 == b3 or b1 == b3:
                    # Try other combinations? For now, just skip or try next
                    # Actually, we need distinct b's.
                    # Let's just pick the first distinct combination if possible
                    found = False
                    for b1 in S12:
                        for b2 in S23:
                            if b2 == b1: continue
                            for b3 in S31:
                                if b3 == b1 or b3 == b2: continue
                                # Cycle: a1-b1-a2-b2-a3-b3-a1
                                # Edges: (a1,b1), (a2,b1), (a2,b2), (a3,b2), (a3,b3), (a1,b3)
                                return [(a1,b1), (a2,b1), (a2,b2), (a3,b2), (a3,b3), (a1,b3)]
                    continue
                else:
                    return [(a1,b1), (a2,b1), (a2,b2), (a3,b2), (a3,b3), (a1,b3)]
    return None

def _check_add_edge(u, v, adj):
    # u in A, v in B
    # Check if path of length 5 from u to v exists
    # u -> b1 -> a2 -> b2 -> a3 -> v
    # b1 in N(u)
    # a2 in N(b1) \ {u}
    # b2 in N(a2) \ {b1}
    # a3 in N(b2) \ {a2}
    # v in N(a3)
    
    for b1 in adj[u]:
        if b1 == v: continue # Edge exists? No, we are checking if adding creates C6. u-v not in adj yet.
        # Actually, if b1 == v, that's length 1. We need length 5.
        # So b1 != v.
        
        for a2 in adj[b1]:
            if a2 == u: continue
            for b2 in adj[a2]:
                if b2 == b1: continue
                for a3 in adj[b2]:
                    if a3 == a2: continue
                    if v in adj[a3]:
                        return True
    return False

def construct():
    n = 40
    A = list(range(20))
    B = list(range(20, 40))
    
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 115
    
    for trial in range(20):
        if time.time() - start_time > time_limit: break
        
        # Initialize random bipartite graph
        p = 0.3 + random.random() * 0.2 # 0.3 to 0.5
        adj = [set() for _ in range(n)]
        edges = []
        for u in A:
            for v in B:
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
        
        current_count = len(edges)
        
        # Phase 1: Remove C6s
        while time.time() - start_time < time_limit:
            c6 = _find_c6(adj, A, B)
            if not c6:
                break
            # Remove a random edge from the cycle
            edge_to_remove = random.choice(c6)
            u, v = edge_to_remove
            adj[u].remove(v)
            adj[v].remove(u)
            edges.remove([u, v]) if [u, v] in edges else None
            # Note: edges list management is O(N). 
            # Better to reconstruct or use a set.
            # Let's use a set of tuples for O(1) removal.
            
        # Rebuild edges list from adj for clarity
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        current_count = len(edges)
        
        # Phase 2: Add edges
        # Candidate edges: all missing bipartite edges
        candidates = []
        existing = set()
        for u in A:
            for v in B:
                if v not in adj[u]:
                    candidates.append((u, v))
                else:
                    existing.add((u, v))
        
        random.shuffle(candidates)
        
        added = 0
        for u, v in candidates:
            if time.time() - start_time > time_limit: break
            if not _check_add_edge(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                current_count += 1
                added += 1
                
        if current_count > best_count:
            best_count = current_count
            best_edges = edges[:]
            
    return best_edges
