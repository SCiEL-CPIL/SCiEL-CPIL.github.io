import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = set(get_graph(n, p, seed))
        if not edges:
            return []
        
        # Build adjacency list for fast lookups
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        while edges:
            # Check time
            if time.time() > deadline:
                break
            
            # Find the edge (u,v) with the most common neighbors
            # This gives us the largest potential clique seed
            best_common = -1
            best_u = -1
            best_v = -1
            for u, v in edges:
                common = len(adj[u] & adj[v])
                if common > best_common:
                    best_common = common
                    best_u = u
                    best_v = v
            
            if best_u == -1:
                break
            
            # Build maximal clique starting from edge (best_u, best_v)
            C = [best_u, best_v]
            # Candidates: common neighbors of all vertices in C
            P = adj[best_u] & adj[best_v]
            P -= {best_u, best_v}
            
            # Greedy expansion: repeatedly add the vertex with most connections to C
            while P:
                if time.time() > deadline:
                    break
                # Find candidate with most neighbors in C (and among remaining P)
                best_cand = None
                best_score = -1
                for cand in P:
                    score = len(adj[cand] & set(C))
                    # Tie-break by degree
                    if score > best_score or (score == best_score and len(adj[cand]) > len(adj[best_cand] if best_cand is not None else set())):
                        best_score = score
                        best_cand = cand
                if best_cand is None:
                    break
                C.append(best_cand)
                P = P & adj[best_cand]
            
            # Add clique to result
            cliques.append(C)
            
            # Remove edges in C
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    u, v2 = C[i], C[j]
                    if u > v2:
                        u, v2 = v2, u
                    if (u, v2) in edges:
                        edges.remove((u, v2))
                        adj[u].remove(v2)
                        adj[v2].remove(u)
            
            # Check time
            if time.time() > deadline:
                break
                
        return cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))
    
    if time.time() > deadline:
        # If we're too late, still try to complete instance 2 quickly
        # or return what we have
        results.append(solve_instance(64, 0.55, 602))
        return results
    
    # Instance 2
    results.append(solve_instance(64, 0.55, 602))
    
    return results
