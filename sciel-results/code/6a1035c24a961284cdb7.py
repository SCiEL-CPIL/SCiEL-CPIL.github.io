import time
import random

def construct():
    start_time = time.time()
    
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
            if time.time() - start_time > 115:
                break
                
            # Find the vertex with the highest degree in remaining graph
            v = max(range(n), key=lambda x: len(adj[x]))
            
            if not adj[v]:
                break
                
            # Use a better heuristic: find a large clique using the common neighbors
            # Start with v, then greedily add vertices that connect to all current clique members
            # Sort candidates by how many connections they have to current clique members
            C = [v]
            # Candidates are neighbors of v
            candidates = list(adj[v])
            
            # Sort by degree in remaining graph (higher degree first)
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            # Greedily build clique
            changed = True
            while changed:
                changed = False
                new_candidates = []
                for u in candidates:
                    # Check if u connects to all vertices in C
                    if all(u in adj[c] for c in C):
                        C.append(u)
                        changed = True
                        # Update candidates: keep only those that connect to u
                        candidates = [x for x in candidates if x != u and x in adj[u]]
                        # Re-sort by degree
                        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
                        break
                if not changed:
                    # No more vertices can be added
                    pass
            
            # Add clique to result
            cliques.append(C)
            
            # Remove all edges within C from the graph
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    u, v2 = C[i], C[j]
                    if u > v2:
                        u, v2 = v2, u
                    if (u, v2) in edges:
                        edges.remove((u, v2))
                        adj[u].remove(v2)
                        adj[v2].remove(u)
            
        return cliques

    results = []
    # Instance 1
    results.append(solve_instance(32, 0.6, 601))
    
    # Instance 2
    results.append(solve_instance(64, 0.55, 602))
    
    return results
