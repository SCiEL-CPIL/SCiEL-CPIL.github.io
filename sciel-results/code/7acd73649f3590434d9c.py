import time
import random

_START_TIME = time.time()
_TIME_LIMIT = 110.0

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    results = []
    
    for n, p, seed in [(32, 0.6, 601), (64, 0.55, 602)]:
        edges = instance(n, p, seed)
        
        # Check time before starting heavy work
        if time.time() - _START_TIME > _TIME_LIMIT:
            # Emergency exit: return trivial partitions for all remaining instances
            # This is a fallback for safety
            for _ in range(2 - len(results)):
                if _ == 0:
                    # We are at instance 0 or 1? 
                    # Actually, if we are here, we haven't added anything yet or failed.
                    # Let's just return a valid trivial partition for the current and subsequent.
                    pass
            # Simpler: just break and handle outside? No, must return list of 2.
            # If we timeout, we just return the trivial partition for the current instance 
            # and any unprocessed ones.
            # But for the sake of the contract, let's assume we have time.
            # The graphs are small.
            pass

        remaining_edges = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        while remaining_edges:
            if time.time() - _START_TIME > _TIME_LIMIT:
                # Fallback: add remaining edges as individual cliques (size 2)
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Find vertex with max degree in remaining graph
            max_deg = -1
            v = -1
            for u in range(n):
                deg = len(adj[u])
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if v == -1 or max_deg == 0:
                # This should not happen if remaining_edges is not empty
                # But if it does, pick any edge
                e = next(iter(remaining_edges))
                v = e[0]
                # Force adjacency update if needed? No, adj is consistent.
                # If max_deg is 0, adj[v] is empty. 
                # This implies v is isolated. 
                # But remaining_edges has an edge. 
                # So some other vertex has degree > 0.
                # We should have picked that vertex.
                # Unless there's a bug in degree counting.
                # adj[u] is updated correctly.
                pass

            # Greedy expansion
            clique = [v]
            candidates = set(adj[v])
            
            while candidates:
                # Pick candidate with highest degree
                best_candidate = -1
                best_deg = -1
                for c in candidates:
                    deg = len(adj[c])
                    if deg > best_deg:
                        best_deg = deg
                        best_candidate = c
                
                if best_candidate == -1:
                    break
                    
                u = best_candidate
                candidates.remove(u)
                
                # Check validity
                valid = True
                for c in clique:
                    if u not in adj[c]:
                        valid = False
                        break
                
                if valid:
                    clique.append(u)
                    candidates.intersection_update(adj[u])
            
            cliques.append(clique)
            
            # Remove edges
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v_c = clique[j]
                    edge = (u, v_c) if u < v_c else (v_c, u)
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                    adj[u].discard(v_c)
                    adj[v_c].discard(u)
                    
        results.append(cliques)
        
    return results
