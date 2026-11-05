import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    def get_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, edges):
        if not edges:
            return []
        
        rem_edges = set(edges)
        cliques = []
        
        # Helper to check if edge is in rem
        # We can use a local variable for speed if needed, but set lookup is fast
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: just add remaining edges as size-2 cliques
                # This ensures validity
                for e in rem_edges:
                    cliques.append(list(e))
                break
            
            # Pick an edge
            u, v = next(iter(rem_edges))
            
            # Expand to maximal clique
            C = {u, v}
            
            # Find candidates S
            # w must be connected to all in C
            # Since C starts with u,v, S is common neighbors of u and v in rem_edges
            
            # Optimization: Precompute neighbors? No, rem_edges changes.
            # Just filter range(n)
            
            # To speed up, we can maintain a list of all vertices
            # S = {w for w in range(n) if w not in C and (min(u,w), max(u,w)) in rem_edges and (min(v,w), max(v,w)) in rem_edges}
            
            # Let's try to be smarter about candidate selection to get larger cliques
            # A simple heuristic: pick the candidate w that has the most neighbors in the current S?
            # That requires computing degrees in S, which is O(|S|^2).
            # Given small N, this might be okay.
            # But let's start with arbitrary selection for speed, as density is high.
            
            S = set()
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check if w is connected to u and v in rem_edges
                eu = (u, w) if u < w else (w, u)
                ev = (v, w) if v < w else (w, v)
                if eu in rem_edges and ev in rem_edges:
                    S.add(w)
            
            while S:
                # Pick a vertex from S
                # Heuristic: pick the one with highest degree in S to keep clique dense?
                # Let's just pick arbitrary for now to ensure speed.
                w = next(iter(S))
                C.add(w)
                
                # Update S
                S.discard(w)
                new_S = set()
                for x in S:
                    ew = (w, x) if w < x else (x, w)
                    if ew in rem_edges:
                        new_S.add(x)
                S = new_S
            
            # Remove all edges in C from rem_edges
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    a = list(C)[i]
                    b = list(C)[j]
                    e = (a, b) if a < b else (b, a)
                    if e in rem_edges:
                        rem_edges.discard(e)
            
            cliques.append(sorted(C))
            
        return cliques

    # Instance 1
    n1, p1, seed1 = 32, 0.6, 601
    edges1 = get_graph(n1, p1, seed1)
    res1 = partition_cliques(n1, edges1)
    
    # Instance 2
    n2, p2, seed2 = 64, 0.55, 602
    edges2 = get_graph(n2, p2, seed2)
    res2 = partition_cliques(n2, edges2)
    
    return [res1, res2]
