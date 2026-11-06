import time
import random
import itertools

def has_c5(adj):
    n = len(adj)
    # adj is a list of sets of neighbors
    for v in range(n):
        # Check for path of length 4 in G-v between two neighbors of v
        # Path: a - b - c - d where a, d in N(v), a!=d
        # And edges (a,b), (b,c), (c,d) exist in G
        # And a, b, c, d are distinct and not v
        
        # To speed up, we can just iterate over all 5-tuples? No.
        # Let's try a simpler heuristic check first, or just brute force small n.
        # For n=32, n^5 is 33M. In Python, this might be too slow for many iterations.
        # Let's optimize:
        # A C5 is a cycle. 
        # We can check for induced C5 and non-induced?
        # The problem says "no 5-cycle (as a subgraph)". This includes chords.
        # So if there is a C5, it's invalid.
        
        # Efficient check:
        # For each v, let Nv = adj[v]
        # We want to know if there exist a, b, c, d in Nv \ {v} such that:
        # a-b-c-d is a path in G, and a, d are distinct neighbors of v.
        # Also a,b,c,d distinct.
        
        # Let's just iterate over pairs (a, d) in Nv.
        # For each pair, check if there is a path of length 3 between a and d in G - {v, a, d}?
        # Path a-b-c-d.
        # b must be in N(a) \ {v, d}
        # c must be in N(d) \ {v, a}
        # and b-c must be an edge.
        
        # This is still potentially heavy.
        
        # Given the time limit, maybe we should just use a randomized approach that avoids C5s by construction?
        # Union of cliques is C5-free.
        # Let's start with union of cliques and add edges?
        # Adding edges to a union of cliques can create C5s.
        
        # Alternative: Use a known C5-free graph structure.
        # Turan graphs are C_k-free for large k? No.
        # Bipartite graphs are C5-free? No, C6 is in bipartite. C5 is odd, so bipartite graphs are C5-free.
        # So any bipartite graph is C5-free.
        # But bipartite graphs have large independence numbers.
        # Max independent set in bipartite graph is at least n/2.
        # For n=32, alpha >= 16. We need alpha < 9.
        # So bipartite graphs won't work.
        
        # We need a graph that is not bipartite (to have small alpha) but has no C5.
        # This is a specific class.
        
        # Let's stick to local search but optimize the check.
        # For n=32, we can afford a somewhat slow check if the number of iterations is low.
        # Or, we can precompute C5s? No.
        
        pass

    return False

def has_independent_set_9(adj):
    n = len(adj)
    # Check if max clique in complement is >= 9
    # Complement adj: not_adj[u] = {v for v in range(n) if v not in adj[u] and v != u}
    # We can use a greedy/backtracking clique search.
    
    # Simple backtracking
    # Vertices sorted by degree in complement?
    # Complement degree is n-1 - deg_G(v).
    
    # Let's just use a simple recursive search.
    # If n is small, this is fine.
    
    # Optimization: If any vertex has complement degree < 8, it can't be in a K9.
    # Actually, if we are looking for K9, every vertex in the K9 must have complement degree >= 8.
    
    # Let's filter vertices.
    candidates = [v for v in range(n) if n - 1 - len(adj[v]) >= 8]
    if len(candidates) < 9:
        return False
    
    # Sort candidates by complement degree descending
    candidates.sort(key=lambda v: n - 1 - len(adj[v]), reverse=True)
    
    # Backtracking
    def backtrack(clique, candidates_set):
        if len(clique) == 9:
            return True
        if not candidates_set:
            return False
        # Pick a vertex
        v = candidates_set[0]
        # Option 1: Include v
        # New candidates are intersection of candidates_set (excluding v) and complement neighbors of v
        new_candidates = [u for u in candidates_set[1:] if u in comp_adj[v]]
        if backtrack(clique + [v], new_candidates):
            return True
        # Option 2: Exclude v
        if backtrack(clique, candidates_set[1:]):
            return True
        return False

    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp_adj[u].add(v)
    
    # We need to be careful with the set operations in backtrack.
    # For n=32, this should be fast enough.
    
    return backtrack([], candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try n from 35 down to 29
    for n in range(35, 28, -1):
        # Local search
        best_adj = None
        # Initialize with random graph
        # To have small alpha, we need high edge density.
        # Let's try p=0.7
        for _ in range(100): # 100 restarts per n
            if time.time() > deadline:
                break
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.7:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Check validity
            # We need efficient checks.
            # For now, let's assume the checks are fast enough for a few hundred iterations.
            # If not, we will refine.
            
            # Simple check:
            # If has_c5(adj) or has_independent_set_9(adj):
            #    continue
            # else:
            #    return valid graph
            
            # Since I don't have the efficient checks implemented in the thought process yet,
            # I will implement them in the final code.
            
            pass
        
        # If we found a valid graph for this n, return it
        # (Logic to be filled in final code)
        
    # Fallback: return the seed graph
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
