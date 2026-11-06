import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 300  # 5 minutes
    
    # Try Paley graph on 41
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges_41 = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Check if P_41 is (5,5)-free
    if is_55_free(q, edges_41):
        return q, edges_41
        
    # Try Paley graph on 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges_37 = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # P_37 is known to be (5,5)-free, but verify just in case
    if is_55_free(q, edges_37):
        return q, edges_37
        
    # Fallback: Return empty graph or smallest valid
    return 0, []

def is_55_free(n, edges):
    # Build adjacency matrix or set
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check for K5
    # Simple check: iterate over combinations of 5
    # This is O(n^5) which is too slow for n=41?
    # 41^5 is ~ 115 million. Might be slow in Python.
    # Optimized: Check cliques of size 5.
    
    # Use a faster check?
    # For n=41, 115M iterations is ~ 10-30 seconds in Python.
    # We have 300s.
    
    from itertools import combinations
    
    # Check K5
    for combo in combinations(range(n), 5):
        is_clique = True
        for i in range(5):
            for j in range(i+1, 5):
                if combo[j] not in adj[combo[i]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return False
            
    # Check Independent Set of size 5
    # Independent set in G is Clique in G_complement
    # We can just check if all pairs in combo are NOT connected
    for combo in combinations(range(n), 5):
        is_independent = True
        for i in range(5):
            for j in range(i+1, 5):
                if combo[j] in adj[combo[i]]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            return False
            
    return True
