"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Returns (n, edges) for a (5,5)-free graph on n vertices.
    Strategy:
    1. Start with Paley graph on 37 vertices (known (5,5)-free).
    2. Use local search (hill climbing) to try to add vertices or rearrange edges
       to maximize n while maintaining the (5,5)-free property.
       However, simply adding vertices to a (5,5)-free graph is hard.
       A better approach for R(5,5) is to look for known constructions or use
       a randomized search for larger graphs.
       
       Actually, the Paley graph on 37 vertices is a specific construction.
       There is a known 42-vertex graph by Exoo.
       Since we cannot hardcode the 42-vertex graph easily without a database,
       we can try to construct a larger graph via random generation and local search,
       but 300s is a lot of time. 
       
       Let's try to find the Exoo graph or a similar one.
       The Exoo graph is a specific (5,5)-free graph on 42 vertices.
       If we can generate it or find it, we get 42.
       
       Alternative: Use a randomized local search to build a graph from scratch.
       Start with n=40, 41, 42... and try to satisfy constraints.
       
       Let's try a simple hill climbing approach:
       1. Start with a random graph on n vertices.
       2. Check for K5 or I5.
       3. If found, remove an edge from the K5 or add an edge to the I5.
       4. Repeat until no K5 or I5.
       5. If successful, increase n.
       
       This is computationally expensive but might work within 300s for small n.
       
       Let's implement a checker for K5 and I5.
       For n=42, checking all 5-subsets is C(42,5) = 850668, which is manageable.
       
       We will try to build a graph on n vertices for n in range(43, 36, -1) or similar.
       Actually, we want the MAXIMUM n. So we try n=46, 45, ..., 37.
       If we find a valid graph for n=42, we return it.
       
       Let's try to use the known Exoo graph structure if possible, or just search.
       Searching for a (5,5)-free graph on 42 vertices is a known hard problem, but
       random restarts might find one.
       
       Let's write a function to check if a graph is (5,5)-free.
       """
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    def is_55_free(n, edges):
        # Build adjacency matrix or set of neighbors
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for K5
        for comb in combinations(range(n), 5):
            # Check if all pairs in comb are connected
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    if comb[j] not in adj[comb[i]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
        
        # Check for I5 (independent set of size 5)
        for comb in combinations(range(n), 5):
            # Check if no pairs in comb are connected
            is_independent = True
            for i in range(5):
                for j in range(i+1, 5):
                    if comb[j] in adj[comb[i]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
        
        return True

    # Start with the Paley graph on 37 vertices as a fallback
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    best_n = q
    best_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to find larger graphs
    for n in range(46, 36, -1):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(50):  # Limited attempts per n
            if time.time() > deadline:
                break
            # Random graph with p=0.5
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.append((i, j))
            
            if is_55_free(n, edges):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
    return best_n, best_edges
# EVOLVE-BLOCK-END
