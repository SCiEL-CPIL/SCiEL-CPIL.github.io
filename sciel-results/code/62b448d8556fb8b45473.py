"""Construct a C4-free graph with high minimum degree."""
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a C4-free graph on n vertices with minimum degree >= n - 39.
    
    Approach: Use a probabilistic/exhaustive search for small n, or construct
    known C4-free graphs with high minimum degree.
    
    For R(C4, K1,39) > n, we need a C4-free graph on n vertices with min degree >= n-39.
    This means the complement has no vertex of degree >= 39, i.e., max degree in complement <= 38.
    
    Known results: The Erdős–Rényi construction or polarity graphs can give C4-free graphs
    with good degree properties.
    
    Let's try a systematic approach:
    - For n up to 47, min degree >= n-39 means:
      n=47: min deg >= 8
      n=46: min deg >= 7
      ...
      n=39: min deg >= 0 (trivial)
    
    We want to maximize n. Let's try to find a C4-free graph on 47 vertices with min degree 8,
    or on 46 vertices with min degree 7, etc.
    
    A well-known family: polarity graphs of projective planes.
    The polarity graph of PG(2, q) has q^2 + q + 1 vertices, each vertex has degree q+1 (or q+2 for the absolute points), and is C4-free.
    
    For q=5: 31 vertices, degree ~6-7. Not enough for n=47.
    For q=7: 57 vertices, too many.
    
    Let's try a different approach: use a random graph with controlled edge density and check for C4.
    
    Actually, let's think about this more carefully. We need a C4-free graph with high minimum degree.
    
    Another approach: start with a complete graph and remove edges to break all C4s while keeping min degree high.
    
    Or: use the fact that a graph is C4-free iff for every pair of vertices, they have at most one common neighbor.
    
    Let's try a greedy/randomized approach:
    1. Start with n=47, target min degree 8.
    2. Randomly generate graphs with average degree around 8-10.
    3. Check if C4-free and min degree >= 8.
    
    But this might be slow. Let's try a more structured approach.
    
    Actually, let's use a known construction: the Clebsch graph is C5-free but not necessarily C4-free.
    
    Let me try a simple randomized search with backtracking for smaller instances, and use known constructions for larger ones.
    
    For n=47, we need min degree >= 8. Let's try to build such a graph.
    
    Strategy: 
    - Use a random regular graph of degree 8 or 9 on 47 vertices.
    - Check if it's C4-free.
    - If not, try to fix it by removing/adding edges.
    
    Given time constraints, let's try a simpler approach: use a known C4-free graph from literature or construct one using a polarity graph variant.
    
    Actually, the simplest robust approach: try n from 47 down to 39, and for each n, try to find a C4-free graph with min degree >= n-39 using randomized search.
    """
    
    def has_c4(n, edges):
        """Check if graph has a C4."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # For each pair of vertices, check if they have >= 2 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common = 0
                for k in range(n):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        common += 1
                        if common >= 2:
                            return True
        return False
    
    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0
    
    import random
    random.seed(42)
    
    # Try from n=47 down to n=39
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        
        # Try multiple random attempts
        for attempt in range(500):
            # Generate a random graph with edges chosen to achieve approximately target_min_deg
            # Use a random regular-ish graph approach
            edges = []
            edge_set = set()
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            
            # Try to build a graph where each vertex has degree around target_min_deg to target_min_deg+2
            # Use a configuration model or just random edges
            
            # Random edges
            all_possible = list(combinations(range(n), 2))
            random.shuffle(all_possible)
            
            # Start with a random subset
            num_edges = n * target_min_deg // 2 + random.randint(0, n)
            
            for u, v in all_possible[:num_edges]:
                edges.append((u, v))
                edge_set.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            # Check min degree
            if min_degree(n, edges) < target_min_deg:
                # Try to add edges to boost low-degree vertices
                deg = [len(adj[i]) for i in range(n)]
                for i in range(n):
                    while deg[i] < target_min_deg:
                        # Find a vertex j not adjacent to i
                        candidates = [j for j in range(n) if j != i and (i, j) not in edge_set and (j, i) not in edge_set]
                        if not candidates:
                            break
                        j = random.choice(candidates)
                        edges.append((i, j))
                        edge_set.add((i, j))
                        adj[i].append(j)
                        adj[j].append(i)
                        deg[i] += 1
                        deg[j] += 1
                
                if min_degree(n, edges) < target_min_deg:
                    continue
            
            # Check for C4
            if not has_c4(n, edges):
                # Success!
                return {'n': n, 'edges': [[u, v] for u, v in edges]}
        
        # If we couldn't find one for this n with random search, try a more structured approach
        # For now, continue to smaller n
    
    # Fallback: return the trivial solution
    return {'n': 39, 'edges': []}
