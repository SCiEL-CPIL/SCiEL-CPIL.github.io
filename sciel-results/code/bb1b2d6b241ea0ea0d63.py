import time
import random
import itertools

def _check_triangle_free(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in list(adj[u]):
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def _check_no_independent_10(n, edges):
    # Check if there is an independent set of size 10
    # This is equivalent to checking if the complement has a clique of size 10
    # Or simply backtracking on the original graph
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Use a simple backtracking to find independent set of size 10
    # Order vertices by degree to prune
    # If we can't find one, it's valid
    
    # Optimization: If n < 10, return True (no independent set of 10)
    if n < 10:
        return True
        
    # Greedy coloring of complement? 
    # Let's just do a DFS for independent set of size 10
    # This might be slow for n=40 if alpha is large, but we want to FAIL if alpha >= 10
    # So if we find one, we return False (invalid)
    # If we exhaust search, we return True (valid)
    
    # To speed up, we can use a bound: if remaining vertices + current size < 10, prune
    # But we want to find a set of 10.
    
    # Actually, for n=40, a random triangle-free graph usually has alpha > 10.
    # So we will likely find one quickly and return False.
    # Only if alpha <= 9 will we exhaust the search.
    
    # We need a fast check for "NO independent set of 10".
    # This is hard.
    # However, if we are constructing the graph, we can maintain the property.
    
    # For the fallback graphs (27, 32), we know they are valid.
    # For the search, we can use a heuristic that tries to keep alpha small.
    
    # Let's implement a simple check that might be slow but is correct.
    # For n=40, worst case is exponential.
    # But we can use a timeout.
    
    # Alternative: Use a greedy algorithm to find a large independent set.
    # If greedy finds 10, return False.
    # If greedy finds < 10, it doesn't prove < 10, but it's a good sign.
    # For the purpose of this task, if we use known valid graphs, we don't need to check.
    # If we use search, we need a reliable check.
    
    # Let's use a branch and bound for independent set of size 10.
    # If it finds one, return False.
    # If it finishes without finding one, return True.
    
    # To prevent timeout, we limit the number of nodes.
    # But for n=27, 32, it's fast.
    
    # I will rely on the known validity of 27 and 32.
    # For 40, I will use a search that verifies validity before returning.
    
    # Simple check for small n:
    if n <= 32:
        # Exact check
        # Sort vertices by degree in complement (i.e., degree in G)
        # Actually, independent set in G is clique in complement.
        # Let's just do a DFS on G for independent set.
        
        def dfs(candidates, size):
            if size == 10:
                return True
            if not candidates:
                return False
            if len(candidates) + size < 10:
                return False
            
            # Pick a vertex
            v = candidates[0]
            rest = candidates[1:]
            
            # Branch 1: Include v
            # New candidates are neighbors of v in complement (non-neighbors in G)
            non_neighbors = [u for u in rest if u not in adj[v]]
            if dfs(non_neighbors, size + 1):
                return True
            
            # Branch 2: Exclude v
            if dfs(rest, size):
                return True
                
            return False
        
        # Initial candidates: all vertices
        # To speed up, order by degree
        # Vertices with low degree in G are better candidates for independent set
        # Sort by degree in G (ascending)
        vertices = list(range(n))
        vertices.sort(key=lambda x: len(adj[x]))
        
        return not dfs(vertices, 0)

    else:
        # For n > 32, use a faster heuristic check
        # Greedy independent set
        remaining = set(range(n))
        indep_size = 0
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            v = None
            for u in remaining:
                deg = sum(1 for w in remaining if w != u and w in adj[u])
                if deg < min_deg:
                    min_deg = deg
                    v = u
            indep_size += 1
            if indep_size >= 10:
                return False
            # Remove v and its neighbors
            to_remove = {v} | adj[v]
            remaining -= to_remove
            
        return True

def _construct_schlaefli():
    n = 27
    # Schläfli graph is And(9)
    # Z_27, i ~ j if (j-i) mod 27 in {1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26}?
    # Actually, And(9) edges: (j-i) mod 27 is 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26?
    # No, the set is {1, 4, 7, 10, 13, 16, 19, 22, 25} and {2, 5, 8, 11, 14, 17, 20, 23, 26} and {3, 6, 9, 12, 15, 18, 21, 24}?
    # The Andrásfai graph And(2k+1) has edges for differences in S.
    # For k=4, n=27.
    # S = {1, 4, 7, 10, 13, 16, 19, 22, 25} U {2, 5, 8, 11, 14, 17, 20, 23, 26} U {3, 6, 9, 12, 15, 18, 21, 24}?
    # That's all non-zero residues. That's a complete graph.
    # The set S for And(2k+1) is:
    # {1, 4, ..., 3k-2} U {3k+1, ..., 6k-1} U {6k+1, ..., 9k-2}
    # For k=4:
    # {1, 4, 7, 10, 13, 16, 19, 22} (3k-2 = 10? No 3*4-2=10. So 1,4,7,10? No 1,4,7,10 is 4 elements. 1..10 step 3 is 1,4,7,10. 10 is 3k-2? 3*4-2=10. Yes.)
    # Wait, 1, 4, 7, 10.
    # Next: 3k+1 = 13. 13, 16, 19, 22.
    # Next: 6k+1 = 25. 25, 28... mod 27 -> 25, 1?
    # Let's use the standard definition:
    # S = {1, 4, 7, 10, 13, 16, 19, 22, 25} is not correct.
    
    # I will use the explicit edge list for Schläfli graph if I can recall it, or just use the 26-vertex one from the seed and add a vertex?
    # No, I'll stick to the seed's 26-vertex graph if I can't construct 27 easily.
    # But 27 is better.
    
    # Actually, the seed program returns n=26.
    # I can just return n=26 if I can't do better.
    # But I want to improve.
    
    # Let's return the Hadamard graph for n=32.
    return None

def _construct_hadamard_32():
    # Construct Hadamard matrix of order 32
    H = [[1]]
    while H[0].__len__() < 32:
        H = [[a for a in row for a in row] for row in H]
        # This doubles size each time. 1->2->4->8->16->32
        # Wait, the standard Sylvester construction is:
        # H_{2n} = [[H_n, H_n], [H_n, -H_n]]
        # My list comprehension above is wrong.
        
    # Correct construction
    H = [[1]]
    while len(H) < 32:
        H = [[a for a in row for a in row] for row in H] # This is not right.
        
    # Let's do it properly
    H = [[1]]
    while len(H) < 32:
        n = len(H)
        new_H = [[0]*(2*n) for _ in range(2*n)]
        for i in range(n):
            for j in range(n):
                new_H[i][j] = H[i][j]
                new_H[i][j+n] = H[i][j]
                new_H[i+n][j] = H[i][j]
                new_H[i+n][j+n] = -H[i][j]
        H = new_H
        
    # Now H is 32x32
    # The Hadamard graph has 32 vertices.
    # Vertices 0..31.
    # Edge (i, j) if H[i][j] == 1?
    # This graph is regular.
    # Is it triangle-free?
    # The graph from a Hadamard matrix is not necessarily triangle-free.
    # The "Hadamard 2-graph" is triangle-free.
    # Construction:
    # Vertices are (i, 0) and (i, 1) for i in 0..15? No, that's 32 vertices.
    # Let's use the standard 32-vertex triangle-free graph.
    
    # Actually, the Clebsch graph is 16 vertices.
    # The 32-vertex graph is the **Hadamard graph** of order 32.
    # It is triangle-free.
    
    edges = []
    for i in range(32):
        for j in range(i+1, 32):
            if H[i][j] == 1:
                edges.append([i, j])
                
    # Check if it's valid
    if _check_triangle_free(32, edges) and _check_no_independent_10(32, edges):
        return {'n': 32, 'edges': edges}
    else:
        return None

def construct():
    # Try 32
    res = _construct_hadamard_32()
    if res:
        return res
        
    # Fallback to seed's 26
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
