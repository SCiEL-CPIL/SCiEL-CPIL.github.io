import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 255
    target = 255
    
    # Candidates: numbers in [1,255] sharing a factor with 255 (=3*5*17)
    candidates = [i for i in range(1, n + 1) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    
    # Build adjacency
    m = len(candidates)
    adj = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = adj[j][i] = True
    
    # Find max clique containing 255 using branch and bound
    # Since 255 connects to all candidates, we just need max clique in candidate graph
    # that includes 255's index
    
    idx_255 = candidates.index(target)
    
    # Use a heuristic maximum clique algorithm
    # Tomita-style greedy coloring bound
    
    # Order vertices by degree
    degrees = [sum(adj[i]) for i in range(m)]
    order = sorted(range(m), key=lambda x: degrees[x], reverse=True)
    
    # We need clique containing idx_255
    # Let's try multiple strategies and take the best
    
    best_clique = set()
    best_size = 0
    
    # Strategy 1: All multiples of 3
    mult3 = [i for i in range(m) if candidates[i] % 3 == 0]
    if idx_255 in mult3:
        # Verify it's a clique
        is_clique = all(adj[mult3[i]][mult3[j]] for i in range(len(mult3)) for j in range(i+1, len(mult3)))
        if is_clique:
            best_clique = set(mult3)
            best_size = len(mult3)
    
    # Strategy 2: All multiples of 5
    mult5 = [i for i in range(m) if candidates[i] % 5 == 0]
    if idx_255 in mult5:
        is_clique = all(adj[mult5[i]][mult5[j]] for i in range(len(mult5)) for j in range(i+1, len(mult5)))
        if is_clique and len(mult5) > best_size:
            best_clique = set(mult5)
            best_size = len(mult5)
    
    # Strategy 3: All multiples of 17
    mult17 = [i for i in range(m) if candidates[i] % 17 == 0]
    if idx_255 in mult17:
        is_clique = all(adj[mult17[i]][mult17[j]] for i in range(len(mult17)) for j in range(i+1, len(mult17)))
        if is_clique and len(mult17) > best_size:
            best_clique = set(mult17)
            best_size = len(mult17)
    
    # Strategy 4: Try to expand from multiples of 3 by adding compatible vertices
    # A vertex v can be added to a set S if gcd(v, s) > 1 for all s in S
    # For multiples of 3, v must share a factor with every multiple of 3 in the set
    # This means v must be even (to share 2 with even multiples of 3) AND 
    # share factors with odd multiples of 3... which is hard
    
    # Actually, let's just do a proper maximum clique search with time limit
    # Use a simple recursive backtracking with pruning
    
    def max_clique_search(vertices, current_clique):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if not vertices:
            if len(current_clique) > best_size:
                best_clique = set(current_clique)
                best_size = len(current_clique)
            return
        
        # If even adding all remaining can't beat best, prune
        if len(current_clique) + len(vertices) <= best_size:
            return
        
        # Pick a vertex with high degree in the subgraph
        v = vertices[0]
        
        # Branch: include v
        new_vertices = [u for u in vertices[1:] if adj[v][u]]
        max_clique_search(new_vertices, current_clique + [v])
        
        # Branch: exclude v
        if time.time() < deadline:
            max_clique_search(vertices[1:], current_clique)
    
    # This is too slow for 80+ vertices. Let me use a better approach.
    
    # Better: Use the fact that we need a clique. 
    # The graph has a specific structure. Let me try a greedy approach with restarts.
    
    def greedy_clique(start_vertex):
        """Greedy maximum clique starting from start_vertex"""
        clique = [start_vertex]
        available = [v for v in range(m) if v != start_vertex and adj[start_vertex][v]]
        
        while available:
            # Pick vertex with max degree in available subgraph
            best_v = -1
            best_deg = -1
            for v in available:
                deg = sum(1 for u in available if u != v and adj[v][u])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            clique.append(best_v)
            available = [v for v in available if v != best_v and adj[best_v][v]]
        
        return clique
    
    # Try greedy from 255
    if time.time() < deadline:
        g_clique = greedy_clique(idx_255)
        if len(g_clique) > best_size:
            best_clique = set(g_clique)
            best_size = len(g_clique)
    
    # Try greedy from each candidate, keep those containing 255
    for start in range(m):
        if time.time() > deadline:
            break
        if start == idx_255:
            continue
        # Only try if 255 is adjacent to start (it is, by construction)
        g_clique = greedy_clique(start)
        if idx_255 in g_clique and len(g_clique) > best_size:
            best_clique = set(g_clique)
            best_size = len(g_clique)
    
    # Local search: try to swap vertices to improve
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_v in list(best_clique):
                if time.time() > deadline:
                    break
                new_set = best_clique - {remove_v}
                for add_v in range(m):
                    if add_v in new_set:
                        continue
                    if all(adj[add_v][u] for u in new_set):
                        if len(new_set) + 1 > len(best_clique):
                            best_clique = new_set | {add_v}
                            improved = True
                            break
    
    # Convert to values
    result = sorted(candidates[i] for i in best_clique)
    
    # Ensure 255 is present
    if target not in result:
        result.append(target)
        result = sorted(set(result))
    
    # Final validation
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # Fallback to multiples of 3
                return list(range(3, 256, 3))
    
    return result
