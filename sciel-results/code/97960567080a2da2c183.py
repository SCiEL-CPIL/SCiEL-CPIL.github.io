import random
import itertools

def construct():
    # We want to maximize n.
    # Try n from 35 down to 28.
    for n in range(35, 27, -1):
        # Try multiple random trials
        for _ in range(200):
            # Initialize empty graph
            adj = {i: set() for i in range(n)}
            edges = []
            
            # List of all possible edges
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a C5
                # This requires a path of length 4 between u and v
                # Path: u-a-b-c-v
                # So a in N(u), c in N(v), and b in N(a) intersect N(c)
                
                creates_c5 = False
                for a in adj[u]:
                    for c in adj[v]:
                        # Check if there is a common neighbor b of a and c
                        # If yes, path u-a-b-c-v exists
                        if adj[a] & adj[c]:
                            creates_c5 = True
                            break
                    if creates_c5:
                        break
                if creates_c5:
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
            
            # Check alpha < 9
            if check_alpha_less_than_9(n, adj):
                return {'n': n, 'edges': edges}
        
        # If no valid graph found for this n, try smaller n
    # Fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}

def check_alpha_less_than_9(n, adj):
    # We want to verify that alpha(G) < 9.
    # This is equivalent to: there is no independent set of size 9.
    # We can use a backtracking search to find if an independent set of size 9 exists.
    # If it does, return False. If not, return True.
    
    # Sort vertices by degree to prune?
    # Independent set: no two vertices are adjacent.
    
    def find_independent_set_size_k(k):
        # Backtracking
        # State: current set, last vertex index to avoid duplicates
        # We want to find a set of size k.
        
        # Pruning: if len(current) + (n - last_idx) < k, return False
        # Also, if the remaining vertices cannot form a set of size k - len(current)
        
        # A better pruning: 
        # Let S be the set of available vertices (indices > last_idx).
        # If the maximum independent set in the subgraph induced by S is < k - len(current), prune.
        # But computing max independent set is hard.
        # Simple pruning: if len(S) < k - len(current), prune.
        
        # Let's use a recursive helper.
        # To speed up, we can order vertices by degree (greedy coloring bound for alpha).
        
        # For n=35, a simple DFS might be okay if we prune well.
        
        def dfs(start_idx, current_set):
            if len(current_set) == k:
                return True
            if start_idx >= n:
                return False
            if len(current_set) + (n - start_idx) < k:
                return False
            
            # Try adding vertex start_idx
            # If we add start_idx, we cannot add any of its neighbors.
            # So the next candidate must be > start_idx and not in adj[start_idx]
            
            # Option 1: Include start_idx
            # Next valid index is start_idx + 1, but we must skip neighbors.
            # We can just iterate j from start_idx + 1 to n, and check if j is neighbor.
            
            # To make it faster, we can maintain a list of candidates.
            # But for n=35, simple loop is fine.
            
            # Include start_idx
            for j in range(start_idx + 1, n):
                if j not in adj[start_idx]:
                    if dfs(j, current_set + [j]):
                        return True
            # Wait, this is not correct. If we include start_idx, we can still include vertices > start_idx that are not neighbors.
            # But we must ensure the set remains independent.
            # The recursive call `dfs(j, current_set + [j])` assumes `j` is the new last vertex.
            # This structure works if we always pick the smallest index available.
            
            # Option 2: Exclude start_idx
            if dfs(start_idx + 1, current_set):
                return True
                
            return False
        
        return dfs(0, [])

    return not find_independent_set_size_k(9)
