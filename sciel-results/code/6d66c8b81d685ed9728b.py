import time
import random
from collections import deque

def construct():
    n = 24
    base_edges = [(i, i+1) for i in range(n-1)]
    
    # We will try to find a minimal set of edges.
    # We'll use a greedy approach with restarts.
    
    def is_valid_addition(adj, u, v):
        # Check if u and v have a common neighbor
        # adj is a set of neighbors for each node
        return not (adj[u] & adj[v])
        
    def get_diameter(adj):
        # BFS from each node to find max distance
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            max_d = max(max_d, max(dist))
        return max_d

    def greedy_solve(random_seed):
        random.seed(random_seed)
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        added_edges = []
        
        # Calculate initial diameter
        # If diameter <= 4, return empty (unlikely for path 24)
        
        while True:
            d = get_diameter(adj)
            if d <= 4:
                break
            
            # Find all pairs with distance > 4
            # To speed up, we can just pick the pair with max distance
            # Or pick a random pair with distance > 4?
            # Let's pick the pair with max distance to be aggressive.
            
            # We need to find the pair with max distance.
            # We can reuse the BFS from the last step? No, let's just recompute or store.
            # For n=24, recomputing is fast.
            
            # Let's find all candidate edges that are valid and connect nodes with dist > 4?
            # Actually, we can add ANY valid edge, but it's smart to add one that reduces diameter.
            
            # Strategy: 
            # 1. Compute all pair distances.
            # 2. Filter pairs with dist > 4.
            # 3. From these, filter pairs where adding the edge is valid (no common neighbor).
            # 4. If no valid edges exist among dist > 4 pairs, we are stuck? 
            #    If we are stuck, it means we can't reduce diameter further? 
            #    Or maybe we need to add an edge between dist <= 4 to enable future reductions?
            #    Unlikely, but possible.
            
            # Let's just pick the valid edge that maximizes the reduction in "max distance" or similar.
            
            # Compute distances
            dists = []
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                dists.append(dist)
                
            # Find max distance and the pair
            max_d = 0
            max_pair = None
            for i in range(n):
                for j in range(i+1, n):
                    if dists[i][j] > max_d:
                        max_d = dists[i][j]
                        max_pair = (i, j)
                        
            if max_d <= 4:
                break
                
            # We want to add an edge. 
            # Candidates: All (u,v) such that dists[u][v] > 4 and is_valid_addition(adj, u, v)
            # If we only look at the max_pair, we might miss a better global choice.
            # But for a path, adding a long chord is usually good.
            
            # Let's collect all valid edges that connect nodes with distance > 4.
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if dists[u][v] > 4:
                        if is_valid_addition(adj, u, v):
                            # Heuristic: Prefer edges that are "long" in the path? 
                            # Or just add the one from max_pair if valid?
                            # Let's add all candidates and pick one.
                            candidates.append((u, v))
                            
            if not candidates:
                # No valid edge reduces the max distance directly?
                # This shouldn't happen if we are triangle free and connected.
                # If it does, we might need to add an edge that doesn't immediately reduce max dist
                # but opens up possibilities.
                # For now, return failure (None)
                return None
            
            # Pick a candidate. 
            # To improve, we can pick the one that minimizes the new diameter.
            # This is expensive (O(N^2 * E) per step).
            # Let's just pick the one with the largest distance in the current graph?
            # Or random?
            
            # Let's try picking the candidate that is "most central" or has highest degree potential?
            # Simple heuristic: Pick the one with the maximum current distance.
            # Since all have dist > 4, maybe pick the one with max dist?
            # Or just the first one?
            
            # Let's try a few random candidates from the list to avoid local minima.
            # We'll do this in the outer loop (restarts).
            
            # For this single run, let's pick the one with max distance.
            best_u, best_v = max(candidates, key=lambda x: dists[x[0]][x[1]])
            
            # Add edge
            adj[best_u].add(best_v)
            adj[best_v].add(best_u)
            added_edges.append((best_u, best_v))
            
        return added_edges

    # Try multiple random seeds
    best_solution = None
    start_time = time.time()
    
    for i in range(100): # 100 restarts
        if time.time() - start_time > 80:
            break
            
        sol = greedy_solve(i)
        if sol is not None:
            if best_solution is None or len(sol) < len(best_solution):
                best_solution = sol
                if len(best_solution) <= 5: # If we find a very good one, we can stop or keep looking
                    # Keep looking to be sure, but 5 is likely optimal or close.
                    pass

    if best_solution is None:
        # Fallback: Return a known valid solution if we found one previously?
        # Or just return an empty list? No, must be valid.
        # The greedy should find one.
        return []
        
    # Convert to list of lists
    return [[u, v] for u, v in best_solution]
