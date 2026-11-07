"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Build adjacency list for base graph
    adj_base = [[] for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
        
    # Candidate edges: only between even and odd vertices (to maintain triangle-freeness)
    # and not already in base
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2) != (v % 2):  # different parity
                if (min(u,v), max(u,v)) not in base_edges:
                    candidates.append((u, v))
    
    def compute_diameter(adj_list):
        """Compute diameter using BFS from each vertex."""
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj_list[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
                        if dist[neighbor] > max_dist:
                            max_dist = dist[neighbor]
        return max_dist

    def is_valid_solution(added_edges):
        # Check triangle-free (implied by parity if we only add cross-parity)
        # Check connected and diameter <= 4
        adj = [list(a) for a in adj_base]
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check connectivity and diameter
        diam = compute_diameter(adj)
        if diam == -1 or diam > 4:
            return False, diam
        return True, diam

    # Greedy approach with multiple restarts to find a good local minimum
    best_edges = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # Run greedy multiple times with different tie-breaking/randomization
    for restart in range(20):
        if time.time() > deadline:
            break
            
        added_edges = []
        current_adj = [list(a) for a in adj_base]
        
        # Greedy loop
        while True:
            diam, _ = is_valid_solution(added_edges)
            if diam <= 4:
                break
            if time.time() > deadline:
                break
            
            best_edge = None
            best_new_diam = float('inf')
            # To break ties, we can look at the number of pairs that become distance 4
            # But for now, just min diameter.
            
            # Shuffle candidates for randomization in restarts > 0
            cand_list = list(candidates)
            if restart > 0:
                random.shuffle(cand_list)
                
            for u, v in cand_list:
                edge_key = (min(u,v), max(u,v))
                # Check if already added
                if edge_key in [(min(e[0],e[1]), max(e[0],e[1])) for e in added_edges]:
                    continue
                
                # Temporarily add
                current_adj[u].append(v)
                current_adj[v].append(u)
                
                new_diam = compute_diameter(current_adj)
                
                # Undo
                current_adj[u].pop()
                current_adj[v].pop()
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                
            if best_edge is None:
                break
                
            u, v = best_edge
            current_adj[u].append(v)
            current_adj[v].append(u)
            added_edges.append([u, v])
            
        # Local Search: Try to remove edges
        if added_edges:
            improved = True
            while improved:
                improved = False
                for i in range(len(added_edges)):
                    # Try removing edge i
                    trial_edges = added_edges[:i] + added_edges[i+1:]
                    valid, _ = is_valid_solution(trial_edges)
                    if valid:
                        added_edges = trial_edges
                        improved = True
                        break # Restart the loop as indices changed
                    
        if len(added_edges) < best_count:
            best_count = len(added_edges)
            best_edges = added_edges[:]
            
    if best_edges is None:
        return []
        
    # Final verification
    valid, _ = is_valid_solution(best_edges)
    if not valid:
        return []
        
    return best_edges
# EVOLVE-BLOCK-END
