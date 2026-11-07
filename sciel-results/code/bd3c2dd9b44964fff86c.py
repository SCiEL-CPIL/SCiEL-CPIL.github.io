import sys
import time
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}
    n = P['n']
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Pre-compute all valid cross-parity edges
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0:
                if (u, v) not in base_edges:
                    all_candidates.append((u, v))
                    
    # Helper to check if adding edge (u,v) creates a triangle
    def is_triangle_free_add(adj, u, v):
        # Check common neighbors
        # adj is a list of sets
        common = adj[u].intersection(adj[v])
        return len(common) == 0

    # Helper to compute diameter
    def get_diameter(adj):
        max_dist = 0
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
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist

    # Initial adjacency list
    initial_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        initial_adj[u].add(v)
        initial_adj[v].add(u)
        
    # State: set of added edges
    added_edges = set()
    
    # 1. Greedy Phase
    while True:
        if time.time() > deadline - 5.0:
            break
            
        # Build current adjacency
        adj = [set(s) for s in initial_adj] # Copy
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        diam = get_diameter(adj)
        if diam <= 4:
            break
            
        best_edge = None
        best_diam = float('inf')
        
        for u, v in all_candidates:
            if (u, v) in added_edges or (v, u) in added_edges:
                continue
            # Check triangle constraint
            if not is_triangle_free_add(adj, u, v):
                continue
                
            # Try adding
            adj[u].add(v)
            adj[v].add(u)
            d = get_diameter(adj)
            adj[u].remove(v)
            adj[v].remove(u)
            
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        added_edges.add(best_edge)
        
    # 2. Pruning Phase
    # Try to remove each edge
    changed = True
    while changed:
        if time.time() > deadline - 5.0:
            break
        changed = False
        for e in list(added_edges):
            added_edges.remove(e)
            adj = [set(s) for s in initial_adj]
            for u, v in added_edges:
                adj[u].add(v)
                adj[v].add(u)
            d = get_diameter(adj)
            if d > 4:
                added_edges.add(e)
            else:
                changed = True
                break # Restart loop as adjacency changed
                
    # 3. Local Search (2-swap) to try to reduce count further
    # Try swapping one added edge with one candidate edge
    # This might allow a subsequent prune
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Snapshot current best
        current_count = len(added_edges)
        
        for e_remove in list(added_edges):
            for e_add in all_candidates:
                if e_add in added_edges or e_add == e_remove or (e_add[1], e_add[0]) in added_edges:
                    continue
                
                # Check if e_add is valid in graph without e_remove
                # Build adj without e_remove
                adj = [set(s) for s in initial_adj]
                for u, v in added_edges:
                    if u != e_remove[0] and u != e_remove[1]: # Simple check, better to check tuple
                        pass # This logic is flawed, let's do it properly
                
                # Properly build adj without e_remove
                adj = [set(s) for s in initial_adj]
                for u, v in added_edges:
                    if (u, v) != e_remove and (v, u) != e_remove:
                        adj[u].add(v)
                        adj[v].add(u)
                        
                # Check triangle for e_add
                if not is_triangle_free_add(adj, e_add[0], e_add[1]):
                    continue
                    
                # Add e_add
                u, v = e_add
                adj[u].add(v)
                adj[v].add(u)
                
                d = get_diameter(adj)
                
                if d <= 4:
                    # We have a valid graph with same number of edges
                    # Does this allow us to prune?
                    # Let's try to prune this new state
                    new_added = set(added_edges)
                    new_added.remove(e_remove)
                    new_added.add(e_add)
                    
                    # Prune new_added
                    prune_changed = True
                    while prune_changed:
                        if time.time() > deadline - 5.0:
                            break
                        prune_changed = False
                        for pe in list(new_added):
                            new_added.remove(pe)
                            adj_p = [set(s) for s in initial_adj]
                            for pu, pv in new_added:
                                adj_p[pu].add(pv)
                                adj_p[pv].add(pu)
                            dp = get_diameter(adj_p)
                            if dp > 4:
                                new_added.add(pe)
                            else:
                                prune_changed = True
                                break
                                
                    if len(new_added) < current_count:
                        added_edges = new_added
                        improved = True
                        break
            if improved:
                break
        if improved:
            break

    # Final check and return
    # Ensure types are correct
    result = [[u, v] for u, v in added_edges]
    return result

# To ensure the function is defined and callable as required
# The prompt asks to keep function names and signatures.
# The current program defines construct().

if __name__ == "__main__":
    # Self-test
    edges = construct()
    # Basic validation
    n = 24
    base_edges = set()
    for i in range(n-1):
        base_edges.add((i, i+1))
        base_edges.add((i+1, i))
        
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    valid = True
    for u, v in edges:
        if (u, v) in base_edges:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        
    if valid:
        # Check triangle free
        for i in range(n):
            for j in adj[i]:
                for k in adj[j]:
                    if k in adj[i] and i != j and j != k and i != k:
                        # Triangle i-j-k
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
    if valid:
        # Check diameter
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
            if -1 in dist:
                valid = False
                break
            max_d = max(max_d, max(dist))
        if max_d > 4:
            valid = False
            
    if valid:
        print(f"Valid solution with {len(edges)} edges")
    else:
        print("Invalid solution")
