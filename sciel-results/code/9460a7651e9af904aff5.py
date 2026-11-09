import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_all(adj, n):
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    queue.append(v)
    return dist

def _check_solution(added_edges, n):
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check triangle-free (bipartite check is faster, but this is explicit)
    # Since we only add even-odd edges, it's always bipartite if path is bipartite.
    # But let's double check no duplicate edges or self loops.
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        if u == v: return False
        pair = (min(u,v), max(u,v))
        if pair in edge_set: return False
        edge_set.add(pair)
        
    # Diameter check
    dist = _bfs_all(adj, n)
    for i in range(n):
        for j in range(n):
            if dist[i][j] == -1 or dist[i][j] > 4:
                return False
    return True

def construct():
    start_time = time.time()
    n = P['n']
    deadline = 55 # Safety margin
    
    # Generate all valid candidate edges (even-odd, not in path)
    path_edges = set((min(u,v), max(u,v)) for u,v in P['edges'])
    candidates = []
    for e in range(0, n, 2):
        for o in range(1, n, 2):
            pair = (min(e, o), max(e, o))
            if pair not in path_edges:
                candidates.append(pair)
                
    # We want to find the smallest k such that a valid solution exists.
    # We know 10 works. Let's try 9, then 8.
    
    best_solution = None
    
    # Try k=9
    # Use a heuristic: Start with a random set of 9, or greedy.
    # Greedy: Pick edge that reduces diameter most.
    
    def greedy_init(k):
        # Start with empty set
        selected = []
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
            
        # Compute initial distances
        dist = _bfs_all(adj, n)
        
        # Identify pairs with dist > 4
        bad_pairs = [(i,j) for i in range(n) for j in range(i+1,n) if dist[i][j] > 4]
        
        while len(selected) < k and bad_pairs:
            best_edge = None
            best_improvement = 0
            for e in candidates:
                if e in selected: continue
                u, v = e
                # Check if adding e reduces the number of bad pairs
                # Approximate: check if e helps connect the far ends
                # For speed, just pick random or highest degree potential
                # Let's try to pick edge that minimizes max distance
                # This is slow. Let's use a simpler heuristic.
                # Pick edge connecting to a hub.
                pass
            
            # Fallback: Random selection from candidates that are "central"
            # Central vertices: 10, 11, 12, 13
            # Try to connect these to far ends.
            if not selected:
                 # Initialize with some central connections
                 central = [10, 11, 12, 13]
                 for c in central:
                     # Connect to far ends
                     if c % 2 == 0:
                         targets = [1, 3, 21, 23]
                     else:
                         targets = [0, 2, 22, 24-1] # 23 is odd
                     for t in targets:
                         if t < n:
                             pair = (min(c,t), max(c,t))
                             if pair in candidates and pair not in selected:
                                 selected.append(pair)
                                 if len(selected) >= k: break
                     if len(selected) >= k: break
            else:
                 # Add random candidate
                 remaining = [c for c in candidates if c not in selected]
                 if remaining:
                     selected.append(random.choice(remaining))
                     
        return selected

    # Better approach: Local Search from a known 10-edge solution or random 9-edge set.
    
    # Let's try to find a 9-edge solution by modifying a 10-edge one.
    # Or just run a random restart local search for k=9.
    
    for k in [9, 8]:
        found = False
        start_search = time.time()
        while (time.time() - start_search) < 10: # 10s per k
            # Generate random set of k edges
            # Bias towards central vertices
            central_verts = [8, 9, 10, 11, 12, 13, 14, 15]
            random_edges = set()
            attempts = 0
            while len(random_edges) < k and attempts < 100:
                e = random.choice(candidates)
                # Prefer edges involving central vertices
                if e[0] in central_verts or e[1] in central_verts:
                    random_edges.add(e)
                else:
                    random_edges.add(e)
                attempts += 1
            
            if len(random_edges) < k: continue
            
            edges_list = list(random_edges)
            
            # Local search: Try to swap one edge for another to improve validity
            # Validity is binary (True/False). We need to get to True.
            # So we need a scoring function or just random walk.
            
            current = edges_list
            for _ in range(100): # 100 steps
                if _check_solution(current, n):
                    best_solution = current
                    found = True
                    break
                
                # Pick a random edge to remove
                idx = random.randint(0, len(current)-1)
                removed = current.pop(idx)
                
                # Pick a random edge to add
                # Prefer edges that are "central" or connect far nodes
                add_candidates = [c for c in candidates if c not in current]
                if not add_candidates:
                    current.append(removed)
                    break
                added = random.choice(add_candidates)
                current.append(added)
                
                # If time is running out, break
                if time.time() - start_search > 10:
                    break
            
            if found:
                break
        
        if found:
            break

    if best_solution:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback to 10 edges if 9/8 not found
    # Use the known 10-edge solution from inspiration
    # Hub 12 connected to odds?
    # Let's construct a valid 10-edge solution.
    # 12 (even) connected to 1, 3, 5, 7, 9, 15, 17, 19, 21, 23?
    # That's 10 edges.
    # Check diameter.
    # 0 to 23: 0-1-12-23? No, 12-23 is edge. 0-1 is path. 1-12 is edge.
    # 0 -> 1 -> 12 -> 23. Length 3. OK.
    # 0 to 22: 0-1-12-11-22? No.
    # 12 connected to 11? No, 12-11 is path.
    # 0 to 22: 0-1-12-11-10-...-22?
    # d(12, 22): 12-13-14-15-16-17-18-19-20-21-22 (10 steps).
    # Is there a shorter path?
    # 12 connected to 11? No.
    # 12 connected to 13? No.
    # 12 connected to odds.
    # 22 is even.
    # Path 12 -> 11 (path) -> 10 (path) ...
    # d(12, 22) via path is 10.
    # We need d(12, 22) <= 4.
    # So 12 must be connected to something close to 22.
    # 22 is connected to 21, 23.
    # If 12 is connected to 21 (odd), then 12-21-22 is length 2.
    # So we need 12 connected to 21.
    # Similarly for 0. 12 connected to 1. 12-1-0 is length 2.
    # So 12 connected to 1, 21.
    # What about 2? d(12, 2). 12-1-0-1-2? No.
    # 12-1-2 is length 2. (1-2 is path).
    # What about 4? 12-3-4? If 12 connected to 3.
    # So if 12 is connected to 1, 3, 5, 7, 9, 15, 17, 19, 21, 23.
    # d(12, 0) = 2 (12-1-0).
    # d(12, 2) = 2 (12-1-2).
    # d(12, 4) = 2 (12-3-4).
    # d(12, 6) = 2 (12-5-6).
    # d(12, 8) = 2 (12-7-8).
    # d(12, 10) = 2 (12-11-10).
    # d(12, 14) = 2 (12-13-14).
    # d(12, 16) = 2 (12-15-16).
    # d(12, 18) = 2 (12-17-18).
    # d(12, 20) = 2 (12-19-20).
    # d(12, 22) = 2 (12-21-22).
    # d(12, 24) N/A.
    
    # So any two evens $e_1, e_2$ have $d(e_1, e_2) \le d(e_1, 12) + d(12, e_2) \le 2 + 2 = 4$.
    # Any two odds $o_1, o_2$ have $d(o_1, o_2) \le d(o_1, 12) + d(12, o_2) \le 1 + 1 = 2$.
    # Any even $e$ and odd $o$:
    # $d(e, o) \le d(e, 12) + d(12, o) \le 2 + 1 = 3$.
    
    # So this 10-edge solution is VALID.
    # Edges: (12, 1), (12, 3), (12, 5), (12, 7), (12, 9), (12, 15), (12, 17), (12, 18-1=17? No 17 is odd), (12, 19), (12, 21), (12, 23).
    # Wait, 12 is even. Odds are 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23.
    # Path edges already include (11,12) and (12,13).
    # So we don't need to add (12, 11) or (12, 13).
    # We need to add edges to the other odds.
    # Odds: 1, 3, 5, 7, 9, 15, 17, 19, 21, 23.
    # Count: 10 edges.
    
    fallback_edges = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9], [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    
    if _check_solution(fallback_edges, n):
        return [[int(u), int(v)] for u, v in fallback_edges]
        
    # If even 10 fails (shouldn't), return empty or best effort
    return []
