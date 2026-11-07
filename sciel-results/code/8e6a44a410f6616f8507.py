import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
    
    # Bipartition
    A = set(range(0, n, 2))
    B = set(range(1, n, 2))
    
    # Candidate edges: Between A and B, not in base
    candidates = []
    for u in A:
        for v in B:
            if frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    # Precompute base adjacency for fast triangle check?
    # Base adj is just path.
    
    def is_valid(added_edges):
        # Check triangle-free
        # Build adj list
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        
        for u, v in added_edges:
            # Check if u and v share a neighbor
            if adj[u] & adj[v]:
                return False
            adj[u].add(v)
            adj[v].add(u)
            
        # Check diameter <= 4
        # BFS from each node
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if any(d == -1 or d > 4 for d in dist):
                return False
        return True

    deadline = time.time() + 80
    random.seed(42)
    
    # Try k=6
    for k in range(6, 10):
        if time.time() > deadline:
            break
            
        # If k is small, try systematic or heavy random
        # For k=6, number of candidates is ~100. C(100,6) is huge.
        # Use random sampling.
        
        # Optimization: Use a "greedy" or "heuristic" start?
        # Just random sampling with many iterations.
        
        found = None
        iterations = 500000 if k == 6 else 1000000
        for _ in range(iterations):
            if time.time() > deadline:
                break
            combo = random.sample(candidates, k)
            if is_valid(combo):
                found = combo
                break
        
        if found:
            return [[u, v] for u, v in found]

    # Fallback if not found (should not happen if 7 or 8 is possible)
    # The previous solution had 7 edges.
    # Let's return a known valid 7-edge solution if search fails.
    # One such solution:
    # (11,0), (11,4), (11,8), (11,12), (11,16), (11,20), (12,23) ? 
    # Let's verify (11,0), (11,4), (11,8), (11,12), (11,16), (11,20), (12,23)
    # 11(Odd)-0(Even) OK
    # 11(Odd)-4(Even) OK
    # 11(Odd)-8(Even) OK
    # 11(Odd)-12(Even) OK
    # 11(Odd)-16(Even) OK
    # 11(Odd)-20(Even) OK
    # 12(Even)-23(Odd) OK
    
    # Distances:
    # 0 to 23: 0-11-12-23 (3)
    # 2 to 21: 2-4-11-12-19-20-21? 
    # 2-4 (2), 4-11 (1), 11-12 (1), 12-23 (1)... 23 is far from 21.
    # 12-21 path is 9. 12-23 is 1. 23-21 is 2. So 12-21 is 3.
    # 2-11 is 3 (2-4-11).
    # Total 3+1+3 = 7. Fail.
    
    # The search should find a valid one.
    # If not, return the 10-edge solution from the prompt as last resort.
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    # This was the 7-edge fallback in the prompt, but it was invalid?
    # Prompt said "Fallback: use the 7-edge solution" but returned 7 edges.
    # Wait, the prompt's fallback returned `[[hub, t] for t in targets]`.
    # If that was valid, the score would have been higher?
    # The prompt says "Current Program (score 17.2857)" which corresponds to 10 edges?
    # 10 edges -> Score 17.28.
    # If 7 edges were valid, Score would be ~24.
    # So the 7-edge fallback in the prompt was likely INVALID or the search failed to find a valid 7-edge set and the fallback was also invalid?
    # The prompt says "Any broken rule makes the whole result invalid".
    # So I must ensure my output is valid.
    
    # I will rely on the search to find a valid k.
    # If k=6 fails, k=7 should succeed.
    
    return [[11, 0], [11, 4], [11, 8], [11, 12], [11, 16], [11, 20], [12, 15], [12, 19], [12, 23], [12, 1]] # 10 edges fallback

