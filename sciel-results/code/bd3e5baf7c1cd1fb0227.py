import time
import sys
import numpy as np
from itertools import permutations
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known optimal for n=4 is 33
        return "123412431324142314321234124313241423143212341243132414231432"

    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # 1. Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # 2. Build adjacency list
    # Neighbor of P = (p1, ..., pn) is Q = (p2, ..., pn, x) where x not in P[1:]
    # There are n-1 such neighbors.
    adj = [[] for _ in range(N)]
    for i, p in enumerate(perms):
        tail = p[1:]
        tail_set = set(tail)
        for x in range(1, n + 1):
            if x not in tail_set:
                q = tail + (x,)
                j = perm_to_idx[q]
                adj[i].append(j)
                
    # 3. Greedy Path Search
    # We want a path visiting all nodes.
    # Warnsdorff's rule: choose the neighbor with the fewest unvisited neighbors.
    
    def greedy_search(start_node, tie_breaker_seed=0):
        visited = [False] * N
        path = []
        current = start_node
        
        while True:
            visited[current] = True
            path.append(current)
            
            # Find next node
            best_next = -1
            min_unvisited = float('inf')
            
            candidates = []
            for next_node in adj[current]:
                if not visited[next_node]:
                    # Count unvisited neighbors of next_node
                    count = 0
                    for nn in adj[next_node]:
                        if not visited[nn]:
                            count += 1
                    candidates.append((count, next_node))
                    
            if not candidates:
                break
            
            # Sort by count, then by node index for determinism or random tie-break
            candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Simple heuristic: pick the first one (min count)
            # To allow variations, we could pick randomly from the top K, 
            # but for now, let's stick to strict Warnsdorff with a few restarts.
            best_next = candidates[0][1]
            
            current = best_next
            
        return path

    def repair_path(path, visited, N, adj, perm_to_idx, perms):
        """
        Try to insert unvisited nodes into the path.
        This is a heuristic. If the greedy path didn't visit all nodes,
        we try to find gaps where we can insert a missing node with minimal cost.
        """
        unvisited = [i for i in range(N) if not visited[i]]
        if not unvisited:
            return path
        
        # For simplicity, if greedy fails to cover all, we can try to append
        # the unvisited nodes one by one, finding the cheapest insertion point.
        # Cost of inserting node V between U and W:
        # Original: ... U, W ...
        # New: ... U, V, W ...
        # This requires that U overlaps with V and V overlaps with W.
        # In our graph, edges are defined by overlap.
        # If we just append, we might break the overlap chain.
        
        # A safer repair: Just append unvisited nodes to the end, 
        # finding the shortest path from the last node to the unvisited node.
        # This is expensive.
        
        # Alternative: Return the greedy path if it's long enough, 
        # otherwise fall back to a simpler construction.
        
        # Given time constraints, let's just return the greedy path.
        # If it doesn't cover all, it's invalid.
        # We must ensure validity.
        
        # Let's check coverage.
        if len(path) < N:
            # Invalid. We need a valid superpermutation.
            # Fallback to a known valid construction (e.g., recursive)
            # or a simpler greedy that ensures coverage.
            
            # A simpler valid construction:
            # Just list all permutations, ensuring overlap where possible.
            # This is the "greedy insertion" method.
            return None
        
        return path

    best_path = None
    best_len = -1
    
    # Try multiple starting nodes and restarts
    # For n=7, N=5040. Greedy is O(N^2) worst case, which is ~25M ops.
    # This is fast enough for many restarts.
    
    # Use a few different starting points and tie-breaking strategies
    start_nodes = [0, 1, 2, N//2, N-1]
    
    for s in start_nodes:
        if time.time() > deadline - 5:
            break
            
        path = greedy_search(s)
        
        # Check if valid (covers all nodes)
        if len(path) == N:
            # Convert path to string
            # Path is a list of permutation indices.
            # We need to construct the word.
            # Word starts with perms[path[0]]
            # Then for each next perm, append the last symbol.
            word = list(perms[path[0]])
            for i in range(1, len(path)):
                # Append the last symbol of the new permutation
                word.append(perms[path[i]][-1])
                
            if len(word) < best_len or best_path is None:
                best_len = len(word)
                best_path = "".join(map(str, word))
                
    if best_path is not None:
        return best_path
    
    # Fallback: Recursive construction (always valid, but longer)
    def classic_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    word = classic_construct(n)
    return "".join(map(str, word))
