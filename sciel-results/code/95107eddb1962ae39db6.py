import time
from collections import deque
import numpy as np
import sys
import random
from itertools import combinations

# Increase recursion limit
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Find a simple path from s to t avoiding forbidden vertices (BFS for shortest)
    def find_shortest_path(s, t, forbidden):
        if s == t:
            return [s]
        if s in forbidden or t in forbidden:
            return None
        if t in adj[s]:
            return [s, t]
        visited = {s}
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in forbidden:
                    continue
                if v == t:
                    path = [t]
                    cur = t
                    while cur != s:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None
    
    # Check if B is a valid set of branches by finding vertex-disjoint paths for all pairs
    def check_subdivision(B):
        B_set = set(B)
        k = len(B)
        if k < 2:
            return None, None
        
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append((B[i], B[j]))
        
        # Try multiple pair orderings
        def pair_key(p):
            u, v = p
            if v in adj[u]:
                return (0, 0)
            cn = len((adj[u] & adj[v]) - B_set)
            return (1, -cn)
        
        orderings = [sorted(pairs, key=pair_key)]
        # Add a few random orderings
        for _ in range(3):
            shuffled = list(pairs)
            random.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            paths = {}
            forbidden = set(B_set)
            success = True
            
            for (u, v) in ordering:
                path = find_shortest_path(u, v, forbidden)
                if path is None:
                    success = False
                    break
                paths[(u, v)] = path
                for x in path[1:-1]:
                    forbidden.add(x)
            
            if success:
                # Verify
                all_interiors = set()
                valid = True
                for (u, v) in pairs:
                    p = paths.get((u, v))
                    if p is None or len(p) < 2:
                        valid = False
                        break
                    if p[0] != u and p[0] != v:
                        valid = False
                        break
                    if p[-1] != u and p[-1] != v:
                        valid = False
                        break
                    for i in range(len(p) - 1):
                        if p[i+1] not in adj[p[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                    for x in p[1:-1]:
                        if x in B_set:
                            valid = False
                            break
                    if not valid:
                        break
                    all_interiors.update(p[1:-1])
                
                if valid:
                    result_paths = [list(paths[(u, v)]) for (u, v) in pairs]
                    return result_paths, all_interiors
        
        return None, None
    
    # Local search
    def local_search(initial_B, max_iters=3000):
        current_B = list(initial_B)
        paths, interiors = check_subdivision(current_B)
        if paths is None:
            # Try smaller subsets
            found = False
            for sz in range(len(current_B) - 1, 1, -1):
                if found:
                    break
                for comb in combinations(current_B, sz):
                    paths, interiors = check_subdivision(list(comb))
                    if paths is not None:
                        current_B = list(comb)
                        found = True
                        break
            if not found:
                return None, None
        
        best_B = list(current_B)
        best_paths = paths
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            added = False
            B_set = set(current_B)
            candidates = [v for v in range(n) if v not in B_set]
            candidates.sort(key=lambda x: -degrees[x])
            
            for v in candidates:
                if time.time() > deadline:
                    break
                new_B = current_B + [v]
                new_paths, _ = check_subdivision(new_B)
                if new_paths is not None:
                    current_B = new_B
                    paths = new_paths
                    added = True
                    break
            
            if not added:
                swapped = False
                B_set = set(current_B)
                for rem in list(current_B):
                    if time.time() > deadline:
                        break
                    B_without = [b for b in current_B if b != rem]
                    for add in candidates:
                        if add in B_set:
                            continue
                        new_B = B_without + [add]
                        new_paths, _ = check_subdivision(new_B)
                        if new_paths is not None:
                            current_B = new_B
                            paths = new_paths
                            swapped = True
                            break
                    if swapped:
                        break
            
            if not added and not swapped:
                break
            
            if len(current_B) > len(best_B):
                best_B = list(current_B)
                best_paths = paths
        
        return best_B, best_paths
    
    best_B = None
    best_paths = None
    
    # 1. Greedy clique
    clique = []
    remaining = set(range(n))
    while remaining:
        v = max(remaining, key=lambda x: len(adj[x] & remaining))
        clique.append(v)
        remaining = remaining & adj[v]
    
    if len(clique) >= 2:
        paths, _ = check_subdivision(clique)
        if paths is not None:
            best_B = clique
            best_paths = paths
    
    # 2. Local search from clique
    if best_B is not None:
        res_B, res_paths = local_search(best_B)
        if res_B is not None and (best_B is None or len(res_B) > len(best_B)):
            best_B = res_B
            best_paths = res_paths
    
    # 3. Try high-degree pairs as seeds
    high_deg = sorted(range(n), key=lambda x: -degrees[x])
    for i in range(min(12, n)):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, min(12, n)):
            if time.time() > deadline - 5:
                break
            seed = [high_deg[i], high_deg[j]]
            paths, _ = check_subdivision(seed)
            if paths is not None:
                res_B, res_paths = local_search(seed, max_iters=1500)
                if res_B is not None and (best_B is None or len(res_B) > len(best_B)):
                    best_B = res_B
                    best_paths = res_paths
    
    # Fallback
    if best_B is None or len(best_B) < 2:
        u, v = P['edges'][0]
        best_B = [u, v]
        best_paths = [[u, v]]
    
    # Final verification
    final_paths, _ = check_subdivision(best_B)
    if final_paths is None:
        while len(best_B) > 2:
            best_B = best_B[:-1]
            final_paths, _ = check_subdivision(best_B)
            if final_paths is not None:
                break
        if final_paths is None:
            u, v = P['edges'][0]
            best_B = [u, v]
            final_paths = [[u, v]]
    
    return {'branches': [int(b) for b in best_B], 'paths': [[int(x) for x in p] for p in final_paths]}

# EVOLVE-BLOCK-END
