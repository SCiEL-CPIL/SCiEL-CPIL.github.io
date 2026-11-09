"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            comp = []
            stack = [i]
            visited.add(i)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    best_path = []
    
    # Strategy 1: For each component, try to find longest path
    for comp in components:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_set = set(comp)
        
        # For small components, try exhaustive DFS with pruning
        if comp_size <= 25:
            # Try all starting vertices
            for start in comp:
                if time.time() > deadline:
                    break
                path = [start]
                seen = {start}
                
                def dfs_extend(path, seen):
                    nonlocal best_path
                    if len(path) > len(best_path):
                        best_path = path[:]
                    
                    last = path[-1]
                    # Get unvisited neighbors
                    unvisited = [v for v in adj[last] if v in comp_set and v not in seen]
                    if not unvisited:
                        return
                    
                    # Warnsdorff: prefer neighbors with fewer onward options
                    def warnsdorff_score(v):
                        onward = len([w for w in adj[v] if w in comp_set and w not in seen and w != last])
                        return (onward, v)
                    
                    unvisited.sort(key=warnsdorff_score)
                    
                    for v in unvisited:
                        if time.time() > deadline:
                            return
                        path.append(v)
                        seen.add(v)
                        dfs_extend(path, seen)
                        path.pop()
                        seen.remove(v)
                
                dfs_extend(path, seen)
        
        else:
            # For larger components, use heuristic + local search
            # Multiple restarts with different heuristics
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Choose start vertex
                if attempt == 0:
                    start = comp[0]
                elif attempt == 1:
                    # Pick vertex with max degree
                    start = max(comp, key=lambda v: len(adj[v]))
                else:
                    start = random.choice(comp)
                
                path = [start]
                seen = {start}
                
                # Greedy extension with Warnsdorff + randomization
                while True:
                    last = path[-1]
                    unvisited = [v for v in adj[last] if v in comp_set and v not in seen]
                    if not unvisited:
                        break
                    
                    if attempt < 10:
                        # Pure Warnsdorff
                        def score(v):
                            onward = len([w for w in adj[v] if w in comp_set and w not in seen and w != last])
                            return (onward, v)
                        v = min(unvisited, key=score)
                    elif attempt < 20:
                        # Max degree tie-break
                        def score(v):
                            onward = len([w for w in adj[v] if w in comp_set and w not in seen and w != last])
                            return (onward, -len(adj[v]), v)
                        v = min(unvisited, key=score)
                    else:
                        # Random
                        v = random.choice(unvisited)
                    
                    path.append(v)
                    seen.add(v)
                
                if len(path) > len(best_path):
                    best_path = path[:]
                
                # Local search: try to insert skipped vertices
                # Try 2-opt style improvements
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    improved = False
                    path_set = set(path)
                    # Try inserting a vertex between consecutive pairs
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        # Find a vertex w not in path that connects to both u and v
                        candidates = [w for w in comp if w not in path_set and w in adj[u] and w in adj[v]]
                        if candidates:
                            # Insert the one with best position
                            w = candidates[0]
                            new_path = path[:i+1] + [w] + path[i+1:]
                            # Verify validity
                            valid = True
                            for j in range(len(new_path) - 1):
                                if new_path[j+1] not in adj[new_path[j]]:
                                    valid = False
                                    break
                            if valid and len(set(new_path)) == len(new_path):
                                path = new_path
                                path_set = set(path)
                                improved = True
                                if len(path) > len(best_path):
                                    best_path = path[:]
                                break
                    if not improved:
                        break
    
    # Final: ensure best_path is valid
    if not best_path:
        # Fallback: return a single vertex
        return [0]
    
    # Verify the path
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in adj[best_path[i]]:
            # Truncate to valid prefix
            best_path = best_path[:i+1]
            break
    
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
