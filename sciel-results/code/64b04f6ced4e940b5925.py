"""Construct a graph on 12 vertices with min degree 3 and minimum distinct cycle lengths."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s safety margin
    
    # Fallback: Two K_{3,3} (Raw value 2, Score 2.5)
    fallback_edges = []
    for u in range(3):
        for v in range(3, 6):
            fallback_edges.append([u, v])
    for u in range(6, 9):
        for v in range(9, 12):
            fallback_edges.append([u, v])
    fallback_edges = [[int(a), int(b)] for a, b in fallback_edges]
    
    best_edges = fallback_edges
    best_raw_value = 2
    
    # Search for a graph with Raw Value 1 (only 4-cycles)
    # Bipartite graph on 12 vertices (6+6)
    # Min degree 3
    
    # Heuristic: If we can find a graph where all cycles are 4, raw value is 1.
    # We try random bipartite graphs.
    
    def get_cycle_lengths(edges):
        # Build adjacency list
        adj = {i: set() for i in range(12)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all simple cycles up to length 8 (since min degree 3, girth >= 4, 
        # and N=12, max cycle length is 12. But 4, 6, 8 are likely candidates for small sets)
        # Actually, just find all simple cycles.
        # DFS from each node, tracking path.
        # Since N=12, this is fast.
        
        cycles = set()
        
        def dfs(start, current, path, visited):
            if len(path) > 1 and current == start:
                # Found a cycle
                # Normalize cycle to avoid duplicates
                # Cycle is path[:-1]
                c = tuple(path[:-1])
                # Canonical form: rotate to smallest element, then reverse if needed
                min_idx = c.index(min(c))
                c_rot = c[min_idx:] + c[:min_idx]
                c_rev = c_rot[::-1]
                if c_rot < c_rev:
                    c_norm = c_rot
                else:
                    c_norm = c_rev
                cycles.add(len(c_norm))
                return
                
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    dfs(start, start, path + [start], visited)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
                    
        # To avoid finding the same cycle from different starting points, 
        # we only start DFS from node i if i is the smallest node in the cycle.
        # Or just collect all and deduplicate by length? No, we need distinct lengths.
        # The DFS above adds to a set of lengths? No, I added normalized cycle to a set?
        # I added len(c_norm) to a set? No, I added c_norm to a set?
        # Let's just collect lengths.
        
        lengths = set()
        
        def dfs_len(start, current, path, visited):
            if len(path) > 1 and current == start:
                lengths.add(len(path) - 1)
                return
                
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    dfs_len(start, start, path + [start], visited)
                elif neighbor not in visited:
                    # Pruning: Only explore if neighbor > start? 
                    # No, standard cycle finding is tricky.
                    # Given N=12, we can just do it.
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_len(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
                    
        for i in range(12):
            # Only start from i to avoid duplicates? 
            # Cycles will be found multiple times. 
            # But we only care about the SET of lengths.
            # So we can just run DFS from 0? No, cycles not containing 0.
            # We must run from all nodes.
            # To save time, we can check if the graph is bipartite and 
            # verify no 6-cycles using a specific check.
            pass
            
        # Simpler check for "Only 4-cycles":
        # 1. Graph is bipartite.
        # 2. No 6-cycles.
        # 3. No 8-cycles?
        # If min degree is 3, and no 6-cycles, are there 8-cycles?
        # Yes.
        
        # Let's just enumerate all cycles for N=12. It's fast.
        lengths = set()
        
        def find_cycles(start):
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 4:
                        lengths.add(len(path))
                    elif neighbor not in visited:
                        # Optimization: only visit neighbors > start? 
                        # No, this misses cycles where start is not the min.
                        # But we run find_cycles for all starts.
                        # To avoid duplicates, we can enforce that all nodes in path > start?
                        # This works!
                        if neighbor > start:
                            stack.append((neighbor, path + [neighbor], visited | {neighbor}))
                            
        for i in range(12):
            find_cycles(i)
            
        return lengths

    # Search Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts % 1000 == 0:
            # Optional: check time
            pass
            
        # Generate random bipartite graph
        # Partitions: 0-5 and 6-11
        edges = []
        # Ensure min degree 3
        # Randomly add edges
        # Start with a random regular-ish graph
        # Shuffle potential edges
        potential_edges = [(u, v) for u in range(6) for v in range(6, 12)]
        random.shuffle(potential_edges)
        
        # Greedy addition to satisfy degree
        deg = [0] * 12
        for u, v in potential_edges:
            if deg[u] < 3 and deg[v] < 3:
                edges.append((u, v))
                deg[u] += 1
                deg[v] += 1
            
        # Check if min degree is 3
        if min(deg) < 3:
            # Try to fix by adding more edges? 
            # Or just skip.
            # Better: Generate a random 3-regular bipartite graph directly?
            # Or just retry.
            continue
            
        # Check cycle lengths
        lengths = get_cycle_lengths(edges)
        
        if 1 not in lengths and len(lengths) == 1:
            # Only 4-cycles? Or only 6-cycles?
            # If only 4-cycles, raw value 1.
            # If only 6-cycles, raw value 1.
            # Any single length is raw value 1.
            best_edges = [[int(u), int(v)] for u, v in edges]
            best_raw_value = 1
            break # Found optimal
            
        if len(lengths) < best_raw_value:
            best_edges = [[int(u), int(v)] for u, v in edges]
            best_raw_value = len(lengths)
            
    # Return best found
    # Ensure types are correct
    out = []
    seen = set()
    for u, v in best_edges:
        if u == v:
            continue
        a, b = (min(u, v), max(u, v))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([int(a), int(b)])
        
    return out
