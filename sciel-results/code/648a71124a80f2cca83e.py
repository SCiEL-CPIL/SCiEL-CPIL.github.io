"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 85.0
    
    def check_proper(color):
        """Check if coloring is proper (adjacent vertices have different colors)."""
        for u, v in edges:
            if color[u] == color[v]:
                return False
        return True
    
    def count_bicolor_cycles(color, k):
        """Count total bicolored cycles in the coloring using k colors."""
        total_cycles = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                # Find vertices with color c1 or c2
                verts = set()
                for i in range(n):
                    if color[i] == c1 or color[i] == c2:
                        verts.add(i)
                
                if len(verts) < 3:
                    continue
                
                # DFS to count cycles in the induced subgraph
                visited = set()
                cycle_count = 0
                for start in verts:
                    if start in visited:
                        continue
                    # BFS to find connected component and count edges
                    comp = set()
                    queue = deque([start])
                    visited.add(start)
                    edge_count = 0
                    while queue:
                        curr = queue.popleft()
                        comp.add(curr)
                        for nxt in adj[curr]:
                            if color[nxt] == c1 or color[nxt] == c2:
                                edge_count += 1
                                if nxt not in visited:
                                    visited.add(nxt)
                                    queue.append(nxt)
                    edge_count //= 2  # Each edge counted twice
                    # A connected component with V vertices and E edges has E - V + 1 cycles
                    if len(comp) > 0:
                        cycles = edge_count - len(comp) + 1
                        if cycles > 0:
                            total_cycles += cycles
        return total_cycles
    
    def is_fully_valid(color, k):
        """Check if coloring is proper and has no bicolored cycles."""
        if not check_proper(color):
            return False
        return count_bicolor_cycles(color, k) == 0
    
    def greedy_color(k, order=None):
        """Greedy coloring with MRV heuristic."""
        if order is None:
            # Order by degree descending, then by index
            order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        
        color = [-1] * n
        
        def can_assign(v, c):
            # Check properness
            for u in adj[v]:
                if color[u] == c:
                    return False
            
            # Check bicolored cycle constraint incrementally
            for d in range(k):
                if d == c:
                    continue
                # Find neighbors of v with color d
                neighbors_d = [u for u in adj[v] if color[u] == d]
                if len(neighbors_d) < 2:
                    continue
                
                # Check if any two neighbors in neighbors_d are connected
                # in the {c, d} subgraph (considering only assigned vertices)
                start = neighbors_d[0]
                visited = {start}
                queue = deque([start])
                found_path = False
                
                while queue and not found_path:
                    curr = queue.popleft()
                    for nxt in adj[curr]:
                        if color[nxt] == c or color[nxt] == d:
                            if nxt != start and nxt in neighbors_d[1:]:
                                found_path = True
                                break
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append(nxt)
                
                if found_path:
                    return False
            
            return True
        
        for v in order:
            assigned = False
            for c in range(k):
                if can_assign(v, c):
                    color[v] = c
                    assigned = True
                    break
            
            if not assigned:
                return None
        
        return color
    
    def local_search(initial_color, k, max_iter=5000):
        """Local search to fix bicolored cycles by recoloring vertices."""
        color = initial_color[:]
        best_color = color[:]
        best_cost = count_bicolor_cycles(color, k)
        
        if not check_proper(color):
            # Fix properness first
            for v in range(n):
                used = set(color[u] for u in adj[v] if color[u] != color[v])
                for c in range(k):
                    if c not in used:
                        color[v] = c
                        break
            best_color = color[:]
            best_cost = count_bicolor_cycles(color, k)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            if best_cost == 0:
                break
            
            # Pick a random vertex to try recoloring
            v = random.randint(0, n - 1)
            old_c = color[v]
            
            # Try each color
            best_new_c = -1
            best_new_cost = best_cost
            
            for c in range(k):
                if c == old_c:
                    continue
                
                # Check if c is valid (not used by neighbors)
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Temporarily assign and check
                color[v] = c
                new_cost = count_bicolor_cycles(color, k)
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_c = c
                
                color[v] = old_c
            
            if best_new_c != -1:
                color[v] = best_new_c
                if best_new_cost < best_cost:
                    best_cost = best_new_cost
                    best_color = color[:]
            else:
                # No improvement, try random perturbation
                if random.random() < 0.3:
                    v = random.randint(0, n - 1)
                    old_c = color[v]
                    valid_colors = [c for c in range(k) if all(color[u] != c for u in adj[v])]
                    if valid_colors:
                        color[v] = random.choice(valid_colors)
        
        return best_color, best_cost
    
    def try_k_colors(k):
        """Try to find a valid coloring with k colors."""
        # Multiple random restarts
        best_result = None
        best_num_colors = k + 1
        
        for restart in range(20):
            if time.time() > deadline - 10.0:
                break
            
            # Random initial coloring
            color = [random.randint(0, k - 1) for _ in range(n)]
            
            # Fix properness
            for v in range(n):
                used = set(color[u] for u in adj[v])
                if color[v] in used:
                    for c in range(k):
                        if c not in used:
                            color[v] = c
                            break
            
            if not check_proper(color):
                continue
            
            # Local search
            improved_color, cost = local_search(color, k, max_iter=2000)
            
            if cost == 0:
                # Check if we can reduce colors
                used_colors = set(improved_color)
                num_used = len(used_colors)
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_result = improved_color[:]
                
                # Try to reduce colors
                for target_k in range(num_used - 1, 2, -1):
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Remap colors to 0..target_k-1
                    color_map = {}
                    next_c = 0
                    remapped = []
                    for c in improved_color:
                        if c not in color_map:
                            if next_c < target_k:
                                color_map[c] = next_c
                                next_c += 1
                            else:
                                color_map[c] = random.randint(0, target_k - 1)
                        remapped.append(color_map[c])
                    
                    # Fix properness
                    for v in range(n):
                        used = set(remapped[u] for u in adj[v])
                        if remapped[v] in used:
                            for c in range(target_k):
                                if c not in used:
                                    remapped[v] = c
                                    break
                    
                    if not check_proper(remapped):
                        continue
                    
                    # Local search
                    improved2, cost2 = local_search(remapped, target_k, max_iter=1000)
                    if cost2 == 0:
                        used2 = set(improved2)
                        if len(used2) < best_num_colors:
                            best_num_colors = len(used2)
                            best_result = improved2[:]
                        break
        
        return best_result
    
    # Try from 3 colors upward
    for k in range(3, 12):
        if time.time() > deadline - 15.0:
            break
        
        result = try_k_colors(k)
        if result is not None:
            # Verify
            if is_fully_valid(result, k):
                return [int(x) for x in result]
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
