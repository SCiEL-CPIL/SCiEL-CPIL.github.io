import random
import math
from collections import deque

def construct():
    n = 100
    edges = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find the largest connected component
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # We focus on the largest component as it's the most likely to contain the longest path
    # If there are multiple large components, we should probably check the top few, 
    # but for G(100, 0.018), one giant component is typical.
    components.sort(key=len, reverse=True)
    largest_comp = components[0]
    comp_set = set(largest_comp)
    comp_adj = {v: adj[v] & comp_set for v in largest_comp}
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iterations = 50000
    
    # Initial State: Greedy path
    # Start from a random node, pick neighbor with highest degree
    def get_initial_path():
        if not largest_comp:
            return []
        start = random.choice(largest_comp)
        path = [start]
        seen = {start}
        while True:
            last = path[-1]
            neighbors = [v for v in comp_adj[last] if v not in seen]
            if not neighbors:
                # Try extending from the front?
                # For simplicity, just stop.
                break
            # Heuristic: pick neighbor with highest degree in unvisited
            best_next = max(neighbors, key=lambda x: len(comp_adj[x] - seen))
            path.append(best_next)
            seen.add(best_next)
        return path
    
    current_path = get_initial_path()
    best_path = current_path[:]
    current_len = len(current_path)
    
    # Precompute degrees for heuristic
    deg = {v: len(comp_adj[v]) for v in largest_comp}
    
    # Helper to check if path is valid (it should be by construction, but for safety)
    def is_valid_path(p):
        if len(p) < 2:
            return True
        for i in range(len(p)-1):
            if p[i+1] not in comp_adj[p[i]]:
                return False
        return True
        
    # Neighborhood moves:
    # 1. Insert a vertex v into the path at position i
    #    Requires p[i-1]-v and v-p[i] to be edges.
    # 2. Remove a vertex v from the path
    #    Requires p[i-1]-p[i+1] to be an edge (or i is at end)
    
    T = T_start
    for it in range(max_iterations):
        # Generate a neighbor
        move_type = random.random()
        new_path = current_path[:]
        accepted = False
        
        if move_type < 0.7:
            # Try to insert
            # Pick a random position i (0 to len(new_path))
            # and a random unvisited vertex v
            unvisited = [v for v in largest_comp if v not in set(new_path)]
            if unvisited:
                v = random.choice(unvisited)
                i = random.randint(0, len(new_path))
                
                # Check if v can be inserted at i
                valid_insert = True
                if i > 0 and new_path[i-1] not in comp_adj[v]:
                    valid_insert = False
                if i < len(new_path) and new_path[i] not in comp_adj[v]:
                    valid_insert = False
                
                if valid_insert:
                    new_path.insert(i, v)
                    accepted = True
                    delta = 1 # Length increases by 1
                else:
                    # Cannot insert, try another? Or reject?
                    # Rejecting is fine, but might slow down.
                    # Let's just reject.
                    pass
        else:
            # Try to remove
            if len(new_path) > 1:
                i = random.randint(0, len(new_path)-1)
                v = new_path[i]
                
                # Check if removing v keeps path connected
                # If i==0: need new_path[0]-new_path[1] (which is v-next, so next is new_path[1])
                # Wait, if we remove i, the neighbors are i-1 and i+1.
                # We need new_path[i-1] connected to new_path[i+1]
                if i == 0:
                    # No left neighbor, so no constraint on left. 
                    # But we need to ensure the rest is a path.
                    # Removing head is always valid for path structure.
                    new_path.pop(0)
                    accepted = True
                    delta = -1
                elif i == len(new_path)-1:
                    new_path.pop()
                    accepted = True
                    delta = -1
                else:
                    left = new_path[i-1]
                    right = new_path[i+1]
                    if left in comp_adj[right]:
                        new_path.pop(i)
                        accepted = True
                        delta = -1
                    else:
                        pass # Cannot remove
        
        if not accepted:
            # No valid move generated, or move was invalid
            # Cool down
            T *= cooling_rate
            continue
            
        # Metropolis Acceptance
        # Energy is -length, so delta_energy = -delta_length
        # If delta > 0 (length increases), always accept.
        # If delta < 0 (length decreases), accept with prob exp(-delta/T)
        
        if delta > 0:
            current_path = new_path
            if len(current_path) > len(best_path):
                best_path = current_path[:]
        else:
            # delta is -1
            prob = math.exp(1.0 / T) # exp(-(-1)/T) = exp(1/T)
            if random.random() < prob:
                current_path = new_path
        
        T *= cooling_rate
        
        # If we get stuck for a while, restart?
        # For now, just cool down.
        
    return best_path
