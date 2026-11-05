import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # Try n from 18 down to 9
    # 18 is the target if possible, 17 is the known best
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                return result
        except Exception as e:
            # print(f"Error for n={n}: {e}")
            continue
            
    # Fallback to seed n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    # Edge indices
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Initial random coloring
    colors = np.random.randint(0, 4, size=num_edges)
    
    # Adjacency matrices for each color
    # adj[c][u][v] = 1 if edge (u,v) has color c
    adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u][v] = 1
        adj[c][v][u] = 1
        
    def check_c6_creation(u, v, color_idx):
        """
        Check if adding edge (u,v) to color_idx creates a C6.
        This is true if there is a path of length 5 between u and v in G_color_idx.
        """
        # BFS from u in G_color_idx, limit depth 5
        # We don't need to exclude (u,v) because it's not in the graph yet
        visited = [0] * n
        dist = [-1] * n
        dist[u] = 0
        queue = [u]
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            if dist[curr] >= 5:
                continue
            # Check neighbors
            # Using numpy for speed if possible, but n is small
            neighbors = np.nonzero(adj[color_idx][curr])[0]
            for nb in neighbors:
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    if dist[nb] == 5:
                        if nb == v:
                            return True
                        queue.append(nb)
        return False

    def check_c6_destruction(u, v, color_idx):
        """
        Check if removing edge (u,v) from color_idx destroys a C6.
        This is true if there is a path of length 5 between u and v in G_color_idx \ {(u,v)}.
        """
        # BFS from u in G_color_idx, but ignoring edge (u,v)
        visited = [0] * n
        dist = [-1] * n
        dist[u] = 0
        queue = [u]
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            if dist[curr] >= 5:
                continue
            neighbors = np.nonzero(adj[color_idx][curr])[0]
            for nb in neighbors:
                # Skip the edge (u,v) if we are going from u to v or v to u
                # Since we start at u, we only care about the first step u->v
                if curr == u and nb == v:
                    continue
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    if dist[nb] == 5:
                        if nb == v:
                            return True
                        queue.append(nb)
        return False

    def count_c6s():
        """
        Count total C6s in current coloring.
        Used for initial cost and final verification.
        """
        count = 0
        for c in range(4):
            # For each edge (u,v) in color c, check if it's in a C6
            # To avoid double counting, we can just sum over all edges
            # and divide by 6? No, each C6 has 6 edges.
            # If we check for each edge if it's in a C6, each C6 is counted 6 times.
            # So total C6s = sum(checks) / 6.
            # But check_c6_creation is for ADDING.
            # Here we want to know if edge (u,v) is PART of a C6.
            # This is equivalent to: Is there a path of length 5 between u and v in G_c \ {(u,v)}?
            # So we use check_c6_destruction logic.
            for idx, (u, v) in enumerate(edges):
                if colors[idx] == c:
                    if check_c6_destruction(u, v, c):
                        count += 1
        return count // 6

    # Initial Cost
    current_cost = count_c6s()
    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    best_cost = current_cost
    best_colors = colors.copy()
    
    # Local Search / Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute edge neighbors for faster access?
    # Not strictly necessary with n=17
    
    steps = 0
    while time.time() < deadline - 5.0:
        # Pick random edge
        idx = np.random.randint(num_edges)
        u, v = edges[idx]
        c_old = colors[idx]
        
        # Pick new color
        c_new = np.random.randint(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate Delta Cost
        delta_cost = 0
        
        # Check if adding to c_new creates C6
        if check_c6_creation(u, v, c_new):
            delta_cost += 1
            
        # Check if removing from c_old destroys C6
        if check_c6_destruction(u, v, c_old):
            delta_cost -= 1
            
        # Acceptance
        if delta_cost < 0 or (delta_cost == 0 and random.random() < 0.5) or (delta_cost > 0 and random.random() < np.exp(-delta_cost / T)):
            # Apply change
            colors[idx] = c_new
            adj[c_old][u][v] = 0
            adj[c_old][v][u] = 0
            adj[c_new][u][v] = 1
            adj[c_new][v][u] = 1
            
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
                if best_cost == 0:
                    return {'n': n, 'colors': colors.tolist()}
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
        steps += 1

    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    else:
        return None
