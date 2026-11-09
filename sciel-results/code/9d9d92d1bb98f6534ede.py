"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def get_triangles(adj):
        tris = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            tris.append((u, v, w))
        return tris
    
    def calc_score(adj):
        degs = get_degrees(adj)
        tris = get_triangles(adj)
        if not tris:
            return 0
        max_sum = max(degs[u] + degs[v] + degs[w] for u, v, w in tris)
        return max_sum
    
    # Start with K_{8,8} + one intra edge
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj = build_adj(edges)
    best_score = calc_score(adj)
    best_adj = [row[:] for row in adj]
    
    # Local search with simulated annealing
    current_adj = [row[:] for row in adj]
    current_score = best_score
    
    def random_swap(adj):
        """Pick a random existing edge to remove and a random non-edge to add."""
        edge_list = []
        non_edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edge_list.append((i, j))
                else:
                    non_edge_list.append((i, j))
        
        if not edge_list or not non_edge_list:
            return None, None
        
        # Try multiple random pairs to avoid creating duplicate edges
        for _ in range(50):
            rem = random.choice(edge_list)
            add = random.choice(non_edge_list)
            if rem == add or (rem[0], rem[1]) == (add[0], add[1]) or (rem[0], rem[1]) == (add[1], add[0]):
                continue
            # Make sure add is not already an edge
            if adj[add[0]][add[1]]:
                continue
            return rem, add
        return None, None
    
    def apply_swap(adj, rem, add):
        new_adj = [row[:] for row in adj]
        new_adj[rem[0]][rem[1]] = 0
        new_adj[rem[1]][rem[0]] = 0
        new_adj[add[0]][add[1]] = 1
        new_adj[add[1]][add[0]] = 1
        return new_adj
    
    # Simulated annealing
    temp = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    iteration = 0
    while time.time() < deadline and temp > min_temp:
        rem, add = random_swap(current_adj)
        if rem is None:
            temp *= cooling
            iteration += 1
            continue
        
        new_adj = apply_swap(current_adj, rem, add)
        new_score = calc_score(new_adj)
        
        delta = new_score - current_score
        if delta <= 0 or random.random() < np.exp(-delta / temp):
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = [row[:] for row in current_adj]
        
        temp *= cooling
        iteration += 1
        
        # Periodic restarts to escape local minima
        if iteration % 2000 == 0 and time.time() < deadline - 30:
            # Try a few random restarts from best
            for _ in range(3):
                restart_adj = [row[:] for row in best_adj]
                for _ in range(50):
                    rem2, add2 = random_swap(restart_adj)
                    if rem2 is None:
                        break
                    restart_adj = apply_swap(restart_adj, rem2, add2)
                rs = calc_score(restart_adj)
                if rs < best_score:
                    best_score = rs
                    best_adj = [row[:] for row in restart_adj]
                    current_adj = [row[:] for row in restart_adj]
                    current_score = best_score
        
        # Also try targeted improvements: move edges to reduce degree concentration
        if iteration % 500 == 0 and time.time() < deadline - 30:
            degs = get_degrees(current_adj)
            # Find high-degree vertices and try to move their edges
            high_deg = sorted(range(n), key=lambda i: -degs[i])[:4]
            for hi in high_deg:
                for _ in range(10):
                    neighbors = [j for j in range(n) if current_adj[hi][j]]
                    if not neighbors:
                        continue
                    nb = random.choice(neighbors)
                    # Find a vertex not adjacent to hi to add edge to
                    non_nb = [j for j in range(n) if j != hi and not current_adj[hi][j]]
                    if not non_nb:
                        continue
                    new_nb = random.choice(non_nb)
                    if new_nb == nb:
                        continue
                    test_adj = [row[:] for row in current_adj]
                    test_adj[hi][nb] = 0
                    test_adj[nb][hi] = 0
                    test_adj[hi][new_nb] = 1
                    test_adj[new_nb][hi] = 1
                    ts = calc_score(test_adj)
                    if ts < current_score:
                        current_adj = test_adj
                        current_score = ts
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = [row[:] for row in current_adj]
    
    # Final polish: try all single-edge swaps from best
    if time.time() < deadline - 10:
        polish_adj = [row[:] for row in best_adj]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if polish_adj[i][j]:
                        for k in range(n):
                            for l in range(k+1, n):
                                if k == i or k == j or l == i or l == j:
                                    continue
                                if not polish_adj[k][l]:
                                    test_adj = [row[:] for row in polish_adj]
                                    test_adj[i][j] = 0
                                    test_adj[j][i] = 0
                                    test_adj[k][l] = 1
                                    test_adj[l][k] = 1
                                    ts = calc_score(test_adj)
                                    if ts < best_score:
                                        best_score = ts
                                        best_adj = test_adj
                                        polish_adj = [row[:] for row in test_adj]
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
    
    # Convert best_adj back to edge list
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                best_edges.append([i, j])
    
    assert len(best_edges) == target_edges, f"Expected {target_edges} edges, got {len(best_edges)}"
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
