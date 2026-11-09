"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    best_len = 0
    
    rng = random.Random(748)
    start_time = time.time()
    max_time = 55
    
    def generate_initial_path():
        # Randomized greedy: pick random start, then random neighbor
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        while True:
            last = path[-1]
            candidates = [v for v in adj[last] if v not in seen]
            if not candidates:
                break
            next_v = rng.choice(candidates)
            path.append(next_v)
            seen.add(next_v)
        return path

    def improve_path(path):
        # Local search: try to extend or insert
        improved = True
        while improved:
            improved = False
            seen = set(path)
            unvisited = [v for v in range(n) if v not in seen]
            
            # 1. Try to extend from ends
            # Extend from start
            if unvisited:
                for v in adj[path[0]]:
                    if v in unvisited:
                        path.insert(0, v)
                        seen.add(v)
                        unvisited.remove(v)
                        improved = True
                        break
            # Extend from end
            if improved:
                continue
            if unvisited:
                for v in adj[path[-1]]:
                    if v in unvisited:
                        path.append(v)
                        seen.add(v)
                        unvisited.remove(v)
                        improved = True
                        break
            
            # 2. Try to insert unvisited vertices into the path
            if improved:
                continue
            
            if unvisited:
                # For each unvisited vertex, check if it can be inserted
                # This is O(L * U) where L is path length, U is unvisited count
                # We can optimize by checking edges
                for w in unvisited:
                    # Check if w can be inserted between u and v
                    # u is path[i], v is path[i+1]
                    # Need edge (u, w) and (w, v)
                    # We can precompute neighbors of w
                    neighbors_w = adj[w]
                    inserted = False
                    for i in range(len(path) - 1):
                        u = path[i]
                        v = path[i + 1]
                        if u in neighbors_w and v in neighbors_w:
                            # Insert w at i+1
                            path.insert(i + 1, w)
                            seen.add(w)
                            unvisited.remove(w)
                            improved = True
                            inserted = True
                            break
                    if inserted:
                        break
                        
        return path

    # Main loop
    for _ in range(10000): # Max restarts
        if time.time() - start_time > max_time:
            break
            
        p = generate_initial_path()
        p = improve_path(p)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
            if best_len == n:
                return best_path
                
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
