import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output

    def check_graph(n, edges_set):
        # Returns (True, None) if valid, (False, "K4" or "Alpha6") otherwise
        adj = [[False] * n for _ in range(n)]
        for u, v in edges_set:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check K4
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    # Triangle (i,j,k) found. Check for 4th vertex
                    for x in range(k+1, n):
                        if adj[i][x] and adj[j][x] and adj[k][x]:
                            return False, "K4"
        
        # Check Alpha6
        for combo in itertools.combinations(range(n), 6):
            is_independent = True
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False, "Alpha6"
                
        return True, None

    def find_k4(n, adj):
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for x in range(k+1, n):
                        if adj[i][x] and adj[j][x] and adj[k][x]:
                            return (i, j, k, x)
        return None

    def find_alpha6(n, adj):
        for combo in itertools.combinations(range(n), 6):
            is_independent = True
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return combo
        return None

    # Fallback: Paley(17)
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17

    def local_search(n, edges_set):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges_set:
            adj[u][v] = True
            adj[v][u] = True
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Check for K4
            k4 = find_k4(n, adj)
            if k4:
                # Remove a random edge from the K4
                edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                               (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                u, v = random.choice(edges_in_k4)
                if u > v:
                    u, v = v, u
                if (u, v) in edges_set:
                    edges_set.remove((u, v))
                    adj[u][v] = False
                    adj[v][u] = False
                continue
            
            # Check for Alpha6
            alpha6 = find_alpha6(n, adj)
            if alpha6:
                # Add a random edge within the alpha6 set
                pairs = list(itertools.combinations(alpha6, 2))
                u, v = random.choice(pairs)
                if not adj[u][v]:
                    edges_set.add((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
                continue
            
            # No violations found
            return True
        
        return False

    for target_n in [35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline - 10.0:
            break
            
        found = False
        attempts = 0
        max_attempts = 50
        
        while not found and attempts < max_attempts and time.time() < deadline - 10.0:
            attempts += 1
            
            # Initialize random graph with edge probability ~0.5
            edges_set = set()
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.5:
                        edges_set.add((i, j))
            
            if local_search(target_n, edges_set):
                found = True
                best_n = target_n
                best_edges = [[u, v] for u, v in sorted(edges_set)]
                break
        
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
