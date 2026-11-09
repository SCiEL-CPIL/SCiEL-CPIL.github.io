import random
import time

def construct():
    n = 20
    adj = [set() for _ in range(n)]
    common = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        if common[u][v] > 2:
            return False
        
        for y in adj[v]:
            if common[u][y] > 1:
                return False
        
        for y in adj[u]:
            if common[v][y] > 1:
                return False
        
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        for y in adj[v]:
            common[u][y] += 1
            common[y][u] += 1
            
        for y in adj[u]:
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        for y in adj[v]:
            common[u][y] -= 1
            common[y][u] -= 1
        for y in adj[u]:
            common[v][y] -= 1
            common[y][v] -= 1

    def get_edge_count():
        return sum(len(s) for s in adj) // 2

    start_time = time.time()
    time_limit = 110.0
    best_edges = []
    best_count = 0
    
    def save_best():
        nonlocal best_count, best_edges
        curr_count = get_edge_count()
        if curr_count > best_count:
            best_count = curr_count
            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Initial greedy fill
    for _ in range(2000):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if can_add(u, v):
            add_edge(u, v)
    
    save_best()

    current_count = get_edge_count()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            break
            
        # Generate a move
        # 1. Try to add a random edge
        # 2. Try to remove a random edge
        # 3. Try to swap (remove one, add another) - optional, but simple add/remove is enough for SA
        
        move_type = random.random()
        if move_type < 0.7:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or v in adj[u]:
                continue
            if can_add(u, v):
                add_edge(u, v)
                new_count = current_count + 1
                # Always accept improvement
                current_count = new_count
                save_best()
            else:
                # Cannot add, do nothing? Or treat as a failed move.
                # In SA for max edges, we only consider valid moves.
                # If we can't add, we don't change state.
                pass
        else:
            # Remove edge
            if current_count == 0:
                continue
            edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if not edges_list:
                continue
            u, v = random.choice(edges_list)
            remove_edge(u, v)
            new_count = current_count - 1
            # Accept with probability exp(-delta/T) where delta = new - old = -1
            # So prob = exp(1/T) ? No, standard SA: accept if new > old OR with prob exp((new-old)/T)
            # Here new < old. delta = -1.
            # prob = exp(-1/T)
            if random.random() < (2.718281828 ** (-1.0 / T)):
                current_count = new_count
            else:
                # Reject, revert
                add_edge(u, v)
                # current_count remains unchanged
        
        T *= cooling_rate

    # Final polish: greedy add
    for _ in range(500):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if can_add(u, v):
            add_edge(u, v)
    save_best()

    if not best_edges:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Verification
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            common_count = len(adj_check[u] & adj_check[v])
            if common_count > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
