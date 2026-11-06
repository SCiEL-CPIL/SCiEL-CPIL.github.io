import time
import random
import math

N = 20
C = 20
DEADLINE = 145.0

def construct():
    start_time = time.time()
    
    # Data structures
    adj = [[None] * N for _ in range(N)]
    neigh = [set() for _ in range(N)]
    edges = []
    
    def add_edge(u, v, c):
        adj[u][v] = c
        adj[v][u] = c
        neigh[u].add(v)
        neigh[v].add(u)
        edges.append([u, v, c])

    def remove_edge(u, v):
        c = adj[u][v]
        adj[u][v] = None
        adj[v][u] = None
        neigh[u].discard(v)
        neigh[v].discard(u)
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    def change_color(u, v, new_c):
        old_c = adj[u][v]
        adj[u][v] = new_c
        adj[v][u] = new_c
        for e in edges:
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                e[2] = new_c
                break

    def is_valid_add(u, v, c):
        for x in neigh[v]:
            if x == u: continue
            cx_v = adj[v][x]
            for y in neigh[u]:
                if y == v or y == x: continue
                if adj[x][y] is None: continue
                cy_u = adj[y][u]
                cxy = adj[x][y]
                if c == cx_v or c == cxy or c == cy_u:
                    return False
                if cx_v == cxy or cx_v == cy_u:
                    return False
                if cxy == cy_u:
                    return False
        return True

    def is_valid_recolor(u, v, new_c):
        for x in neigh[v]:
            if x == u: continue
            cx_v = adj[v][x]
            for y in neigh[u]:
                if y == v or y == x: continue
                if adj[x][y] is None: continue
                cy_u = adj[y][u]
                cxy = adj[x][y]
                if new_c == cx_v or new_c == cxy or new_c == cy_u:
                    return False
                if cx_v == cxy or cx_v == cy_u:
                    return False
                if cxy == cy_u:
                    return False
        return True

    # Initialize
    rng = random.Random(42)
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() - start_time > DEADLINE - 10: break
        colors = list(range(C))
        rng.shuffle(colors)
        for c in colors:
            if is_valid_add(u, v, c):
                add_edge(u, v, c)
                break

    best_edges_count = len(edges)
    best_state = [e[:] for e in edges]
    
    def restore_state(state_edges):
        nonlocal adj, neigh, edges
        adj = [[None] * N for _ in range(N)]
        neigh = [set() for _ in range(N)]
        edges = []
        for u, v, c in state_edges:
            add_edge(u, v, c)

    T = 100.0
    T_min = 0.1
    cooling = 0.995
    no_improve_count = 0
    
    while time.time() - start_time < DEADLINE - 2:
        move_type = rng.random()
        
        if move_type < 0.6:
            # Add
            attempts = 0
            while attempts < 5:
                u = rng.randint(0, N - 1)
                v = rng.randint(0, N - 1)
                if u == v: continue
                if u > v: u, v = v, u
                if adj[u][v] is None:
                    c = rng.randint(0, C - 1)
                    if is_valid_add(u, v, c):
                        add_edge(u, v, c)
                        if len(edges) > best_edges_count:
                            best_edges_count = len(edges)
                            best_state = [e[:] for e in edges]
                            no_improve_count = 0
                        break
                    else:
                        no_improve_count += 1
                        break
                attempts += 1
        elif move_type < 0.85:
            # Recolor
            if edges:
                e = rng.choice(edges)
                u, v, old_c = e
                new_c = rng.randint(0, C - 1)
                if new_c != old_c and is_valid_recolor(u, v, new_c):
                    change_color(u, v, new_c)
                    no_improve_count += 1
        else:
            # Remove
            if edges:
                e = rng.choice(edges)
                u, v, c = e
                remove_edge(u, v)
                no_improve_count += 1

        # Kick if stuck
        if no_improve_count > 50:
            # Remove 3 random edges
            for _ in range(3):
                if edges:
                    e = rng.choice(edges)
                    remove_edge(e[0], e[1])
            # Try to add back
            for _ in range(5):
                u = rng.randint(0, N - 1)
                v = rng.randint(0, N - 1)
                if u == v: continue
                if u > v: u, v = v, u
                if adj[u][v] is None:
                    c = rng.randint(0, C - 1)
                    if is_valid_add(u, v, c):
                        add_edge(u, v, c)
            no_improve_count = 0
            T = max(T, 50.0) # Reheat

        T *= cooling
        if T < T_min:
            T = T_min

    if best_state:
        restore_state(best_state)
        
    return edges

if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
