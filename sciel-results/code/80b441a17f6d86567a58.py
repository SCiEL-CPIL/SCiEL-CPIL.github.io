import random
import time

def construct():
    N = 20
    NUM_COLORS = 20
    deadline = time.time() + 145  # 5s margin
    
    best_edges = 0
    best_solution = None
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Function to check if adding (u, v, c) is valid
    def can_add(adj, u, v, c):
        # Check all 4-cycles that would include (u, v)
        # Cycle: u - x - y - v - u
        # Edges: (u, x), (x, y), (y, v) must exist
        for x in range(N):
            if x == u or x == v:
                continue
            if adj[u][x] is None:
                continue
            cx = adj[u][x]
            # Optimization: if cx == c, we already have a duplicate color in the cycle?
            # The cycle colors are c, cx, cy, cv.
            # If cx == c, then {c, cx, ...} has size < 4 regardless of cy, cv.
            # So we can return False immediately?
            # Yes, because the set of 4 colors must have size 4.
            # If two are equal, size is at most 3.
            if cx == c:
                return False
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                if adj[x][y] is None:
                    continue
                if adj[y][v] is None:
                    continue
                cy = adj[x][y]
                cv = adj[y][v]
                # Check if c, cx, cy, cv are all distinct
                # We already know c != cx
                if cy == c or cy == cx or cy == cv:
                    return False
                if cv == c or cv == cx:
                    return False
                # If we pass all these, they are distinct
                # Wait, I should just use the set check for safety and clarity, 
                # the manual checks are prone to error if I miss a combination.
                # len({c, cx, cy, cv}) < 4 is robust.
                if len({c, cx, cy, cv}) < 4:
                    return False
        return True

    for restart in range(100000):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        adj = [[None for _ in range(N)] for _ in range(N)]
        edges_to_add = all_edges.copy()
        random.shuffle(edges_to_add)
        
        current_edge_count = 0
        
        for u, v in edges_to_add:
            # Try to find a valid color
            colors = list(range(NUM_COLORS))
            random.shuffle(colors)
            added = False
            for c in colors:
                if can_add(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    current_edge_count += 1
                    added = True
                    break
            # If not added, skip
            
        if current_edge_count > best_edges:
            best_edges = current_edge_count
            # Convert adj to list of lists [u, v, c]
            solution = []
            for u in range(N):
                for v in range(u + 1, N):
                    if adj[u][v] is not None:
                        solution.append([u, v, adj[u][v]])
            best_solution = solution
            # If we found a complete graph, we can't do better
            if best_edges == len(all_edges):
                return best_solution

    if best_solution is None:
        # Fallback: return a valid small graph, e.g., a star
        return [[0, v, 0] for v in range(1, N)]
        
    return best_solution
