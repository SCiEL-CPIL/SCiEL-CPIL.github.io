import time
import random

def is_3_colorable(adj):
    n = len(adj)
    neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                neighbors[u].add(v)
                
    colors = [-1] * n
    
    def get_available_colors(v):
        used = set()
        for u in neighbors[v]:
            if colors[u] != -1:
                used.add(colors[u])
        return [c for c in range(3) if c not in used]

    def backtrack():
        v = -1
        min_options = 4
        best_options = []
        
        for i in range(n):
            if colors[i] == -1:
                opts = get_available_colors(i)
                if len(opts) < min_options:
                    min_options = len(opts)
                    v = i
                    best_options = opts
                if min_options == 0:
                    return False
                if min_options == 1:
                    break
                    
        if v == -1:
            return True
            
        for c in best_options:
            colors[v] = c
            if backtrack():
                return True
            colors[v] = -1
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = time.time() + 115.0
    
    n = 16
    
    # Grötzsch graph edges
    groetzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (0,6), (1,5), (1,6), (2,5), (2,7), (3,6), (3,8), (4,7), (4,9),
        (5,10), (6,10), (7,10), (8,10), (9,10)
    ]
    
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        
    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
        
    def get_edges_from_adj(adj_matrix):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_matrix[u][v]:
                    edges.append([u, v])
        return edges
        
    def count_edges(adj_matrix):
        return sum(1 for u in range(n) for v in range(u+1, n) if adj_matrix[u][v])
        
    # Initialize
    for u, v in groetzsch_edges:
        add_edge(u, v)
        
    # Check if initial is 4-chromatic
    if is_3_colorable(adj):
        # Fallback: Force a 4-chromatic graph?
        # Grötzsch is 4-chromatic. If this returns True, our is_3_colorable is buggy.
        pass
        
    best_adj = [row[:] for row in adj]
    best_count = count_edges(adj)
    current_count = best_count
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    steps_since_improvement = 0
    
    while time.time() < deadline:
        # Pick random edge
        u, v = random.choice(all_edges)
        
        if adj[u][v]:
            # Remove
            remove_edge(u, v)
            if not is_3_colorable(adj):
                # Still 4-chromatic
                # Only accept removal if it helps? 
                # For now, reject removals unless we are in a "perturb" phase
                # We'll stick to strict improvement for add, and allow removal only if we are stuck
                if steps_since_improvement > 100:
                    current_count -= 1
                    steps_since_improvement = 0
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
        else:
            # Add
            has_triangle = False
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    has_triangle = True
                    break
                    
            if not has_triangle:
                add_edge(u, v)
                if not is_3_colorable(adj):
                    current_count += 1
                    steps_since_improvement = 0
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = [row[:] for row in adj]
                else:
                    remove_edge(u, v)
            else:
                pass
                
        if steps_since_improvement > 500:
            # Reset to best and perturb
            adj = [row[:] for row in best_adj]
            current_count = best_count
            steps_since_improvement = 0
            # Remove a few random edges to allow new additions
            for _ in range(random.randint(1, 3)):
                eu, ev = random.choice(all_edges)
                if adj[eu][ev]:
                    remove_edge(eu, ev)
                    # Check if still 4-chromatic, if not, restore
                    if is_3_colorable(adj):
                        add_edge(eu, ev)
                    else:
                        current_count -= 1

    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-START
import time
import random

def is_3_colorable(adj):
    n = len(adj)
    neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                neighbors[u].add(v)
                
    colors = [-1] * n
    
    def get_available_colors(v):
        used = set()
        for u in neighbors[v]:
            if colors[u] != -1:
                used.add(colors[u])
        return [c for c in range(3) if c not in used]

    def backtrack():
        v = -1
        min_options = 4
        best_options = []
        
        for i in range(n):
            if colors[i] == -1:
                opts = get_available_colors(i)
                if len(opts) < min_options:
                    min_options = len(opts)
                    v = i
                    best_options = opts
                if min_options == 0:
                    return False
                if min_options == 1:
                    break
                    
        if v == -1:
            return True
            
        for c in best_options:
            colors[v] = c
            if backtrack():
                return True
            colors[v] = -1
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = time.time() + 115.0
    
    n = 16
    
    groetzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (0,6), (1,5), (1,6), (2,5), (2,7), (3,6), (3,8), (4,7), (4,9),
        (5,10), (6,10), (7,10), (8,10), (9,10)
    ]
    
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        
    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
        
    def get_edges_from_adj(adj_matrix):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_matrix[u][v]:
                    edges.append([u, v])
        return edges
        
    def count_edges(adj_matrix):
        return sum(1 for u in range(n) for v in range(u+1, n) if adj_matrix[u][v])
        
    for u, v in groetzsch_edges:
        add_edge(u, v)
        
    best_adj = [row[:] for row in adj]
    best_count = count_edges(adj)
    current_count = best_count
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    steps_since_improvement = 0
    
    while time.time() < deadline:
        u, v = random.choice(all_edges)
        
        if adj[u][v]:
            remove_edge(u, v)
            if not is_3_colorable(adj):
                if steps_since_improvement > 100:
                    current_count -= 1
                    steps_since_improvement = 0
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
        else:
            has_triangle = False
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    has_triangle = True
                    break
                    
            if not has_triangle:
                add_edge(u, v)
                if not is_3_colorable(adj):
                    current_count += 1
                    steps_since_improvement = 0
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = [row[:] for row in adj]
                else:
                    remove_edge(u, v)
            else:
                pass
                
        if steps_since_improvement > 500:
            adj = [row[:] for row in best_adj]
            current_count = best_count
            steps_since_improvement = 0
            for _ in range(random.randint(1, 3)):
                eu, ev = random.choice(all_edges)
                if adj[eu][ev]:
                    remove_edge(eu, ev)
                    if is_3_colorable(adj):
                        add_edge(eu, ev)
                    else:
                        current_count -= 1

    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
