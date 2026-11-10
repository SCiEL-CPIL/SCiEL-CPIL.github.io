import random
import itertools
import time
import math

def construct():
    n = 20
    
    # Precompute pairs for fast checking? 
    # We will use bitmasks for adjacency for speed.
    # adj[i] is an integer bitmask of neighbors.
    
    def get_common_neighbors(adj, a, b):
        return adj[a] & adj[b]
        
    def get_common_neighbors3(adj, a, b, c):
        return adj[a] & adj[b] & adj[c]

    def is_valid(adj):
        # Check all triples
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                # For each k > j, check if |common_ij & adj[k]| >= 3
                # This is slightly faster than triple intersection from scratch?
                # Actually, let's just do the triple intersection.
                # n=20 is small.
                pass
        
        # Optimized validity check:
        # Count bad triples
        bad = 0
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                # We need to check if there exists a k such that |common_ij & adj[k]| >= 3
                # Actually, the condition is: for ALL triples (i,j,k), |N(i) cap N(j) cap N(k)| < 3.
                # So we just need to verify this for all triples.
                for k in range(j+1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # Fast local check for adding edge (u, v)
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        
        # Check all pairs (x, y) where x, y != u, v
        # If v is common neighbor of x, y, then M_uxy increases.
        # If u is common neighbor of x, y, then M_vxy increases.
        
        # We need to ensure that for all x, y:
        # 1. If v in N(x) and v in N(y), then current M_uxy < 2 (so new <= 2)
        # 2. If u in N(x) and u in N(y), then current M_vxy < 2 (so new <= 2)
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                
                # Check condition 1: Triple (u, x, y)
                # v is in N(x) and N(y)?
                if ((adj[x] >> v) & 1) and ((adj[y] >> v) & 1):
                    # v is common neighbor of x and y.
                    # So v is in N(x) cap N(y).
                    # Adding u-v makes v in N(u) cap N(x) cap N(y).
                    # So M_uxy increases by 1.
                    # We need current M_uxy <= 1.
                    current_m = (adj[u] & adj[x] & adj[y]).bit_count()
                    if current_m >= 2:
                        return False
                
                # Check condition 2: Triple (v, x, y)
                # u is in N(x) and N(y)?
                if ((adj[x] >> u) & 1) and ((adj[y] >> u) & 1):
                    # u is common neighbor of x and y.
                    # Adding u-v makes u in N(v) cap N(x) cap N(y).
                    # So M_vxy increases by 1.
                    # We need current M_vxy <= 1.
                    current_m = (adj[v] & adj[x] & adj[y]).bit_count()
                    if current_m >= 2:
                        return False
                        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def simulated_annealing(initial_adj, start_temp=1.0, end_temp=0.01, cooling=0.99, steps=20000):
        current_adj = list(initial_adj)
        current_edges = count_edges(current_adj)
        best_adj = list(current_adj)
        best_edges = current_edges
        
        temp = start_temp
        for step in range(steps):
            temp *= cooling
            if temp < end_temp:
                break
            
            # Generate a move: Add or Remove a random edge
            # Bias towards adding edges if we are below a target, or just random
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add an edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1: # Check bit
                    continue
                
                if can_add_edge(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    new_edges = current_edges + 1
                    current_edges = new_edges
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = list(current_adj)
                else:
                    # Can't add, maybe try a swap? Or just skip.
                    pass
            else:
                # Try to remove an edge
                # Pick a random existing edge
                u = random.randint(0, n-1)
                neighbors = [i for i in range(n) if (current_adj[u] >> i) & 1]
                if not neighbors:
                    continue
                v = random.choice(neighbors)
                
                remove_edge(current_adj, u, v)
                new_edges = current_edges - 1
                delta = -1
                if random.random() < math.exp(delta / temp):
                    current_edges = new_edges
                else:
                    add_edge(current_adj, u, v)
        
        return best_adj, best_edges

    # Initialize
    # Start with a sparse random graph and repair? 
    # Or start with a known good structure?
    # Let's try starting with a random graph and running SA from there.
    # To get a good start, we can try to greedily add edges.
    
    def greedy_construct():
        adj = [0] * n
        # Shuffle vertices to break symmetry
        order = list(range(n))
        random.shuffle(order)
        # This greedy might not work well for K3,3, but let's try a random greedy
        for u in order:
            # Try to add edges to all possible v > u
            candidates = [v for v in range(n) if v != u]
            random.shuffle(candidates)
            for v in candidates:
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
        return adj

    best_adj = None
    best_edges = 0
    
    # Try multiple random starts with different seeds
    for trial in range(10):
        random.seed(trial)
        # Start with empty graph and greedy fill
        adj = greedy_construct()
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
            
        # Run SA from this start
        res_adj, res_edges = simulated_annealing(adj, start_temp=2.0, end_temp=0.001, cooling=0.998, steps=50000)
        
        if res_edges > best_edges:
            best_edges = res_edges
            best_adj = res_adj

    if best_adj is None:
        # Fallback: Empty graph
        best_adj = [0] * n
        best_edges = 0

    # Convert to list of edges
    edges_list = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges_list.append([i, j])
            mask ^= lsb
            
    return edges_list
