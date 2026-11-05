import random
import time
import math

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count()
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(u, v):
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[w]
            if not (mask_w >> v) & 1:
                continue
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mask_z = adj[z]
                if (mask_z >> v) & 1:
                    common = adj[u] & mask_w & mask_z
                    if popcount(common) >= 2:
                        return False
                if (mask_w >> u) & 1 and (mask_z >> u) & 1:
                    common = adj[v] & mask_w & mask_z
                    if popcount(common) >= 2:
                        return False
        return True
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2
    
    def generate_petersen_double_cover():
        """
        Create bipartite double cover of Petersen graph.
        Petersen: 10 vertices, 3-regular, SRG(10,3,0,1)
        Double cover: 20 vertices (two copies of 10), bipartite.
        Vertex i in part A connects to j in part B if i~j in Petersen.
        """
        # Petersen graph edges (standard labeling)
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer 5-cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner 5-cycle
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
        ]
        
        # Part A: vertices 0-9, Part B: vertices 10-19
        # Vertex i in A connects to vertex 10+j in B if (i,j) is Petersen edge
        adj = [0] * 20
        for i, j in petersen_edges:
            # i in A, j in B (vertex 10+j)
            adj[i] |= (1 << (10 + j))
            adj[10 + j] |= (1 << i)
            # Also add the reverse for symmetry in double cover
            # Actually in double cover, we need both directions properly
            # Let me reconsider: double cover of G has V' = V x {0,1}
            # Edge between (u,0) and (v,1) iff (u,v) in E(G)
            # So we need: for each Petersen edge (i,j), connect i (in A) to 10+j (in B)
            # AND connect j (in A) to 10+i (in B)
            adj[j] |= (1 << (10 + i))
            adj[10 + i] |= (1 << j)
        
        return adj[:]
    
    def generate_random_bipartite():
        """Generate random bipartite graph with controlled density"""
        adj = [0] * 20
        # Part A: 0-9, Part B: 10-19
        density = 0.4 + random.random() * 0.3
        for i in range(10):
            for j in range(10):
                if random.random() < density:
                    adj[i] |= (1 << (10 + j))
                    adj[10 + j] |= (1 << i)
        return adj[:]
    
    def greedy_construct(initial_adj=None):
        nonlocal adj
        if initial_adj is not None:
            adj = initial_adj[:]
        else:
            adj = [0] * n
        
        # Polish: try to add any missing edge that is valid
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_addition(u, v):
                        add_edge(u, v)
                        improved = True
        
        return edge_count()
    
    def simulate_annealing(initial_adj, start_temp, end_temp, steps_per_restart):
        nonlocal adj
        adj = initial_adj[:]
        current_count = edge_count()
        best_adj = adj[:]
        best_count = current_count
        
        # Precompute some structures for faster evaluation
        # For each pair (w,z), track how many common neighbors they have with u/v
        
        for step in range(steps_per_restart):
            temp = start_temp * (end_temp / start_temp) ** (step / steps_per_restart)
            
            # Choose random move: add or remove edge
            if random.random() < 0.5:
                # Try to add a random missing edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                if not is_valid_addition(u, v):
                    continue
                add_edge(u, v)
                new_count = current_count + 1
                delta = 1
            else:
                # Try to remove a random existing edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if not ((adj[u] >> v) & 1):
                    continue
                remove_edge(u, v)
                new_count = current_count - 1
                delta = -1
            
            # Accept if improvement, or with probability based on temperature
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Reject: undo the move
                if delta > 0:
                    remove_edge(u, v)
                else:
                    add_edge(u, v)
        
        return best_adj, best_count
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    # Strategy 1: Petersen double cover + polish
    petersen_adj = generate_petersen_double_cover()
    count = greedy_construct(petersen_adj)
    if count > best_count:
        best_count = count
        best_adj = adj[:]
    
    # Strategy 2: Random bipartite + polish
    for _ in range(3):
        if time.time() > deadline:
            break
        rand_adj = generate_random_bipartite()
        count = greedy_construct(rand_adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < deadline - 20:
        sa_adj, sa_count = simulate_annealing(
            best_adj, 
            start_temp=10.0, 
            end_temp=0.1, 
            steps_per_restart=5000
        )
        if sa_count > best_count:
            best_count = sa_count
            best_adj = sa_adj[:]
    
    # Final polish on best solution
    if time.time() < deadline - 5:
        adj = best_adj[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_addition(u, v):
                        add_edge(u, v)
                        improved = True
        final_count = edge_count()
        if final_count > best_count:
            best_count = final_count
            best_adj = adj[:]
    
    # Reconstruct edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
