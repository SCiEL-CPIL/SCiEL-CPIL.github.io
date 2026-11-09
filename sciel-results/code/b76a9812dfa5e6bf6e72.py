import time
import random
import math
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

def construct():
    """
    Maximize edges in a K2,3-free graph on 20 vertices.
    
    STRUCTURAL CHANGE: Instead of random greedy restarts + local search,
    use a simulated annealing approach with a carefully designed cooling schedule,
    combined with a fast vectorized validity check. The key insight is that
    K2,3-free graphs have a specific structure that can be exploited by
    starting from a known good configuration (e.g., a union of cliques or a
    regular graph) and then using SA to escape local optima.
    
    Additionally, we use a 'pair-level' representation where we track the
    number of common neighbors for every pair of vertices, enabling O(1)
    validity checks for edge additions/removals.
    """
    n = 20
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Bitmask of all vertices
    all_mask = (1 << n) - 1
    
    # Precompute bit positions
    def bit_count(x):
        return bin(x).count('1')
    
    def count_edges(adj):
        """Count edges from adjacency bitmask list."""
        return sum(bit_count(x) for x in adj) // 2
    
    def can_add_edge(adj, u, v, common_neighbors):
        """
        Check if adding edge (u,v) is valid.
        common_neighbors[i][j] = number of common neighbors of i and j.
        Returns True if valid, False otherwise.
        """
        if adj[u] & (1 << v):
            return True  # Edge already exists
        
        # Check 1: common neighbors of u and v <= 2
        if common_neighbors[u][v] > 2:
            return False
        
        # Check 2: for each w in N(u) \ {v}, common neighbors of w and v must be <= 1
        neighbors_u = adj[u] & ~(1 << v)
        temp = neighbors_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if common_neighbors[w][v] > 1:
                return False
        
        # Check 3: for each w in N(v) \ {u}, common neighbors of w and u must be <= 1
        neighbors_v = adj[v] & ~(1 << u)
        temp = neighbors_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if common_neighbors[w][u] > 1:
                return False
        
        return True
    
    def update_common_neighbors_on_add(adj, u, v, common_neighbors):
        """Update common_neighbors after adding edge (u,v)."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # For each w in N(u), common_neighbors[w][v] increases by 1 (if w != v)
        temp = adj[u] & ~(1 << v)
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            common_neighbors[w][v] += 1
            common_neighbors[v][w] += 1
        # For each w in N(v), common_neighbors[w][u] increases by 1 (if w != u)
        temp = adj[v] & ~(1 << u)
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            common_neighbors[w][u] += 1
            common_neighbors[u][w] += 1
    
    def update_common_neighbors_on_remove(adj, u, v, common_neighbors):
        """Update common_neighbors after removing edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # For each w in N(u), common_neighbors[w][v] decreases by 1 (if w != v)
        temp = adj[u] & ~(1 << v)
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            common_neighbors[w][v] -= 1
            common_neighbors[v][w] -= 1
        # For each w in N(v), common_neighbors[w][u] decreases by 1 (if w != u)
        temp = adj[v] & ~(1 << u)
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            common_neighbors[w][u] -= 1
            common_neighbors[u][w] -= 1
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                temp -= lsb
        return edges
    
    def build_common_neighbors(adj):
        """Build the common_neighbors matrix from adj."""
        common_neighbors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                cn = bit_count(adj[i] & adj[j])
                common_neighbors[i][j] = cn
                common_neighbors[j][i] = cn
        return common_neighbors
    
    # --- Initial construction: start with a structured graph ---
    # Strategy: Start with a 4-regular graph or union of cliques, then use SA
    # A good starting point: 5 disjoint K4s gives 5*6=30 edges, but we can do better
    # Let's start with a random 3-regular graph and grow it
    
    best_adj = None
    best_edges = 0
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # --- Simulated Annealing ---
    def simulated_annealing(initial_adj, max_iterations, deadline, T_start=10.0, T_end=0.1):
        """
        Simulated annealing to maximize edges in a K2,3-free graph.
        """
        adj = initial_adj[:]
        common_neighbors = build_common_neighbors(adj)
        current_edges = count_edges(adj)
        best_adj_local = adj[:]
        best_edges_local = current_edges
        
        # Cooling schedule
        alpha = (T_end / T_start) ** (1.0 / max(1, max_iterations))
        T = T_start
        
        # Precompute neighbors list for faster iteration
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            T *= alpha
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random non-edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                
                if can_add_edge(adj, u, v, common_neighbors):
                    update_common_neighbors_on_add(adj, u, v, common_neighbors)
                    current_edges += 1
                    if current_edges > best_edges_local:
                        best_edges_local = current_edges
                        best_adj_local = adj[:]
                else:
                    # Reject (can't add)
                    pass
            elif move_type < 0.7:
                # Try to remove a random edge (to enable future additions)
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if not (adj[u] & (1 << v)):
                    continue
                
                # Remove edge
                update_common_neighbors_on_remove(adj, u, v, common_neighbors)
                current_edges -= 1
                
                # With some probability, try to add a different edge
                if random.random() < 0.5:
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if adj[u2] & (1 << v2):
                        continue
                    if can_add_edge(adj, u2, v2, common_neighbors):
                        update_common_neighbors_on_add(adj, u2, v2, common_neighbors)
                        current_edges += 1
            else:
                # Swap: remove one edge, add another
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 == v1:
                    continue
                if not (adj[u1] & (1 << v1)):
                    continue
                
                # Try to find an edge to add
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                if adj[u2] & (1 << v2):
                    continue
                
                # Temporarily remove (u1,v1)
                update_common_neighbors_on_remove(adj, u1, v1, common_neighbors)
                current_edges -= 1
                
                if can_add_edge(adj, u2, v2, common_neighbors):
                    update_common_neighbors_on_add(adj, u2, v2, common_neighbors)
                    current_edges += 1
                    if current_edges > best_edges_local:
                        best_edges_local = current_edges
                        best_adj_local = adj[:]
                else:
                    # Revert: re-add (u1,v1)
                    update_common_neighbors_on_add(adj, u1, v1, common_neighbors)
                    current_edges += 1
        
        return best_adj_local, best_edges_local
    
    # --- Strategy 1: Start from multiple initial configurations and run SA ---
    deadline = start_time + time_limit
    
    # Initial configuration 1: 5 disjoint K4s
    adj_init1 = [0] * n
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                adj_init1[base + a] |= (1 << (base + b))
                adj_init1[base + b] |= (1 << (base + a))
    
    # Initial configuration 2: Random 3-regular graph (approximate)
    adj_init2 = [0] * n
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if bit_count(adj_init2[u]) < 3 and bit_count(adj_init2[v]) < 3:
            if can_add_edge(adj_init2, u, v, build_common_neighbors(adj_init2)):
                adj_init2[u] |= (1 << v)
                adj_init2[v] |= (1 << u)
    
    # Initial configuration 3: Complete bipartite K10,10 minus some edges (too dense, will be invalid)
    # Instead, use a 4-regular random graph
    adj_init3 = [0] * n
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if bit_count(adj_init3[u]) < 4 and bit_count(adj_init3[v]) < 4:
            cn = build_common_neighbors(adj_init3)
            if can_add_edge(adj_init3, u, v, cn):
                adj_init3[u] |= (1 << v)
                adj_init3[v] |= (1 << u)
    
    initial_configs = [adj_init1, adj_init2, adj_init3]
    
    for init_adj in initial_configs:
        if time.time() > deadline - 10:
            break
        
        # Greedy fill first
        cn = build_common_neighbors(init_adj)
        for u, v in all_pairs:
            if time.time() > deadline - 10:
                break
            if not (init_adj[u] & (1 << v)):
                if can_add_edge(init_adj, u, v, cn):
                    update_common_neighbors_on_add(init_adj, u, v, cn)
        
        curr_edges = count_edges(init_adj)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = init_adj[:]
        
        # Then SA
        if time.time() > deadline - 20:
            break
        
        sa_adj, sa_edges = simulated_annealing(init_adj, max_iterations=50000, deadline=deadline)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj
    
    # --- Strategy 2: Random restarts with SA ---
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        
        adj = [0] * n
        cn = build_common_neighbors(adj)
        
        # Random sparse initialization
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if time.time() > deadline - 15:
                break
            if random.random() < 0.15:
                if can_add_edge(adj, u, v, cn):
                    update_common_neighbors_on_add(adj, u, v, cn)
        
        # Greedy fill
        for u, v in all_pairs:
            if time.time() > deadline - 15:
                break
            if not (adj[u] & (1 << v)):
                if can_add_edge(adj, u, v, cn):
                    update_common_neighbors_on_add(adj, u, v, cn)
        
        curr_edges = count_edges(adj)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj[:]
        
        # SA
        if time.time() > deadline - 10:
            break
        
        sa_adj, sa_edges = simulated_annealing(adj, max_iterations=20000, deadline=deadline)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj
    
    # --- Final greedy pass ---
    if best_adj is not None and time.time() < deadline:
        cn = build_common_neighbors(best_adj)
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if not (best_adj[u] & (1 << v)):
                if can_add_edge(best_adj, u, v, cn):
                    update_common_neighbors_on_add(best_adj, u, v, cn)
    
    if best_adj is None:
        # Fallback: 5 disjoint K4s
        best_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    best_adj[base + a] |= (1 << (base + b))
                    best_adj[base + b] |= (1 << (base + a))
    
    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
