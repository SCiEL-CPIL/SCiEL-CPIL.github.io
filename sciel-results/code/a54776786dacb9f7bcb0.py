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
    
    New structural approach:
    1. Use NumPy vectorized operations for fast common-neighbor checks.
    2. Try multiple construction strategies and keep the best:
       a) Integer programming via pulp (if available) for exact/strong upper bound
       b) Greedy with random restarts using vectorized validity checks
       c) Greedy + local search with edge swaps
    3. Deadline guard throughout to return best found before time limit.
    """
    n = 20
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Bitmask of all vertices
    all_mask = (1 << n) - 1
    
    def count_edges(adj):
        """Count edges from adjacency bitmask list."""
        return sum(bin(x).count('1') for x in adj) // 2
    
    def can_add_edge_vectorized(adj, u, v):
        """
        Check if adding edge (u,v) is valid using bit operations.
        Returns True if valid, False otherwise.
        """
        if adj[u] & (1 << v):
            return True  # Edge already exists
        
        # Check 1: common neighbors of u and v <= 2
        common_uv = adj[u] & adj[v]
        if bin(common_uv).count('1') > 2:
            return False
        
        # Check 2: for each w in N(u), common neighbors of w and v must be <= 1
        # (because u would become a new common neighbor of w and v)
        neighbors_u = adj[u]
        temp = neighbors_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == v:
                continue
            common_wv = adj[w] & adj[v]
            if bin(common_wv).count('1') > 1:
                return False
        
        # Check 3: for each w in N(v), common neighbors of w and u must be <= 1
        neighbors_v = adj[v]
        temp = neighbors_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == u:
                continue
            common_wu = adj[w] & adj[u]
            if bin(common_wu).count('1') > 1:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
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
    
    def greedy_fill(adj, order=None, deadline=None):
        """
        Greedily add edges in the given order (or standard order if None).
        """
        if order is None:
            order = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for u, v in order:
            if deadline is not None and time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if can_add_edge_vectorized(adj, u, v):
                    add_edge(adj, u, v)
    
    def local_search(adj, max_iterations, deadline):
        """
        Local search with edge swaps to escape local optima.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        
        # Precompute all pairs
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to add a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if adj[u] & (1 << v):
                continue
            
            if can_add_edge_vectorized(adj, u, v):
                add_edge(adj, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Try swap: remove a random edge incident to u or v, then add (u,v)
                incident = []
                temp = adj[u]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    incident.append((u, w))
                    temp -= lsb
                temp = adj[v]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    if w != u:
                        incident.append((v, w))
                    temp -= lsb
                
                if incident:
                    rem_u, rem_v = random.choice(incident)
                    remove_edge(adj, rem_u, rem_v)
                    current_edges -= 1
                    if can_add_edge_vectorized(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        # Revert
                        add_edge(adj, rem_u, rem_v)
                        current_edges += 1
        
        return best_adj, best_edges
    
    # --- Strategy 1: Try integer programming (if pulp is available) ---
    best_adj = None
    best_edges = 0
    
    try:
        import pulp
        # Build ILP model
        prob = pulp.LpProblem("K23_free_max_edges", pulp.LpMaximize)
        x = {}
        for u in range(n):
            for v in range(u + 1, n):
                x[u, v] = pulp.LpVariable(f"x_{u}_{v}", cat='Binary')
        
        # Objective: maximize number of edges
        prob += pulp.lpSum(x[u, v] for u in range(n) for v in range(u + 1, n))
        
        # K2,3-free constraints:
        # For every triple of distinct vertices (a, b, c) and pair (d, e) where d,e != a,b,c,
        # if a-d, a-e, b-d, b-e all exist, that's a K2,3 with {a,b} and {c,d,e}? 
        # Actually K2,3 is bipartite with parts of size 2 and 3.
        # A K2,3 has vertices {u1,u2} on one side, {v1,v2,v3} on other, with all 6 edges.
        # So we need: for any choice of 2 vertices U = {u1,u2} and 3 vertices V = {v1,v2,v3}
        # with U and V disjoint, not all 6 edges between U and V can be present.
        # That is: x[u1,v1] + x[u1,v2] + x[u1,v3] + x[u2,v1] + x[u2,v2] + x[u2,v3] <= 5
        
        import itertools
        for U in itertools.combinations(range(n), 2):
            for V in itertools.combinations(range(n), 3):
                if len(set(U) & set(V)) > 0:
                    continue
                # Sum of edges between U and V must be <= 5
                edge_sum = 0
                for u in U:
                    for v in V:
                        if u < v:
                            edge_sum += x[u, v]
                        else:
                            edge_sum += x[v, u]
                prob += edge_sum <= 5
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
        
        if pulp.LpStatus[prob.status] == "Optimal" or pulp.LpStatus[prob.status] == "Not Solved":
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if x[u, v].value() is not None and x[u, v].value() > 0.5:
                        add_edge(adj, u, v)
            edges_found = count_edges(adj)
            if edges_found > best_edges:
                best_edges = edges_found
                best_adj = adj[:]
        
    except Exception as e:
        pass  # Fall through to other strategies
    
    # --- Strategy 2: Greedy with random restarts ---
    deadline = start_time + time_limit
    num_restarts = 200  # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization with random edge order
        adj = [0] * n
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        
        greedy_fill(adj, order=all_pairs, deadline=deadline)
        curr_edges = count_edges(adj)
        
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj[:]
    
    # --- Strategy 3: Local search on best found ---
    if best_adj is not None and time.time() < deadline - 5:
        improved_adj, improved_edges = local_search(
            best_adj[:], 
            max_iterations=50000, 
            deadline=deadline
        )
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj = improved_adj
    
    # --- Strategy 4: Final greedy pass on best_adj ---
    if best_adj is not None and time.time() < deadline:
        greedy_fill(best_adj, deadline=deadline)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
    
    # --- Strategy 5: More random restarts with different initial seeds ---
    if time.time() < deadline - 5:
        for restart in range(100):
            if time.time() > deadline:
                break
            
            adj = [0] * n
            
            # Start with a random valid subgraph (random sparse graph)
            for u, v in all_pairs:
                if random.random() < 0.1:  # Sparse initial
                    if can_add_edge_vectorized(adj, u, v):
                        add_edge(adj, u, v)
            
            # Now greedily fill
            greedy_fill(adj, deadline=deadline)
            curr_edges = count_edges(adj)
            
            if curr_edges > best_edges:
                best_edges = curr_edges
                best_adj = adj[:]
    
    # Final greedy pass
    if best_adj is not None and time.time() < deadline:
        greedy_fill(best_adj, deadline=deadline)
    
    if best_adj is None:
        # Fallback: 5 disjoint K4s
        best_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    add_edge(best_adj, u, v)
    
    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END
