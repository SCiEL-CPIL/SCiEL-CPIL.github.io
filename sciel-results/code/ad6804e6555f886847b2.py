"""Optimized K2,3-free graph construction on 20 vertices using Max-SAT-inspired approach."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import os
import subprocess
import tempfile
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        """Count set bits in integer x."""
        return bin(x).count('1')
    
    def common_neighbors_count(adj, u, v):
        """Count common neighbors of u and v using bitsets."""
        return popcount(adj[u] & adj[v])
    
    def is_k23_free(adj):
        """Check if the graph has no K2,3 subgraph."""
        for u in range(n):
            for v in range(u + 1, n):
                if common_neighbors_count(adj, u, v) >= 3:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False  # Edge already exists
        
        # Check pair (u, v):
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        
        # Check pairs (v, y) for y in neighbors of u
        for y in range(n):
            if y == u or y == v:
                continue
            if (adj[u] >> y) & 1:  # y is a neighbor of u
                if popcount(adj[v] & adj[y]) >= 2:
                    return False
        
        # Check pairs (u, x) for x in neighbors of v
        for x in range(n):
            if x == u or x == v:
                continue
            if (adj[v] >> x) & 1:  # x is a neighbor of v
                if popcount(adj[u] & adj[x]) >= 2:
                    return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Max-SAT based construction
    def maxsat_construct():
        """Try to find dense K2,3-free graph using Max-SAT solver."""
        if time.time() > deadline - 10:
            return None
        
        # Check if minisat or picosat is available
        solver = None
        for candidate in ['minisat', 'picosat', 'maxsat']:
            try:
                subprocess.run([candidate, '--help'], stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, timeout=2)
                solver = candidate
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if solver is None:
            return None
        
        # Build the Max-SAT instance
        # Variables: x_{u,v} for each edge (u,v), u < v
        edge_index = {}
        edge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                idx = len(edge_list)
                edge_index[(u, v)] = idx
                edge_list.append((u, v))
        
        m = len(edge_list)  # Number of edge variables
        
        # Generate clauses
        # For each pair (u,v) and each triple of other vertices (a,b,c),
        # we cannot have all three edges (u,a), (v,a), (u,b), (v,b), (u,c), (v,c) true
        # Actually, K2,3 means u and v have at least 3 common neighbors
        # So for each pair (u,v), the number of w such that both (u,w) and (v,w) are edges must be <= 2
        
        # This is equivalent to: for each pair (u,v) and each triple of distinct vertices (a,b,c) != u,v,
        # at most 2 of the 6 edges {(u,a),(v,a),(u,b),(v,b),(u,c),(v,c)} can be true
        # But that's not quite right either.
        
        # Correct formulation: For each pair (u,v), let S_{u,v} be the set of vertices w 
        # such that both (u,w) and (v,w) are edges. We need |S_{u,v}| <= 2.
        # This is a cardinality constraint.
        
        # To encode as Max-SAT soft clauses:
        # For each pair (u,v) and each subset T of {0,...,n-1}\{u,v} with |T|=3,
        # add a soft clause that says: not all three of (u,w1)&(v,w1), (u,w2)&(v,w2), (u,w3)&(v,w3) are true
        # where w1,w2,w3 are the vertices in T.
        
        # This requires auxiliary variables for the ANDs, which gets complex.
        
        # Alternative simpler approach: Use the fact that K2,3-free means no two vertices 
        # share 3+ common neighbors. We can use a different encoding.
        
        # For now, let's try a simpler heuristic: start with a random graph and iteratively 
        # remove edges that violate K2,3-freeness, then try to add edges back.
        
        return None
    
    # Greedy construction with randomization
    def greedy_build(seed):
        rng = random.Random(seed)
        adj = [0] * n
        
        # Try to add edges in random order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        
        # Saturate: try to add any remaining valid edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
                        if time.time() > deadline - 2:
                            return adj
        return adj
    
    # Simulated annealing with bitmask representation
    def simulated_annealing(adj, rng, max_steps=10000):
        """Simulated annealing to escape local optima."""
        adj = adj[:]
        best_count = edge_count(adj)
        best_adj = adj[:]
        
        current_temp = 1.0
        cooling_rate = 0.9995
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Try to add any single missing edge (always accept)
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        if edge_count(adj) > best_count:
                            best_count = edge_count(adj)
                            best_adj = adj[:]
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try a move: remove one edge and try to add two others
            edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges_list.append((u, v))
            
            if not edges_list:
                break
            
            ru, rv = rng.choice(edges_list)
            saved_adj = adj[:]
            
            remove_edge(adj, ru, rv)
            
            # Try to add up to 2 new edges
            added = 0
            candidates = [(u, v) for u in range(n) for v in range(u + 1, n) 
                         if not ((adj[u] >> v) & 1) and (u, v) != (ru, rv)]
            rng.shuffle(candidates)
            
            for u, v in candidates:
                if added >= 2:
                    break
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    added += 1
            
            new_count = edge_count(adj)
            
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
            elif new_count == best_count and rng.random() < current_temp:
                # Accept neutral move with some probability
                pass
            else:
                # Revert
                adj = saved_adj[:]
            
            current_temp *= cooling_rate
        
        return best_adj
    
    # Run multiple greedy builds and pick the best
    best_adj = None
    best_count = 0
    
    # Try several different seeds
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 20:
            break
        
        adj = greedy_build(trial * 12345 + 678)
        count = edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Simulated annealing on the best result
    if best_adj is not None and time.time() < deadline - 10:
        rng = random.Random(42)
        best_adj = simulated_annealing(best_adj, rng, max_steps=15000)
    
    # Try Max-SAT if we have time
    if time.time() < deadline - 15:
        maxsat_result = maxsat_construct()
        if maxsat_result is not None:
            maxsat_count = edge_count(maxsat_result)
            if maxsat_count > best_count:
                best_adj = maxsat_result
                best_count = maxsat_count
    
    # Verify the result is valid
    if best_adj is None:
        return []
    
    if not is_k23_free(best_adj):
        return []
    
    return adj_to_edges(best_adj)

# EVOLVE-BLOCK-END
