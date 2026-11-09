import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    n = 20
    
    def is_k33_free(adj_bits):
        """
        Check if the graph is K3,3-free.
        A K3,3 exists if there are two disjoint triples of vertices 
        such that all 9 edges between them exist.
        Equivalently, for any triple of vertices A, if they have 3 or more 
        common neighbors, we need to check if those common neighbors contain 
        a triple that is also connected to A? No.
        
        Definition: A K3,3 is a subgraph with partition {u1,u2,u3} and {v1,v2,v3} 
        such that all edges ui-vj exist.
        
        Efficient check:
        For each pair of vertices (i, j), let C_ij be the set of common neighbors.
        If |C_ij| >= 3, then for any k in C_ij, the triple {i, j, k} has common neighbors C_ij \ {k}? No.
        
        Let's stick to the definition:
        For every triple of vertices T = {a, b, c}, let N(T) be the set of vertices 
        adjacent to all of a, b, c.
        If |N(T)| >= 3, does that imply K3,3?
        Not necessarily. The vertices in N(T) are connected to a,b,c. 
        But do they form a clique? No.
        Wait. If N(T) contains {x, y, z}, then x,y,z are all connected to a,b,c.
        This forms a K3,3 with parts {a,b,c} and {x,y,z}.
        Yes!
        So the condition "no three vertices have three common neighbours" is EXACTLY 
        the condition for being K3,3-free.
        
        So: For all distinct a, b, c: |N(a) ∩ N(b) ∩ N(c)| <= 2.
        """
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    if common.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def greedy_construct(order):
        """
        Build a graph by adding edges in the specified order.
        Only add edge (u, v) if the resulting graph remains K3,3-free.
        """
        adj_bits = [0] * n
        for u, v in order:
            # Tentatively add edge
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            # Check if K3,3-free
            # Optimization: Only check triples involving u or v
            # If adding (u,v) creates a K3,3, that K3,3 must include u or v.
            # Why? Because the rest of the graph was K3,3-free.
            # If a new K3,3 appears, it must use the new edge (u,v).
            # So it must have u in one part and v in the other (or vice versa).
            
            # Case 1: u is in Part A, v is in Part B.
            # Part A = {u, a, b}, Part B = {v, c, d}
            # Edges needed:
            # u-v (yes)
            # u-c, u-d (u connected to c,d)
            # a-v, b-v (a,b connected to v)
            # a-c, a-d, b-c, b-d (a,b connected to c,d)
            
            # This is a bit complex to check incrementally without full scan.
            # Given n=20, full scan is 1140 triples.
            # Each check is 3 bit ops + bit_count. Very fast.
            # Let's just do full scan for simplicity and reliability.
            
            if not is_k33_free(adj_bits):
                # Revert
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            # else keep edge
        
        return adj_bits

    best_adj = [0] * n
    best_edges = 0
    
    # Generate a few random edge orders
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    num_restarts = 100  # Adjust based on time budget
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        order = all_edges[:]
        random.shuffle(order)
        
        # To improve quality, we can bias the order.
        # But random is a good baseline.
        # Alternatively, we can try to add edges in a structured way?
        # Let's stick to random shuffles for now.
        
        adj = greedy_construct(order)
        e_count = count_edges(adj)
        
        if e_count > best_edges:
            best_edges = e_count
            best_adj = adj[:]
            
            # If we reach a very high number, we might stop early?
            # ex(20, K3,3) is likely around 100-150?
            # Let's see.
            
    # Final local search / refinement on the best graph found
    # Try to add missing edges one by one (greedy repair)
    current_adj = best_adj[:]
    
    # List of missing edges
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] >> j) & 1]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 2:
            break
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        if is_k33_free(current_adj):
            pass # Keep
        else:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (current_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

# EVOLVE-BLOCK-END
