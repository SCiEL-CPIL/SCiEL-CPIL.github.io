"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a graph on as many vertices as possible with no K5 and no independent 5-set.
    Uses local search (hill climbing with restarts) on the adjacency matrix to minimize
    the number of K5 and independent 5-sets.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin
    
    # Target sizes to try, in descending order of difficulty/size
    # We aim for the maximum n. R(5,5) is known to be > 42.
    # We will try to find a valid graph for N=42, then 43, etc.
    # If we find a valid graph for N, we return it.
    # Since R(5,5) <= 46, max N is 45.
    
    best_n = 37
    best_edges = None
    
    # Fallback: Paley graph on 37 vertices
    def get_paley_edges(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return edges

    best_edges = get_paley_edges(37)
    
    def count_k3_in_set(adj, nodes):
        """Count K3s in the subgraph induced by 'nodes'."""
        if len(nodes) < 3:
            return 0
        # nodes is a list of vertex indices
        # We check all triples in nodes
        count = 0
        n_nodes = len(nodes)
        # Optimization: use bitmasks or simple loops
        # For small sets (size < 20), simple loops are fine
        for i in range(n_nodes):
            u = nodes[i]
            for j in range(i + 1, n_nodes):
                v = nodes[j]
                if adj[u, v] == 1:
                    for k in range(j + 1, n_nodes):
                        w = nodes[k]
                        if adj[u, w] == 1 and adj[v, w] == 1:
                            count += 1
        return count

    def count_ind3_in_set(adj, nodes):
        """Count independent sets of size 3 in the subgraph induced by 'nodes'."""
        if len(nodes) < 3:
            return 0
        count = 0
        n_nodes = len(nodes)
        for i in range(n_nodes):
            u = nodes[i]
            for j in range(i + 1, n_nodes):
                v = nodes[j]
                if adj[u, v] == 0:
                    for k in range(j + 1, n_nodes):
                        w = nodes[k]
                        if adj[u, w] == 0 and adj[v, w] == 0:
                            count += 1
        return count

    def calculate_cost(adj, n):
        """
        Calculate total number of K5 and independent 5-sets.
        A K5 contains a pair (u,v) if u~v and there is a K3 in N(u) cap N(v).
        An Ind5 contains a pair (u,v) if u!~v and there is an Ind3 in NonN(u) cap NonN(v).
        Each K5 is counted C(5,2)=10 times (once for each edge pair).
        Each Ind5 is counted C(5,2)=10 times (once for each non-edge pair).
        So total cost = (sum over edges of K3_count) + (sum over non-edges of Ind3_count).
        If cost == 0, graph is valid.
        """
        total_cost = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    # Common neighbors
                    # N(u) & N(v)
                    # We can compute this efficiently
                    common = []
                    for w in range(n):
                        if w != u and w != v and adj[u, w] == 1 and adj[v, w] == 1:
                            common.append(w)
                    if len(common) >= 3:
                        total_cost += count_k3_in_set(adj, common)
                else:
                    # Common non-neighbors
                    # NonN(u) & NonN(v)
                    common = []
                    for w in range(n):
                        if w != u and w != v and adj[u, w] == 0 and adj[v, w] == 0:
                            common.append(w)
                    if len(common) >= 3:
                        total_cost += count_ind3_in_set(adj, common)
        return total_cost

    def delta_cost(adj, n, u, v):
        """
        Calculate change in cost if we flip edge (u,v).
        Returns (delta, new_cost_is_zero)
        """
        # Current state
        if adj[u, v] == 1:
            # We are removing edge (u,v)
            # Old contribution: K3s in N(u) cap N(v)
            common = [w for w in range(n) if w != u and w != v and adj[u, w] == 1 and adj[v, w] == 1]
            old_contrib = count_k3_in_set(adj, common)
            
            # New contribution: Ind3s in NonN(u) cap NonN(v) after flip
            # After flip, u and v are non-adjacent.
            # New common non-neighbors are the same as old common neighbors? No.
            # If w was in N(u) cap N(v), it is adjacent to both. After removing (u,v), w is still adjacent to both.
            # So w is NOT in NonN(u) cap NonN(v).
            # The set of common non-neighbors for the new non-edge (u,v) is:
            # {w | w!=u,v, adj[u,w]==0, adj[v,w]==0}
            new_common_non = [w for w in range(n) if w != u and w != v and adj[u, w] == 0 and adj[v, w] == 0]
            new_contrib = count_ind3_in_set(adj, new_common_non)
            
            delta = new_contrib - old_contrib
        else:
            # We are adding edge (u,v)
            # Old contribution: Ind3s in NonN(u) cap NonN(v)
            common_non = [w for w in range(n) if w != u and w != v and adj[u, w] == 0 and adj[v, w] == 0]
            old_contrib = count_ind3_in_set(adj, common_non)
            
            # New contribution: K3s in N(u) cap N(v) after flip
            new_common = [w for w in range(n) if w != u and w != v and adj[u, w] == 1 and adj[v, w] == 1]
            new_contrib = count_k3_in_set(adj, new_common)
            
            delta = new_contrib - old_contrib
            
        return delta

    def search(N, time_limit):
        """
        Attempt to find a valid graph of size N.
        Returns edge list if found, else None.
        """
        if time.time() > time_limit:
            return None
            
        # Try multiple restarts
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Initialize adjacency matrix
            # Strategy: Start with Paley 37 + 5 random? Or just random?
            # Random is simpler and avoids bias.
            adj = np.zeros((N, N), dtype=np.uint8)
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
                        
            current_cost = calculate_cost(adj, N)
            
            if current_cost == 0:
                # Convert to edges
                edges = []
                for i in range(N):
                    for j in range(i + 1, N):
                        if adj[i, j] == 1:
                            edges.append((i, j))
                return edges
                
            # Local search
            temp = 1.0
            min_temp = 0.01
            cooling = 0.995
            
            while time.time() < time_limit:
                if current_cost == 0:
                    edges = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            if adj[i, j] == 1:
                                edges.append((i, j))
                    return edges
                    
                # Pick random edge
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                    
                delta = delta_cost(adj, N, u, v)
                
                # Simulated Annealing acceptance
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    # Accept
                    adj[u, v] = 1 - adj[u, v]
                    adj[v, u] = adj[u, v]
                    current_cost += delta
                    temp *= cooling
                    if temp < min_temp:
                        temp = min_temp
                else:
                    # Reject
                    pass
                    
                # Cooling
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp
                    
        return None

    # Try to find a graph of size 42, then 43, 44, 45
    # We allocate time based on likelihood. 42 is most likely.
    # Let's give 50% time to 42, 20% to 43, 10% to 44, 10% to 45.
    # Total time available: 290s.
    
    time_alloc = {
        42: 145.0,
        43: 58.0,
        44: 29.0,
        45: 29.0
    }
    
    current_time = time.time()
    
    for N in [42, 43, 44, 45]:
        if time.time() > deadline - 5.0:
            break
            
        alloc = time_alloc.get(N, 10.0)
        end_time = min(time.time() + alloc, deadline - 5.0)
        
        if end_time <= time.time():
            continue
            
        edges = search(N, end_time)
        
        if edges is not None:
            # We found a valid graph of size N
            best_n = N
            best_edges = edges
            # Since we are iterating from high to low? No, low to high.
            # If we find 42, we should keep looking for 43?
            # The loop goes 42, 43, 44, 45.
            # If we find 42, we update best. Then we try 43.
            # If we find 43, we update best.
            # This is correct.
            
    # If we didn't find anything better than 37, we return 37.
    # But search(42) might fail.
    # We should ensure we return the best found.
    
    # Final safety check: verify the best graph found
    if best_n > 37:
        # Re-verify best_edges
        adj = np.zeros((best_n, best_n), dtype=np.uint8)
        for u, v in best_edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        cost = calculate_cost(adj, best_n)
        if cost != 0:
            # Invalid, fallback to 37
            best_n = 37
            best_edges = get_paley_edges(37)
            
    return best_n, best_edges

# EVOLVE-BLOCK-END
