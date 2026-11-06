import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for safety and final checks
    
    # 1. Construct Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = q
    
    # Helper to get all K4 and Ind4 in current graph
    def get_k4s_and_ind4s(adj, n):
        k4s = []
        ind4s = []
        # Iterate over all 4-sets
        for combo in combinations(range(n), 4):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            is_clique = True
            is_indep = True
            for v in combo:
                others = mask ^ (1 << v)
                if (adj[v] & others) != others:
                    is_clique = False
                if ((~adj[v]) & others) != others:
                    is_indep = False
            if is_clique:
                k4s.append(mask)
            if is_indep:
                ind4s.append(mask)
        return k4s, ind4s

    # Helper to find a valid connection vector for a new vertex using Simulated Annealing
    def find_valid_connection(adj, n, k4s, ind4s, deadline):
        full_mask = (1 << n) - 1
        
        def cost(mask):
            c = 0
            for k4 in k4s:
                if (mask & k4) == k4:
                    c += 1
            comp = mask ^ full_mask
            for i4 in ind4s:
                if (comp & i4) == i4:
                    c += 1
            return c
        
        best_mask = None
        best_cost = float('inf')
        
        # Simulated Annealing parameters
        # We want to find cost 0.
        # Start with random masks
        
        for restart in range(100): # More restarts
            if time.time() > deadline:
                break
            
            # Initialize
            mask = random.getrandbits(n)
            c = cost(mask)
            if c == 0:
                return mask
            
            # SA Loop
            T = 1.0
            cooling_rate = 0.99
            min_T = 0.01
            
            while T > min_T:
                if time.time() > deadline:
                    break
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                new_mask = mask ^ (1 << bit)
                new_c = cost(new_mask)
                
                delta = new_c - c
                if delta < 0:
                    # Accept
                    mask = new_mask
                    c = new_c
                    if c == 0:
                        return mask
                else:
                    # Accept with probability
                    if random.random() < pow(2.718281828, -delta / T):
                        mask = new_mask
                        c = new_c
                        if c == 0:
                            return mask
                
                T *= cooling_rate
                
            if c < best_cost:
                best_cost = c
                best_mask = mask
                if c == 0:
                    return mask
        
        return best_mask if best_cost == 0 else None

    # Grow the graph
    for target_n in range(q + 1, 43):
        if time.time() > deadline - 20:
            break
        
        # Get K4s and Ind4s for current graph
        k4s, ind4s = get_k4s_and_ind4s(adj, n)
        
        # Try to find a valid connection for the new vertex
        mask = find_valid_connection(adj, n, k4s, ind4s, deadline)
        
        if mask is None:
            # If we can't add the vertex, we stop. 
            # Note: It is possible that a different choice of earlier vertices would allow growth, 
            # but backtracking is complex. We rely on the strength of the local search.
            break
        else:
            # Add the new vertex
            new_idx = n
            adj.append(mask)
            # Update adjacency of existing vertices
            for i in range(n):
                if (mask >> i) & 1:
                    adj[i] |= (1 << new_idx)
            n += 1
            
    # Final Verification: Check if the resulting graph is (5,5)-free
    # This is crucial to ensure we don't return an invalid graph due to a bug or search error
    def is_55_free(n, adj):
        # Check for K5
        for combo in combinations(range(n), 5):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            is_clique = True
            for v in combo:
                others = mask ^ (1 << v)
                if (adj[v] & others) != others:
                    is_clique = False
                    break
            if is_clique:
                return False
        
        # Check for Independent 5
        for combo in combinations(range(n), 5):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            is_indep = True
            for v in combo:
                others = mask ^ (1 << v)
                if ((~adj[v]) & others) != others:
                    is_indep = False
                    break
            if is_indep:
                return False
        return True

    if not is_55_free(n, adj):
        # Fallback to Paley(37) if something went wrong
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        n = q

    # Convert adj to edge list
    edges = []
    for i in range(n):
        m = adj[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append((i, j))
                
    return n, edges
