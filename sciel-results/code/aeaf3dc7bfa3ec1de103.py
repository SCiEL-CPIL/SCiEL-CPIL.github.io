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
                    break
                if ((~adj[v]) & others) != others:
                    is_indep = False
                    break
            if is_clique:
                k4s.append(mask)
            if is_indep:
                ind4s.append(mask)
        return k4s, ind4s

    # Helper to find a valid connection vector for a new vertex using improved local search
    def find_valid_connection(adj, n, k4s, ind4s, deadline):
        full_mask = (1 << n) - 1
        
        def cost(mask):
            """
            Cost = number of K4s fully contained in mask + number of Ind4s fully contained in complement
            We want cost = 0
            """
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
        
        # Multiple restarts with different strategies
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random
                mask = random.getrandbits(n)
            elif restart % 3 == 1:
                # Start with half the vertices
                mask = 0
                for i in range(n):
                    if random.random() < 0.5:
                        mask |= (1 << i)
            else:
                # Start with vertices that have low degree (more likely to be in independent sets)
                mask = 0
                for i in range(n):
                    deg = bin(adj[i]).count('1')
                    if deg < n // 2 and random.random() < 0.6:
                        mask |= (1 << i)
            
            c = cost(mask)
            if c == 0:
                return mask
            
            # Local search: try flipping single bits
            T = 1.0
            cooling_rate = 0.995
            min_T = 0.005
            stuck_count = 0
            max_stuck = 1000
            
            while T > min_T and stuck_count < max_stuck:
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
                    stuck_count = 0
                    if c == 0:
                        return mask
                else:
                    # Accept with probability
                    if random.random() < pow(2.718281828, -delta / T):
                        mask = new_mask
                        c = new_c
                        stuck_count = 0
                        if c == 0:
                            return mask
                    else:
                        stuck_count += 1
                
                T *= cooling_rate
            
            if c < best_cost:
                best_cost = c
                best_mask = mask
                if c == 0:
                    return mask
            
            # If we're stuck at a low cost, try targeted moves
            if best_cost <= 3 and time.time() < deadline:
                # Try to fix the remaining violations
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    if best_cost == 0:
                        return best_mask
                    
                    # Find which K4s are violated
                    violated_k4s = []
                    for k4 in k4s:
                        if (best_mask & k4) == k4:
                            violated_k4s.append(k4)
                    
                    # Find which Ind4s are violated
                    comp = best_mask ^ full_mask
                    violated_ind4s = []
                    for i4 in ind4s:
                        if (comp & i4) == i4:
                            violated_ind4s.append(i4)
                    
                    if not violated_k4s and not violated_ind4s:
                        return best_mask
                    
                    # Try to fix one violation at a time
                    if violated_k4s and random.random() < 0.5:
                        # Pick a random violated K4 and try to remove one of its vertices from mask
                        k4 = random.choice(violated_k4s)
                        bits_in_k4 = [i for i in range(n) if (k4 >> i) & 1]
                        if bits_in_k4:
                            bit = random.choice(bits_in_k4)
                            new_mask = best_mask & ~(1 << bit)
                            new_c = cost(new_mask)
                            if new_c < best_cost:
                                best_mask = new_mask
                                best_cost = new_c
                                if best_cost == 0:
                                    return best_mask
                    elif violated_ind4s:
                        # Pick a random violated Ind4 and try to add one of its vertices to mask
                        i4 = random.choice(violated_ind4s)
                        bits_in_i4 = [i for i in range(n) if (i4 >> i) & 1]
                        if bits_in_i4:
                            bit = random.choice(bits_in_i4)
                            new_mask = best_mask | (1 << bit)
                            new_c = cost(new_mask)
                            if new_c < best_cost:
                                best_mask = new_mask
                                best_cost = new_c
                                if best_cost == 0:
                                    return best_mask
        
        return best_mask if best_cost == 0 else None

    # Grow the graph
    for target_n in range(q + 1, 43):
        if time.time() > deadline - 30:
            break
        
        # Get K4s and Ind4s for current graph
        k4s, ind4s = get_k4s_and_ind4s(adj, n)
        
        # Try to find a valid connection for the new vertex
        mask = find_valid_connection(adj, n, k4s, ind4s, deadline)
        
        if mask is None:
            # If we can't add the vertex, we stop.
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
