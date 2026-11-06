import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n = 20
    total_possible_edges = n * (n - 1) // 2
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Bitmask adjacency
    adj = [0] * n
    
    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def is_valid_add(u, v):
        # Check if adding (u, v) creates a K2,3
        # 1. (u, v) must not already have > 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # 2. Adding (u, v) increases common neighbors for pairs (u, w) where w in N(v)
        #    and pairs (v, w) where w in N(u).
        #    If any such pair already has 2 common neighbors, adding (u, v) makes it 3.
        
        # Check neighbors of v (excluding u)
        mask_v = adj[v] & ~(1 << u)
        temp = mask_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # If u and w already share 2 neighbors, adding v (which is neighbor of both) makes it 3
            if (adj[u] & adj[w]).bit_count() >= 2:
                return False
            temp ^= lsb
            
        # Check neighbors of u (excluding v)
        mask_u = adj[u] & ~(1 << v)
        temp = mask_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # If v and w already share 2 neighbors, adding u (which is neighbor of both) makes it 3
            if (adj[v] & adj[w]).bit_count() >= 2:
                return False
            temp ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # 1. Greedy Initialization
    # Shuffle edges and add if valid
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    for u, v in edges_to_try:
        if is_valid_add(u, v):
            add_edge(u, v)
    
    best_count = count_edges()
    best_adj = adj[:]
    
    # 2. Local Search / Simulated Annealing
    # We maintain a list of present and missing edges for efficient sampling
    present_edges = []
    missing_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                present_edges.append((i, j))
            else:
                missing_edges.append((i, j))
                
    current_count = best_count
    current_adj = adj[:]
    
    # Parameters
    T = 1.0
    alpha = 0.9999
    
    while time.time() < deadline:
        # Pick a move type
        # Strategy: 
        # - If we can add an edge, we should try to do so (improves count).
        # - If we are stuck (no valid adds), we need to remove some edges to create space.
        # - We can also do swaps: remove one, add one.
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add a random missing edge
            if not missing_edges:
                continue
            idx = random.randint(0, len(missing_edges) - 1)
            u, v = missing_edges[idx]
            if is_valid_add(u, v):
                # Valid add: Accept
                add_edge(u, v)
                current_count += 1
                # Update lists
                missing_edges.pop(idx)
                present_edges.append((u, v))
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
            else:
                # Invalid add: Do nothing for now
                pass
                
        elif move_type < 0.8:
            # Try to remove a random present edge
            # This is a "worse" move, so accept with probability exp(1/T)
            if not present_edges:
                continue
            idx = random.randint(0, len(present_edges) - 1)
            u, v = present_edges[idx]
            
            # Acceptance probability for worsening move (delta = -1)
            # P = exp(-delta/T) = exp(1/T)
            # Cap at 1.0
            if T > 0.01:
                prob = min(1.0, math.exp(1.0 / T))
            else:
                prob = 1.0 # If T is tiny, exp(1/T) is huge, so always accept? 
                           # Actually, if T is tiny, we should NOT accept worse moves.
                           # Wait. Standard SA: accept worse with prob exp(-Delta/T).
                           # Delta = New - Old = -1.
                           # exp(-(-1)/T) = exp(1/T).
                           # If T is small (e.g., 0.01), exp(100) is huge -> Prob ~ 1.
                           # This is WRONG for maximization.
                           # We want to ACCEPT worse moves LESS often as T drops.
                           # Let's re-evaluate.
                           # Objective: Maximize E.
                           # Move: Remove edge. E_new = E_old - 1.
                           # Delta = -1.
                           # Accept if rand < exp(-Delta/T) = exp(1/T).
                           # If T=1, exp(1) = 2.7 -> Prob 1.
                           # If T=0.1, exp(10) = 22000 -> Prob 1.
                           # This logic implies we ALWAYS accept removals at low T?
                           # That would destroy the graph.
                           
                           # Correction:
                           # Usually SA minimizes Energy. Here we Maximize Edges.
                           # Let Energy = -Edges.
                           # Move: Remove edge. Edges decrease by 1. Energy increases by 1.
                           # Delta_E = +1.
                           # Accept if rand < exp(-Delta_E / T) = exp(-1/T).
                           # If T=1, exp(-1) = 0.36.
                           # If T=0.1, exp(-10) = 0.00004.
                           # This makes sense: at low T, we rarely accept worse moves.
                           
            if T > 0.001:
                prob = math.exp(-1.0 / T)
            else:
                prob = 0.0
                
            if random.random() < prob:
                # Accept removal
                remove_edge(u, v)
                current_count -= 1
                present_edges.pop(idx)
                missing_edges.append((u, v))
                
        else:
            # Swap: Remove one present, try to add one missing
            if not present_edges or not missing_edges:
                continue
            
            idx_p = random.randint(0, len(present_edges) - 1)
            u, v = present_edges[idx_p]
            
            idx_m = random.randint(0, len(missing_edges) - 1)
            x, y = missing_edges[idx_m]
            
            # Temporarily remove (u, v)
            remove_edge(u, v)
            
            # Try to add (x, y)
            if is_valid_add(x, y):
                # Valid swap: Edge count stays same (remove 1, add 1)
                # But structure changes. This helps escape local optima.
                # Accept with probability 1 (neutral move) or slightly biased?
                # Let's accept with prob 0.5 to allow exploration, or just 1.0.
                # Since count is same, it's a neutral move. 
                # In SA, neutral moves are often accepted to explore.
                add_edge(x, y)
                
                # Update lists
                present_edges.pop(idx_p)
                present_edges.append((x, y))
                missing_edges.pop(idx_m)
                missing_edges.append((u, v))
            else:
                # Invalid swap: Revert removal
                add_edge(u, v)
                
        # Cooling
        T *= alpha
        if T < 0.001:
            T = 0.001 # Prevent underflow/overflow issues in exp

    # Return best result
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
                
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    # Validate
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    if valid:
        print("Valid")
    else:
        print("Invalid")
