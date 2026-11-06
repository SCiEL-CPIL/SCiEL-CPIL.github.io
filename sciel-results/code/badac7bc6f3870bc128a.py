import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    def count_k5_and_indep5(adj_masks, n):
        """Count all K5 and independent 5-sets in the graph."""
        num_k5 = 0
        num_indep5 = 0
        for i in range(n - 4):
            mi = adj_masks[i]
            for j in range(i + 1, n - 3):
                if (mi >> j) & 1:
                    # i-j is an edge; need k,l,m all adjacent to i,j and to each other
                    common = mi & adj_masks[j]
                    # Candidates must be > j
                    common &= ~((1 << (j + 1)) - 1)
                    # Get candidate list
                    cands = []
                    tmp = common
                    while tmp:
                        bit = tmp & (-tmp)
                        cands.append(bit.bit_length() - 1)
                        tmp ^= bit
                    for k_idx in range(len(cands) - 2):
                        k = cands[k_idx]
                        mk = adj_masks[k]
                        for l_idx in range(k_idx + 1, len(cands) - 1):
                            l = cands[l_idx]
                            if not ((mk >> l) & 1):
                                continue
                            ml = adj_masks[l]
                            for m_idx in range(l_idx + 1, len(cands)):
                                m = cands[m_idx]
                                if (mk >> m) & 1 and (ml >> m) & 1:
                                    num_k5 += 1
                else:
                    # i-j is NOT an edge; need k,l,m all non-adjacent to i,j and to each other
                    # Non-neighbors of i (excluding self): ~(mi | (1<<i))
                    non_i = ~(mi | (1 << i))
                    non_j = ~(adj_masks[j] | (1 << j))
                    common = non_i & non_j
                    common &= ~((1 << (j + 1)) - 1)  # only vertices > j
                    common &= (1 << n) - 1  # mask to n bits
                    cands = []
                    tmp = common
                    while tmp:
                        bit = tmp & (-tmp)
                        cands.append(bit.bit_length() - 1)
                        tmp ^= bit
                    for k_idx in range(len(cands) - 2):
                        k = cands[k_idx]
                        mk = adj_masks[k]
                        for l_idx in range(k_idx + 1, len(cands) - 1):
                            l = cands[l_idx]
                            if (mk >> l) & 1:
                                continue
                            ml = adj_masks[l]
                            for m_idx in range(l_idx + 1, len(cands)):
                                m = cands[m_idx]
                                if not ((mk >> m) & 1) and not ((ml >> m) & 1):
                                    num_indep5 += 1
        return num_k5, num_indep5

    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            mi = adj_masks[i]
            for j in range(i + 1, n):
                if (mi >> j) & 1:
                    edges.append((i, j))
        return edges

    def count_affected(adj_masks, n, u, v):
        """Count K5 and Indep5 containing the pair (u,v) using bitmask acceleration."""
        num_k5 = 0
        num_indep5 = 0
        is_edge = (adj_masks[u] >> v) & 1
        mu = adj_masks[u]
        mv = adj_masks[v]
        
        # Build candidate set: vertices w not in {u,v}
        if is_edge:
            # For K5: w must be adjacent to both u and v
            common = mu & mv
            common &= ~(1 << u)
            common &= ~(1 << v)
        else:
            # For Indep5: w must be non-adjacent to both u and v
            non_u = ~(mu | (1 << u))
            non_v = ~(mv | (1 << v))
            common = non_u & non_v
            common &= ~(1 << u)
            common &= ~(1 << v)
            common &= (1 << n) - 1
        
        # Extract candidate vertices
        cands = []
        tmp = common
        while tmp:
            bit = tmp & (-tmp)
            cands.append(bit.bit_length() - 1)
            tmp ^= bit
        
        nc = len(cands)
        for a_idx in range(nc - 2):
            a = cands[a_idx]
            ma = adj_masks[a]
            for b_idx in range(a_idx + 1, nc - 1):
                b = cands[b_idx]
                ab_edge = (ma >> b) & 1
                mb = adj_masks[b]
                for c_idx in range(b_idx + 1, nc):
                    c = cands[c_idx]
                    ac_edge = (ma >> c) & 1
                    bc_edge = (mb >> c) & 1
                    
                    if is_edge:
                        # K5: a,b,c must form triangle (all edges)
                        if ab_edge and ac_edge and bc_edge:
                            num_k5 += 1
                    else:
                        # Indep5: a,b,c must have no edges
                        if not ab_edge and not ac_edge and not bc_edge:
                            num_indep5 += 1
        return num_k5, num_indep5

    def init_counts(adj_masks, n):
        return count_k5_and_indep5(adj_masks, n)

    def run_sa(n, initial_adj, max_time):
        """Run Simulated Annealing on a graph of size n."""
        if time.time() > max_time:
            return initial_adj, init_counts(initial_adj, n)[0] + init_counts(initial_adj, n)[1]
        
        current_adj = initial_adj[:]
        current_k5, current_i5 = init_counts(current_adj, n)
        current_cost = current_k5 + current_i5
        best_adj = current_adj[:]
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < max_time:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_k5_aff, old_i5_aff = count_affected(current_adj, n, u, v)
            
            mask_u = current_adj[u]
            mask_v = current_adj[v]
            current_adj[u] = mask_u ^ (1 << v)
            current_adj[v] = mask_v ^ (1 << u)
            
            new_k5_aff, new_i5_aff = count_affected(current_adj, n, u, v)
            
            current_k5 = current_k5 - old_k5_aff + new_k5_aff
            current_i5 = current_i5 - old_i5_aff + new_i5_aff
            new_cost = current_k5 + current_i5
            
            delta = new_cost - best_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = current_adj[:]
                    if best_cost == 0:
                        return best_adj, 0
            else:
                # Revert
                current_adj[u] = mask_u
                current_adj[v] = mask_v
                current_k5 = current_k5 - new_k5_aff + old_k5_aff
                current_i5 = current_i5 - new_i5_aff + old_i5_aff
            
            T = max(T_min, T * cooling)
        
        return best_adj, best_cost

    # Target n=42 first
    n_target = 42
    time_budget = 280
    
    # Initialization: Paley 37 + 5 random vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj_init = [0] * n_target
    for i in range(37):
        for j in range(i + 1, 37):
            if (j - i) % q in residues:
                adj_init[i] |= (1 << j)
                adj_init[j] |= (1 << i)
    
    # Randomly connect the last 5 vertices to the first 37 and among themselves
    for i in range(37, n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj_init[i] |= (1 << j)
                adj_init[j] |= (1 << i)
        # Connect to first 37
        for j in range(37):
            if random.random() < 0.5:
                adj_init[i] |= (1 << j)
                adj_init[j] |= (1 << i)

    # Run SA for n=42
    best_adj_42, best_cost_42 = run_sa(n_target, adj_init, start_time + time_budget * 0.6)
    
    if best_cost_42 == 0:
        return n_target, get_edges(best_adj_42, n_target)

    # If 42 failed, try 41
    n_target = 41
    if time.time() < start_time + time_budget * 0.8:
        # Use the best 42-vertex graph's first 41 vertices? Or restart?
        # Restart with Paley 37 + 4 random
        adj_init_41 = adj_init[:41]
        # Need to clear bits for vertex 41 in the first 40? No, adj_init is already built for 42.
        # We need a clean 41-vertex initialization.
        adj_init_41 = [0] * 41
        for i in range(37):
            for j in range(i + 1, 37):
                if (j - i) % q in residues:
                    adj_init_41[i] |= (1 << j)
                    adj_init_41[j] |= (1 << i)
        for i in range(37, 41):
            for j in range(i + 1, 41):
                if random.random() < 0.5:
                    adj_init_41[i] |= (1 << j)
                    adj_init_41[j] |= (1 << i)
            for j in range(37):
                if random.random() < 0.5:
                    adj_init_41[i] |= (1 << j)
                    adj_init_41[j] |= (1 << i)
        
        best_adj_41, best_cost_41 = run_sa(41, adj_init_41, start_time + time_budget * 0.8)
        if best_cost_41 == 0:
            return 41, get_edges(best_adj_41, 41)

    # Fallback to 37
    adj37 = [0] * 37
    for i in range(37):
        for j in range(i + 1, 37):
            if (j - i) % q in residues:
                adj37[i] |= (1 << j)
                adj37[j] |= (1 << i)
    
    return 37, get_edges(adj37, 37)
