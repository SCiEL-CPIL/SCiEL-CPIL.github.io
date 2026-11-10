import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # 1. Solve for max independent set in Paley Graph P_41
    # Vertices 0..40. Edge if (u-v) % 41 is a quadratic residue (non-zero).
    
    def solve_paley_41():
        n = 41
        # Quadratic residues mod 41
        q_res = set()
        for x in range(1, n):
            q_res.add((x * x) % n)
        
        # Build adjacency masks
        adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i == j: continue
                if (i - j) % n in q_res:
                    mask |= (1 << j)
            adj[i] = mask
            
        full_mask = (1 << n) - 1
        
        def greedy_min_degree(seed):
            rng = random.Random(seed)
            candidates = full_mask
            selected = 0
            for _ in range(n):
                if candidates == 0: break
                min_deg = n + 1
                min_nodes = []
                c = candidates
                while c:
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = (adj[v] & candidates).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_nodes = [v]
                    elif deg == min_deg:
                        min_nodes.append(v)
                if not min_nodes: break
                v = rng.choice(min_nodes)
                selected |= (1 << v)
                candidates &= ~(adj[v] | (1 << v))
            return selected

        def local_search_improve(mask):
            improved = True
            while improved:
                improved = False
                blocked = 0
                s = mask
                while s:
                    lsb = s & (-s)
                    u = lsb.bit_length() - 1
                    s ^= lsb
                    blocked |= adj[u]
                candidates = full_mask & ~blocked & ~mask
                if candidates != 0:
                    lsb = candidates & (-candidates)
                    v = lsb.bit_length() - 1
                    mask |= (1 << v)
                    improved = True
            return mask

        def simulated_annealing_41(initial_mask, max_steps=50000):
            current_mask = initial_mask
            current_size = current_mask.bit_count()
            best_mask = current_mask
            best_size = current_size
            
            T = 2.0
            T_min = 0.01
            cooling = 0.99995
            
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    break
                
                # Pick random u in set
                set_nodes = []
                c = current_mask
                while c:
                    lsb = c & -c
                    u = lsb.bit_length() - 1
                    set_nodes.append(u)
                    c ^= lsb
                if not set_nodes: break
                u = random.choice(set_nodes)
                
                new_mask = current_mask & ~(1 << u)
                blocked = 0
                s = new_mask
                while s:
                    lsb = s & -s
                    w = lsb.bit_length() - 1
                    s ^= lsb
                    blocked |= adj[w]
                
                valid_v = full_mask & ~blocked & ~new_mask
                if valid_v == 0: continue
                
                count = valid_v.bit_count()
                k = random.randint(0, count - 1)
                temp_v = valid_v
                for i in range(k):
                    lsb = temp_v & -temp_v
                    temp_v ^= lsb
                lsb = temp_v & -temp_v
                v = lsb.bit_length() - 1
                
                candidate_mask = new_mask | (1 << v)
                candidate_size = candidate_mask.bit_count()
                delta = candidate_size - current_size
                
                if delta >= 0 or (T > T_min and random.random() < np.exp(delta / T)):
                    current_mask = candidate_mask
                    current_size = candidate_size
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                
                T *= cooling
            
            return best_mask

        best_size = 0
        best_mask = 0
        
        # Phase 1: Greedy
        for seed in range(500):
            if time.time() > deadline - 100: break
            m41 = greedy_min_degree(seed)
            m41 = local_search_improve(m41)
            if m41.bit_count() > best_size:
                best_size = m41.bit_count()
                best_mask = m41
                
        # Phase 2: SA from best
        final_mask = simulated_annealing_41(best_mask)
        if final_mask.bit_count() > best_size:
            best_size = final_mask.bit_count()
            best_mask = final_mask
            
        # Phase 3: More SA restarts if time permits
        while time.time() < deadline - 50:
            # Restart from best with perturbation or random
            start_mask = best_mask
            # Perturb: remove 2 random nodes
            nodes = []
            c = start_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                nodes.append(u)
                c ^= lsb
            if len(nodes) >= 2:
                r1, r2 = random.sample(nodes, 2)
                start_mask = start_mask & ~(1 << r1) & ~(1 << r2)
            
            m41 = simulated_annealing_41(start_mask, max_steps=10000)
            if m41.bit_count() > best_size:
                best_size = m41.bit_count()
                best_mask = m41
                
        return best_mask

    # 2. Solve for max independent set in G_5
    # G_5 is C_5. Max independent set size 2.
    # e.g., {0, 2}
    s5_mask = (1 << 0) | (1 << 2) # 0 and 2
    
    # 3. Combine
    # Get best set in 41
    best_mask_41 = solve_paley_41()
    
    # Extract nodes from masks
    nodes_5 = []
    c = s5_mask
    while c:
        lsb = c & -c
        u = lsb.bit_length() - 1
        nodes_5.append(u)
        c ^= lsb
        
    nodes_41 = []
    c = best_mask_41
    while c:
        lsb = c & -c
        u = lsb.bit_length() - 1
        nodes_41.append(u)
        c ^= lsb
        
    # CRT to find x in 0..204 for each pair (a, b)
    # x = 5 * k
    # x mod 41 = b
    # 5 * k = b mod 41
    # k = b * 5^{-1} mod 41
    # 5 * 33 = 165 = 4*41 + 1. So 5^{-1} = 33 mod 41.
    inv_5_41 = 33
    
    result_nodes = []
    for a in nodes_5:
        for b in nodes_41:
            k = (b * inv_5_41) % 41
            x = (5 * k + a) % 205
            # Check if x mod 5 is a?
            # x = 5k + a. x mod 5 = a mod 5 = a. Correct.
            # x mod 41 = 5k + a mod 41.
            # 5k = b mod 41. So x mod 41 = b + a mod 41?
            # Wait.
            # We want x = a mod 5 AND x = b mod 41.
            # My formula: x = 5k + a.
            # x mod 5 = a.
            # x mod 41 = 5k + a mod 41.
            # We set 5k = b - a mod 41?
            # Let's re-derive.
            # x = a + 5j.
            # a + 5j = b mod 41.
            # 5j = b - a mod 41.
            # j = (b - a) * 33 mod 41.
            
            j = ((b - a) * inv_5_41) % 41
            x = (a + 5 * j) % 205
            result_nodes.append(x)
            
    # Verify uniqueness and validity
    # Since (a,b) pairs are unique, and CRT is bijection, x are unique.
    # Validity is guaranteed by construction if s5 and s41 are independent sets in their respective graphs.
    
    if not result_nodes:
        return np.array([0], dtype=np.int64)
        
    return np.array(result_nodes, dtype=np.int64)
