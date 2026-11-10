import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590
    
    # 1. Find max independent set in P_41 (Paley graph 41)
    # Vertices 0..40. Edge if (u-v) % 41 is non-zero quadratic residue.
    n41 = 41
    q_res_41 = set()
    for x in range(1, n41):
        q_res_41.add((x * x) % n41)
        
    adj_41 = [0] * n41
    for i in range(n41):
        mask = 0
        for j in range(n41):
            if i == j: continue
            if (i - j) % n41 in q_res_41:
                mask |= (1 << j)
        adj_41[i] = mask
        
    full_41 = (1 << n41) - 1
    
    def solve_paley_41():
        # Use SA to find large independent set
        best_mask = 0
        best_size = 0
        
        # Greedy init
        for seed in range(100):
            if time.time() > deadline - 100: break
            rng = random.Random(seed)
            candidates = full_41
            selected = 0
            while candidates:
                # Pick node with min degree in candidates
                min_deg = 999
                min_node = -1
                c = candidates
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = (adj_41[v] & candidates).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_node = v
                if min_node == -1: break
                selected |= (1 << min_node)
                candidates &= ~(adj_41[min_node] | (1 << min_node))
            
            # Local search: add if possible
            improved = True
            while improved:
                improved = False
                blocked = 0
                s = selected
                while s:
                    lsb = s & -s
                    u = lsb.bit_length() - 1
                    s ^= lsb
                    blocked |= adj_41[u]
                avail = full_41 & ~blocked & ~selected
                if avail:
                    lsb = avail & -avail
                    v = lsb.bit_length() - 1
                    selected |= (1 << v)
                    improved = True
                    
            if selected.bit_count() > best_size:
                best_size = selected.bit_count()
                best_mask = selected
                
        # SA from best
        current_mask = best_mask
        current_size = best_size
        T = 1.0
        for step in range(50000):
            if time.time() > deadline - 50: break
            if current_mask == 0: break
            # Pick random u in set
            c = current_mask
            lsb = c & -c
            u = lsb.bit_length() - 1
            # Remove u
            new_mask = current_mask & ~(1 << u)
            # Pick random v not in set and not adjacent to new_mask
            blocked = 0
            s = new_mask
            while s:
                lsb = s & -s
                w = lsb.bit_length() - 1
                s ^= lsb
                blocked |= adj_41[w]
            avail = full_41 & ~blocked & ~new_mask
            if avail == 0:
                # Accept removal? No, size decreases.
                # Just skip or accept with prob?
                # If we can't add, we can just keep current.
                continue
            
            count = avail.bit_count()
            k = random.randint(0, count - 1)
            temp = avail
            for i in range(k):
                lsb = temp & -temp
                temp ^= lsb
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            
            candidate_mask = new_mask | (1 << v)
            candidate_size = candidate_mask.bit_count()
            delta = candidate_size - current_size
            
            if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
                current_mask = candidate_mask
                current_size = candidate_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            T *= 0.99995
            
        return best_mask, best_size

    mask_41, size_41 = solve_paley_41()
    
    # 2. Find max independent set in K_5
    # K_5 edges if diff in {0, 1, 4}.
    # Non-edges (distinct) if diff in {2, 3}.
    # Max indep set size 2. e.g. {0, 2}.
    # We will use {0, 2} as the base for K_5.
    # Actually, we can just use the product structure.
    
    # 3. Construct initial solution in G_205
    # Nodes in 41
    nodes_41 = []
    c = mask_41
    while c:
        lsb = c & -c
        u = lsb.bit_length() - 1
        nodes_41.append(u)
        c ^= lsb
        
    # Nodes in 5: {0, 2}
    nodes_5 = [0, 2]
    
    # CRT: x = 5k + a. x mod 41 = b.
    # 5k = b - a mod 41.
    # k = (b - a) * 33 mod 41.
    inv_5_41 = 33
    
    initial_nodes = []
    for a in nodes_5:
        for b in nodes_41:
            j = ((b - a) * inv_5_41) % 41
            x = (a + 5 * j) % 205
            initial_nodes.append(x)
            
    # Convert to mask for G_205
    n205 = 205
    full_205 = (1 << n205) - 1
    
    # Build adj_205
    # d is in F if d mod 5 in {0,1,4} AND d mod 41 in Q_41 U {0} AND d != 0
    # Actually, just check if d is square mod 205 and d != 0.
    q_res_205 = set()
    for x in range(1, 205):
        q_res_205.add((x * x) % 205)
        
    adj_205 = [0] * n205
    for i in range(n205):
        mask = 0
        for j in range(n205):
            if i == j: continue
            d = (i - j) % 205
            if d in q_res_205:
                mask |= (1 << j)
        adj_205[i] = mask
        
    # Convert initial_nodes to mask
    init_mask = 0
    for x in initial_nodes:
        init_mask |= (1 << x)
        
    # 4. SA on G_205
    current_mask = init_mask
    current_size = current_mask.bit_count()
    best_mask = current_mask
    best_size = current_size
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9999
    
    # Precompute list of nodes in mask for faster selection?
    # Or just use bit operations.
    
    for step in range(100000):
        if time.time() > deadline - 10:
            break
            
        # Get list of nodes in current_mask
        set_nodes = []
        c = current_mask
        while c:
            lsb = c & -c
            u = lsb.bit_length() - 1
            set_nodes.append(u)
            c ^= lsb
            
        if not set_nodes:
            break
            
        u = random.choice(set_nodes)
        
        # Remove u
        new_mask = current_mask & ~(1 << u)
        
        # Find valid v
        blocked = 0
        s = new_mask
        while s:
            lsb = s & -s
            w = lsb.bit_length() - 1
            s ^= lsb
            blocked |= adj_205[w]
            
        avail = full_205 & ~blocked & ~new_mask
        
        if avail == 0:
            # Can't add. Maybe just remove?
            # Delta = -1.
            # Accept with prob exp(-1/T)
            if random.random() < np.exp(-1 / T):
                current_mask = new_mask
                current_size -= 1
            # Else keep
        else:
            count = avail.bit_count()
            k = random.randint(0, count - 1)
            temp = avail
            for i in range(k):
                lsb = temp & -temp
                temp ^= lsb
            lsb = temp & -temp
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
        if T < T_min:
            T = T_min

    # Extract nodes from best_mask
    result_nodes = []
    c = best_mask
    while c:
        lsb = c & -c
        u = lsb.bit_length() - 1
        result_nodes.append(u)
        c ^= lsb
        
    if not result_nodes:
        return np.array([0], dtype=np.int64)
        
    return np.array(result_nodes, dtype=np.int64)
