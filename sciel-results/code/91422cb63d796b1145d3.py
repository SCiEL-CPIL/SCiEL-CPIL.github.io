import numpy as np
import random
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask of nodes j such that (i-j) % m in forbidden
    adj = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i + d) % m
            mask |= (1 << j)
        adj[i] = mask
    
    best_set = 0
    best_size = 0
    
    def calc_conflicts(S):
        c = 0
        temp = S
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            c += (S & adj[i]).bit_count()
            temp ^= lsb
        return c // 2
    
    def greedy_init():
        """Greedy construction: pick random order, add if no conflict."""
        S = 0
        nodes = list(range(m))
        random.shuffle(nodes)
        for n in nodes:
            if (S & adj[n]) == 0:
                S |= (1 << n)
        return S
    
    def local_improve(S, max_iters=50000):
        """Local search: try single-bit flips to improve."""
        penalty = 20
        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        
        for _ in range(max_iters):
            idx = random.randint(0, m - 1)
            bit = 1 << idx
            
            if S & bit:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = -neigh
                delta_size = -1
            else:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = neigh
                delta_size = 1
                
            delta_energy = -delta_size + penalty * delta_conf
            
            if delta_energy < 0:
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
            elif delta_conf == 0 and delta_size == 1:
                # Adding a non-conflicting node is always good
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
        
        return S
    
    def run_sa(start_state, iterations, t_start, t_end):
        """Simulated annealing with single-bit flips."""
        nonlocal best_set, best_size
        
        S = start_state
        penalty = 20
        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        
        if current_conf == 0 and current_size > best_size:
            best_size = current_size
            best_set = S
        
        for i in range(iterations):
            t = t_start + (t_end - t_start) * (i / iterations)
            
            idx = random.randint(0, m - 1)
            bit = 1 << idx
            
            if S & bit:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = -neigh
                delta_size = -1
            else:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = neigh
                delta_size = 1
                
            delta_energy = -delta_size + penalty * delta_conf
            
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                if t > 1e-9:
                    accept = random.random() < pow(2.718281828, -delta_energy / t)
            
            if accept:
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
                
                if current_conf == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = S
    
    def run_sa_with_swaps(start_state, iterations, t_start, t_end):
        """SA with both single-bit flips and 2-swap moves."""
        nonlocal best_set, best_size
        
        S = start_state
        penalty = 20
        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        
        if current_conf == 0 and current_size > best_size:
            best_size = current_size
            best_set = S
        
        for i in range(iterations):
            t = t_start + (t_end - t_start) * (i / iterations)
            
            # 80% single flip, 20% 2-swap
            if random.random() < 0.8:
                idx = random.randint(0, m - 1)
                bit = 1 << idx
                
                if S & bit:
                    neigh = (S & adj[idx]).bit_count()
                    delta_conf = -neigh
                    delta_size = -1
                else:
                    neigh = (S & adj[idx]).bit_count()
                    delta_conf = neigh
                    delta_size = 1
                    
                delta_energy = -delta_size + penalty * delta_conf
                
                accept = False
                if delta_energy < 0:
                    accept = True
                else:
                    if t > 1e-9:
                        accept = random.random() < pow(2.718281828, -delta_energy / t)
                
                if accept:
                    S ^= bit
                    current_conf += delta_conf
                    current_size += delta_size
                    
                    if current_conf == 0 and current_size > best_size:
                        best_size = current_size
                        best_set = S
            else:
                # 2-swap: remove one bit, add another
                if current_size < 2:
                    continue
                    
                # Pick a bit to remove
                temp = S
                lsb1 = temp & -temp
                idx1 = lsb1.bit_length() - 1
                temp ^= lsb1
                if temp == 0:
                    continue
                lsb2 = temp & -temp
                idx2 = lsb2.bit_length() - 1
                
                # Pick a bit to add (not in S)
                not_in_S = ((1 << m) - 1) ^ S
                if not_in_S == 0:
                    continue
                lsb3 = not_in_S & -not_in_S
                idx3 = lsb3.bit_length() - 1
                bit3 = 1 << idx3
                
                # Calculate delta for removing idx1, idx2 and adding idx3
                # This is complex; do it carefully
                # Remove idx1: delta_conf1 = -neigh1, delta_size1 = -1
                # Remove idx2: delta_conf2 = -neigh2, delta_size2 = -1
                # Add idx3: delta_conf3 = neigh3, delta_size3 = 1
                # But neighbors change after removals!
                # For simplicity, skip 2-swap or do it approximately
                
                # Actually, let's just do a simpler 2-swap: remove one, add one
                # Pick random idx to remove
                temp = S
                lsb1 = temp & -temp
                idx1 = lsb1.bit_length() - 1
                bit1 = 1 << idx1
                
                # Pick random idx3 to add
                not_in_S = ((1 << m) - 1) ^ S
                if not_in_S == 0:
                    continue
                lsb3 = not_in_S & -not_in_S
                idx3 = lsb3.bit_length() - 1
                bit3 = 1 << idx3
                
                # Remove idx1, add idx3
                # Delta: remove idx1 (size -1, conf -neigh1), add idx3 (size +1, conf +neigh3)
                # But neigh3 should be calculated after removing idx1
                # Approximate: use current state for both
                neigh1 = (S & adj[idx1]).bit_count()
                # After removing idx1, S' = S ^ bit1
                S_prime = S ^ bit1
                neigh3 = (S_prime & adj[idx3]).bit_count()
                
                delta_conf = -neigh1 + neigh3
                delta_size = 0  # Remove one, add one
                
                delta_energy = -delta_size + penalty * delta_conf
                
                accept = False
                if delta_energy < 0:
                    accept = True
                else:
                    if t > 1e-9:
                        accept = random.random() < pow(2.718281828, -delta_energy / t)
                
                if accept:
                    S = S_prime | bit3
                    current_conf += delta_conf
                    # current_size unchanged
                    
                    if current_conf == 0 and current_size > best_size:
                        best_size = current_size
                        best_set = S
    
    # Run multiple restarts
    start_time = time.time()
    timeout = 550  # Safety margin before 630s limit
    
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        S = greedy_init()
        
        # Alternate between different SA strategies
        if restart_count % 3 == 0:
            run_sa_with_swaps(S, 100000, 15.0, 0.01)
        else:
            run_sa(S, 150000, 10.0, 0.1)
        
        # Local search polish
        S_polished = local_improve(S, 30000)
        conf = calc_conflicts(S_polished)
        size = S_polished.bit_count()
        if conf == 0 and size > best_size:
            best_size = size
            best_set = S_polished
        
        # Greedy fill on best_set
        changed = True
        while changed:
            changed = False
            for n in range(m):
                if not (best_set & (1 << n)) and (best_set & adj[n]) == 0:
                    best_set |= (1 << n)
                    best_size += 1
                    changed = True
                    break
    
    # Convert best_set to array
    res = []
    for i in range(m):
        if best_set & (1 << i):
            res.append(i)
            
    return np.array(res, dtype=np.int64)
