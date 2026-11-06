import time
import random
import math
import numpy as np

def construct():
    N = 256
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # NumPy array for vectorized full evaluation
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_mono_np(x):
        """Vectorized count using NumPy. x is np.array of 0s and 1s."""
        vals = x[ap_indices]  # shape (num_aps, 4)
        prod1 = np.prod(vals, axis=1)
        prod0 = np.prod(1 - vals, axis=1)
        return int(np.sum(prod1 + prod0))
    
    # Precompute link_info for fast delta_flip (single bit)
    link_info = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for pos, idx in enumerate(ap):
            if pos == 0:
                others = (ap[1], ap[2], ap[3])
            elif pos == 1:
                others = (ap[0], ap[2], ap[3])
            elif pos == 2:
                others = (ap[0], ap[1], ap[3])
            else:
                others = (ap[0], ap[1], ap[2])
            link_info[idx].append(others)
            
    # Precompute bit_to_ap_indices for pair flips
    bit_to_ap_indices = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            bit_to_ap_indices[idx].append(ap_idx)

    def count_mono(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count
    
    def delta_flip(colors, k):
        old_c = colors[k]
        new_c = 1 - old_c
        delta = 0
        for others in link_info[k]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == new_c:
                delta += 1
            if c1 == c2 == c3 == old_c:
                delta -= 1
        return delta
        
    def eval_pair_flip(colors, i, j):
        old_i = colors[i]
        old_j = colors[j]
        
        # Get AP indices containing i or j
        # Use a set to avoid duplicates
        affected_ap_set = set(bit_to_ap_indices[i])
        affected_ap_set.update(bit_to_ap_indices[j])
        
        delta = 0
        for ap_idx in affected_ap_set:
            ap = aps[ap_idx]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            is_mono_old = (c0 == c1 == c2 == c3)
            
            # Apply flips
            # We can just check the new values
            n0 = c0
            n1 = c1
            n2 = c2
            n3 = c3
            
            if ap[0] == i: n0 = 1 - n0
            elif ap[0] == j: n0 = 1 - n0
            
            if ap[1] == i: n1 = 1 - n1
            elif ap[1] == j: n1 = 1 - n1
            
            if ap[2] == i: n2 = 1 - n2
            elif ap[2] == j: n2 = 1 - n2
            
            if ap[3] == i: n3 = 1 - n3
            elif ap[3] == j: n3 = 1 - n3
            
            is_mono_new = (n0 == n1 == n2 == n3)
            
            if is_mono_old and not is_mono_new:
                delta -= 1
            elif not is_mono_old and is_mono_new:
                delta += 1
                
        return delta

    # Greedy Construction
    def greedy_construct():
        colors = [0] * N
        # Randomize order of bits to avoid bias
        order = list(range(N))
        random.shuffle(order)
        
        # Start with all 0s? Or random?
        # Let's start with random 0/1 and then refine?
        # Or start with all 0s and flip greedily?
        # If all 0s, count is huge.
        # Let's start with Thue-Morse or Random and then do a greedy pass?
        
        # Actually, a good greedy approach:
        # Initialize randomly.
        # Then, for each bit in random order, decide if flipping it improves the local cost?
        # This is just local search.
        
        # Better Greedy:
        # Build the coloring bit by bit? No, APs span the whole range.
        
        # Let's stick to: Random Init + Local Search.
        # But we can use a "Greedy Init" that minimizes conflicts sequentially?
        # Not straightforward for global APs.
        
        # So, we will use multiple random inits + ILS.
        pass

    # Local Search with Single and Pair Flips
    def local_search(colors, cost, max_iters=10000, time_limit=None, start_time=None):
        if start_time is None:
            start_time = time.time()
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            if time_limit and time.time() > start_time + time_limit:
                break
            improved = False
            iterations += 1
            
            # Try single flips
            for i in range(N):
                d = delta_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    cost += d
                    improved = True
            if improved:
                continue
                
            # Try pair flips
            # To speed up, we can randomly sample pairs or iterate through all?
            # All pairs is N^2/2 = 32k pairs. Each eval is fast.
            # Let's try all pairs.
            for i in range(N):
                if time_limit and time.time() > start_time + time_limit:
                    break
                for j in range(i + 1, N):
                    d = eval_pair_flip(colors, i, j)
                    if d < 0:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        cost += d
                        improved = True
                        break # Restart inner loop? Or continue?
                    # If we find an improvement, we can break and restart the whole local search
                    # to avoid getting stuck in a sequence of small improvements that lead to a worse state later?
                    # No, greedy descent is fine.
                if improved:
                    break
                    
        return colors, cost

    # Iterated Local Search
    def iterated_local_search(init_colors, init_cost, time_limit):
        start_time = time.time()
        best_colors = init_colors[:]
        best_cost = init_cost
        
        current_colors = init_colors[:]
        current_cost = init_cost
        
        while time.time() < start_time + time_limit:
            # Local Search
            current_colors, current_cost = local_search(current_colors, current_cost, time_limit=time_limit, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors[:]
                
            # Perturb
            # Flip a random number of bits
            num_flips = random.randint(5, 15)
            indices = random.sample(range(N), num_flips)
            for idx in indices:
                current_colors[idx] = 1 - current_colors[idx]
            current_cost = count_mono(current_colors)
            
        return best_colors, best_cost

    # Main
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate diverse initializations
    inits = []
    
    # Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    tm = [thue_morse(i) for i in range(N)]
    inits.append(tm)
    
    # Random
    random.seed(42)
    for _ in range(10):
        inits.append([random.randint(0, 1) for _ in range(N)])
        
    # Block patterns
    for block_size in [4, 8, 16, 32, 64]:
        inits.append([(i // block_size) % 2 for i in range(N)])
        
    # Evaluate and pick best init
    best_init = None
    best_init_cost = float('inf')
    for init in inits:
        cost = count_mono(init)
        if cost < best_init_cost:
            best_init_cost = cost
            best_init = init[:]
            
    global_best_colors = best_init[:]
    global_best_cost = best_init_cost
    
    # Run ILS for the remaining time
    # We can run ILS on the best init, or multiple inits?
    # Let's run ILS on the best init, and maybe restart if it stagnates?
    # ILS itself restarts.
    
    # Split time among a few different ILS runs starting from different inits?
    # Let's just run one long ILS on the best init.
    # Or, run ILS on top 3 inits in parallel? No, sequential.
    
    # Let's run ILS on the best init for 80% of time, then polish the rest.
    time_for_ils = (deadline - start_time) * 0.9
    global_best_colors, global_best_cost = iterated_local_search(best_init, best_init_cost, time_for_ils)
    
    # Final Polish: Greedy single flips
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for pos in range(N):
                d = delta_flip(global_best_colors, pos)
                if d < 0:
                    global_best_colors[pos] = 1 - global_best_colors[pos]
                    global_best_cost += d
                    improved = True
                    
    return global_best_colors
