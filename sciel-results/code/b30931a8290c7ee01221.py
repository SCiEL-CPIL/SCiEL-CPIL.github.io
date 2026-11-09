def construct():
    import time
    import numpy as np
    import random
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights for all 17-bit vectors
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)
    
    def construct_qr_code():
        """
        Construct the [17,8,6] code by shortening the [18,9,6] extended QR code.
        """
        p = 17
        # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
        QR = set((i * i) % p for i in range(1, p))
        
        # Idempotent vector
        v = np.zeros(p, dtype=np.uint8)
        for q in QR:
            v[q] = 1
        
        # Generator matrix: 8 cyclic shifts
        G17 = np.zeros((8, p), dtype=np.uint8)
        for k in range(8):
            G17[k] = np.roll(v, k)
        
        # Add parity bit to get [18,8,6]
        parities = G17.sum(axis=1) % 2
        G18_8 = np.column_stack((G17, parities))
        
        # Add all-ones row for [18,9,6]
        ones = np.ones(18, dtype=np.uint8)
        G18_9 = np.vstack((G18_8, ones))
        
        # Generate all 512 codewords
        code = np.zeros((512, 18), dtype=np.uint8)
        for m in range(512):
            bits = np.array([(m >> j) & 1 for j in range(9)], dtype=np.uint8)
            code[m] = (bits @ G18_9) % 2
        
        # Shorten: keep words where last bit is 0
        mask = code[:, 17] == 0
        short_code = code[mask, :17]
        
        # Convert to integers
        powers = (1 << np.arange(N, dtype=np.int64))[::-1]
        code_list = (short_code @ powers).astype(np.int64).tolist()
        
        return code_list
    
    # Phase 1: Get the 256-word linear code
    code_list = construct_qr_code()
    C = code_list
    C_set = set(C)
    C_np = np.array(C, dtype=np.int32)
    
    # Phase 2: Find all valid candidates for extension
    # A vector w is valid if d(w, c) >= 6 for all c in C
    valid_candidates = []
    
    t_start = time.time()
    time_limit_phase2 = t_start + 15
    
    # Vectorized approach: process in chunks
    # For each candidate w, compute min distance to all C
    # Use numpy broadcasting in chunks to avoid memory issues
    
    chunk_size = 4096
    for start in range(0, SIZE, chunk_size):
        if time.time() > time_limit_phase2:
            break
        end = min(start + chunk_size, SIZE)
        candidates_chunk = np.arange(start, end, dtype=np.int32)
        
        # Skip if any are in C
        mask_not_in_c = np.array([c not in C_set for c in candidates_chunk])
        candidates_chunk = candidates_chunk[mask_not_in_c]
        
        if len(candidates_chunk) == 0:
            continue
        
        # Compute distances: candidates_chunk[:, None] ^ C_np[None, :]
        # This creates a (len(chunk) x 256) array of XORs
        # Then compute weights (popcount)
        
        # To save memory, process in smaller sub-chunks if needed
        sub_chunk_size = 1024
        for s in range(0, len(candidates_chunk), sub_chunk_size):
            if time.time() > time_limit_phase2:
                break
            e = min(s + sub_chunk_size, len(candidates_chunk))
            sub = candidates_chunk[s:e]
            
            # XOR with all codewords
            xor_matrix = sub[:, None] ^ C_np[None, :]
            
            # Compute popcount for each element
            # Use bit manipulation
            counts = np.zeros((len(sub), len(C_np)), dtype=np.int8)
            for bit in range(N):
                counts += (xor_matrix >> bit) & 1
            
            # Find minimum distance for each candidate
            min_dists = counts.min(axis=1)
            
            # Valid if min_dist >= MIN_D
            valid_mask = min_dists >= MIN_D
            valid_candidates.extend(sub[valid_mask].tolist())
    
    if not valid_candidates:
        return C
    
    V = valid_candidates
    V_len = len(V)
    
    if time.time() > deadline - 5:
        return C
    
    # Phase 3: Find Maximum Independent Set
    # Vertices: V, Edge: d(u,v) < 6
    
    # Build adjacency using weights table
    adj = [set() for _ in range(V_len)]
    
    t_start3 = time.time()
    time_limit_phase3 = t_start3 + (deadline - 3 - t_start3)
    
    # Build graph - O(V^2) but V should be manageable
    for i in range(V_len):
        if time.time() > time_limit_phase3:
            break
        u = V[i]
        for j in range(i + 1, V_len):
            v = V[j]
            if weights[u ^ v] < MIN_D:
                adj[i].add(j)
                adj[j].add(i)
    
    # Solve MIS using branch and bound with greedy coloring
    # This is a structural change: use a proper MIS solver instead of pure greedy
    
    def greedy_coloring(vertices, adj):
        """Greedy coloring for upper bound."""
        color = {}
        colors_used = 0
        for v in vertices:
            neighbor_colors = set()
            for nb in adj[v]:
                if nb in color:
                    neighbor_colors.add(color[nb])
            c = 0
            while c in neighbor_colors:
                c += 1
            color[v] = c
            colors_used = max(colors_used, c + 1)
        return colors_used, color
    
    def branch_and_bound_mis(vertices, adj, time_limit):
        """
        Branch and bound for Maximum Independent Set.
        vertices: set of vertices to consider
        Returns the MIS found.
        """
        best_is = set()
        best_size = 0
        
        # Sort vertices by degree for better pruning
        def get_degree(v, current_set):
            return len(adj[v] & current_set)
        
        def search(current_is, candidates, upper_bound):
            nonlocal best_is, best_size
            
            if time.time() > time_limit:
                return
            
            if not candidates:
                if len(current_is) > best_size:
                    best_size = len(current_is)
                    best_is = set(current_is)
                return
            
            # Upper bound: current_is size + coloring bound
            color_bound, _ = greedy_coloring(candidates, adj)
            if len(current_is) + color_bound <= best_size:
                return
            
            # Pick a vertex with minimum degree (or maximum, depending on strategy)
            # Use minimum degree for better branching
            min_deg = float('inf')
            pivot = None
            for v in candidates:
                deg = get_degree(v, candidates)
                if deg < min_deg:
                    min_deg = deg
                    pivot = v
            
            if pivot is None:
                return
            
            # Branch 1: Include pivot
            new_is = set(current_is)
            new_is.add(pivot)
            new_candidates = candidates - (adj[pivot] | {pivot})
            search(new_is, new_candidates, len(new_is) + color_bound)
            
            # Branch 2: Exclude pivot
            new_candidates = candidates - {pivot}
            search(current_is, new_candidates, len(current_is) + color_bound)
        
        search(set(), set(vertices), greedy_coloring(vertices, adj)[0])
        return list(best_is)
    
    # Run B&B with time limit
    time_limit_bb = deadline - 2
    
    # If V is small enough, try B&B
    if V_len <= 200:
        is_indices = branch_and_bound_mis(set(range(V_len)), adj, time_limit_bb)
        best_extension = [V[i] for i in is_indices]
    else:
        # For larger V, use greedy with multiple restarts
        best_extension = []
        
        def greedy_mis(adj, strategy, time_limit):
            alive = set(range(V_len))
            is_set = set()
            
            while alive and time.time() < time_limit:
                if strategy == 'degree':
                    min_deg = float('inf')
                    best_v = -1
                    for v in alive:
                        deg = len(adj[v] & alive)
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                    v = best_v
                elif strategy == 'random':
                    v = random.choice(list(alive))
                else:
                    v = next(iter(alive))
                
                if v == -1:
                    break
                
                is_set.add(v)
                to_remove = {v} | (adj[v] & alive)
                alive -= to_remove
            
            return [V[i] for i in is_set]
        
        # Multiple restarts
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > time_limit_bb:
                break
            strategy = 'degree' if r % 2 == 0 else 'random'
            ext = greedy_mis(adj, strategy, time_limit_bb)
            if len(ext) > len(best_extension):
                best_extension = ext
        
        # Try to improve with local search
        if best_extension and time.time() < deadline - 5:
            current_set = set(best_extension)
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try adding a vertex
                for v in range(V_len):
                    if v in current_set:
                        continue
                    if all(weights[V[v] ^ V[u]] >= MIN_D for u in current_set):
                        current_set.add(v)
                        improved = True
                        break
                if not improved:
                    # Try swapping
                    for v_out in list(current_set):
                        for v_in in range(V_len):
                            if v_in in current_set:
                                continue
                            if all(weights[V[v_in] ^ V[u]] >= MIN_D for u in current_set - {v_out}):
                                current_set.remove(v_out)
                                current_set.add(v_in)
                                improved = True
                                break
                        if improved:
                            break
            best_extension = list(current_set)
    
    final_code = C + best_extension
    
    # Verify all pairs have distance >= MIN_D (sample check)
    if len(final_code) > 1:
        sample_size = min(100, len(final_code))
        random_sample = random.sample(final_code, sample_size)
        for i in range(len(random_sample)):
            for j in range(i + 1, len(random_sample)):
                if weights[random_sample[i] ^ random_sample[j]] < MIN_D:
                    # Invalid, fall back to just C
                    return C
    
    return final_code
