    def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs as numpy arrays for vectorization
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Convert to numpy arrays for faster access
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorization."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        bad = ((c0 == c1) & (c2 == c3)) | \
              ((c0 == c2) & (c1 == c3)) | \
              ((c0 == c3) & (c1 == c2))
        return int(np.sum(bad))

    def get_violation_indices(colors):
        """Return a list of indices involved in bad APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        bad = ((c0 == c1) & (c2 == c3)) | \
              ((c0 == c2) & (c1 == c3)) | \
              ((c0 == c3) & (c1 == c2))
        
        if not np.any(bad):
            return []
            
        bad_aps = aps[bad]
        # Flatten and get unique indices
        return list(set(bad_aps.flatten().tolist()))

    def compute_delta(colors, i, old_color, new_color):
        """Compute delta in bad count when changing colors[i] from old_color to new_color."""
        ap_indices = idx_to_aps_arr[i]
        if len(ap_indices) == 0:
            return 0
        
        sub_aps = aps[ap_indices]
        
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        pos_mask_0 = (sub_aps[:, 0] == i)
        pos_mask_1 = (sub_aps[:, 1] == i)
        pos_mask_2 = (sub_aps[:, 2] == i)
        pos_mask_3 = (sub_aps[:, 3] == i)
        
        old_bad = ((c0 == c1) & (c2 == c3)) | \
                  ((c0 == c2) & (c1 == c3)) | \
                  ((c0 == c3) & (c1 == c2))
        
        nc0 = np.where(pos_mask_0, new_color, c0)
        nc1 = np.where(pos_mask_1, new_color, c1)
        nc2 = np.where(pos_mask_2, new_color, c2)
        nc3 = np.where(pos_mask_3, new_color, c3)
        
        new_bad = ((nc0 == nc1) & (nc2 == nc3)) | \
                  ((nc0 == nc2) & (nc1 == nc3)) | \
                  ((nc0 == nc3) & (nc1 == nc2))
        
        return int(np.sum(new_bad) - np.sum(old_bad))

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        max_restarts = 200
        restart_count = 0
        
        # Generate a few structured initializations
        init_candidates = []
        init_candidates.append(np.random.randint(0, k, n).astype(np.int32))
        for p in [k, k+1, k+2, 6, 8, 12]:
             if p <= n:
                 init_candidates.append(np.array([i % p for i in range(n)], dtype=np.int32) % k)
        
        for init in init_candidates:
            if time.time() > deadline - 5:
                break
            restart_count += 1
            
            colors = init.copy()
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing
            T = 2.0
            T_min = 0.01
            cooling = 0.999
            no_improve = 0
            max_no_improve = 500
            max_iters = 50000
            iter_count = 0
            viol_indices = []
            
            for iter_count in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Update violation indices every 50 iterations
                if iter_count % 50 == 0:
                    viol_indices = get_violation_indices(colors)
                    
                # Pick index: bias towards violation indices
                if viol_indices and random.random() < 0.7:
                    i = random.choice(viol_indices)
                else:
                    i = np.random.randint(n)
                    
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                delta = compute_delta(colors, i, old_color, new_color)
                
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > max_no_improve:
                            break
                            
                T *= cooling
                if T < T_min:
                    T = T_min

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            return [int(c) for c in best_k_solution]
            
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
