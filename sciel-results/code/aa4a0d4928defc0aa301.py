<<<<<<< SEARCH
def construct():
    """Search for a minimal sorting network on 16 wires using Simulated Annealing."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Start with Batcher's sort (147 comparators)
    # This is a valid starting point and has good structure for reduction
    current_net = _batcher_sort(N)
    
    # Verify start
    if not _verify(current_net):
        # Fallback to OET if Batcher fails (shouldn't happen)
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = list(current_net)
    best_len = len(current_net)
    best_time = time.time()
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.9995 # Cooling rate
    
    random.seed(42) # For reproducibility during development, remove for final if needed? No, keep it stable.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        trial_net = list(current_net)
        
        if move_type < 0.4:
            # Delete
            if len(trial_net) > 0:
                idx = random.randint(0, len(trial_net) - 1)
                trial_net.pop(idx)
        elif move_type < 0.8:
            # Insert
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = random.randint(0, len(trial_net))
            trial_net.insert(pos, (i, j))
        else:
            # Swap adjacent
            if len(trial_net) > 1:
                idx = random.randint(0, len(trial_net) - 2)
                trial_net[idx], trial_net[idx+1] = trial_net[idx+1], trial_net[idx]
        
        # Verify trial
        if _verify(trial_net):
            # Accept if valid (and we want to minimize size, so we always accept valid smaller ones, 
            # but SA logic usually applies to cost. Here cost = -size or just size? 
            # We want to minimize size. 
            # If trial is valid, it's a good candidate. 
            # If we accept larger valid networks, we might wander. 
            # Better: Only accept if len(trial_net) < len(current_net) OR with prob exp(-delta/T) where delta = len(trial)-len(current).
            # Since we want to minimize, delta = len(trial) - len(current).
            # If delta < 0 (smaller), always accept.
            # If delta > 0 (larger), accept with prob exp(-delta/T).
            
            delta = len(trial_net) - len(current_net)
            
            if delta < 0:
                current_net = trial_net
                if len(current_net) < best_len:
                    best_net = list(current_net)
                    best_len = len(current_net)
                    best_time = time.time()
            else:
                # Accept larger network with probability
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_net = trial_net
        else:
            # Invalid network. Reject.
            pass
            
        # Cool down
        T *= alpha
        
        # Restart logic: If no improvement for a while, perturb best_net
        if time.time() - best_time > 10.0:
            # Perturb best_net to escape local minima
            current_net = list(best_net)
            # Add some random noise
            for _ in range(5):
                if random.random() < 0.5:
                    # Delete
                    if len(current_net) > 0:
                        idx = random.randint(0, len(current_net) - 1)
                        current_net.pop(idx)
                else:
                    # Insert
                    i, j = random.sample(range(N), 2)
                    if i > j: i, j = j, i
                    pos = random.randint(0, len(current_net))
                    current_net.insert(pos, (i, j))
            
            # Reset temperature slightly to allow exploration
            T = max(T, 0.5)
            best_time = time.time() # Reset timer for this restart attempt? No, keep tracking global best time.
            # Actually, let's just keep the best_net as the anchor.
            
    return best_net
=======
def construct():
    """Search for a minimal sorting network on 16 wires using aggressive greedy removal."""
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin: evaluator kills at 90s
    
    # Start with bitonic sort network (119 comparators for n=16)
    current_net = _generate_bitonic_network(N)
    
    # Verify start
    if not _verify_fast(current_net):
        # Fallback to odd-even transposition
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = list(current_net)
    best_len = len(current_net)
    
    # Aggressive greedy removal: try to remove each comparator if network remains valid
    # This is a structural change from simulated annealing
    random.seed(42)
    
    # Phase 1: Greedy removal - try removing each comparator multiple times
    for iteration in range(10):  # Multiple passes
        if time.time() > deadline - 10:
            break
        
        removed_any = False
        # Shuffle order for diversity
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10:
                break
            
            if idx >= len(current_net):
                continue
            
            # Try removing this comparator
            trial_net = list(current_net)
            trial_net.pop(idx)
            
            if _verify_fast(trial_net):
                current_net = trial_net
                removed_any = True
                if len(current_net) < best_len:
                    best_net = list(current_net)
                    best_len = len(current_net)
        
        if not removed_any:
            # No more removals possible, break
            break
    
    # Phase 2: Local search with insertions to escape local minima
    if time.time() < deadline - 5:
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        
        while time.time() < deadline - 5:
            if T < T_min:
                break
            
            # Choose a move
            move_type = random.random()
            trial_net = list(current_net)
            
            if move_type < 0.3:
                # Delete
                if len(trial_net) > 0:
                    idx = random.randint(0, len(trial_net) - 1)
                    trial_net.pop(idx)
            elif move_type < 0.7:
                # Insert
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = random.randint(0, len(trial_net))
                trial_net.insert(pos, (i, j))
            else:
                # Swap adjacent
                if len(trial_net) > 1:
                    idx = random.randint(0, len(trial_net) - 2)
                    trial_net[idx], trial_net[idx+1] = trial_net[idx+1], trial_net[idx]
            
            # Verify trial
            if _verify_fast(trial_net):
                delta = len(trial_net) - len(current_net)
                
                if delta < 0:
                    current_net = trial_net
                    if len(current_net) < best_len:
                        best_net = list(current_net)
                        best_len = len(current_net)
                else:
                    # Accept larger network with probability
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_net = trial_net
            else:
                # Invalid network. Reject.
                pass
            
            # Cool down
            T *= alpha
    
    # Final verification
    if not _verify_fast(best_net):
        # Shouldn't happen, but fallback
        best_net = _generate_bitonic_network(N)
    
    return best_net
>>>>>>> REPLACE

