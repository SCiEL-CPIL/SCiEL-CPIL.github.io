"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0
    
    # 1. Input Generation
    # Generate all 2^N boolean inputs
    # We use the periodic tiling method for speed
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    # 2. Fast Verification Setup
    # Select a subset of "hard" inputs for quick rejection
    # We pick 1024 random inputs. In practice, any random subset works well 
    # to filter out obviously broken networks.
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)
    hard_indices = rng.choice(1 << N, size=1024, replace=False)
    hard_inputs = inputs[hard_indices]
    
    state_buffer_full = np.empty_like(inputs)
    state_buffer_hard = np.empty_like(hard_inputs)
    
    def verify_full(net):
        """Verify against all 2^N inputs. Returns True if sorted."""
        np.copyto(state_buffer_full, inputs)
        for i, j in net:
            mi = state_buffer_full[:, i] & state_buffer_full[:, j]
            ma = state_buffer_full[:, i] | state_buffer_full[:, j]
            state_buffer_full[:, i] = mi
            state_buffer_full[:, j] = ma
        # Check sortedness: no '10' pattern
        # state[:, :-1] <= state[:, 1:]
        return np.all(state_buffer_full[:, :-1] <= state_buffer_full[:, 1:])

    def verify_hard(net):
        """Verify against the subset of hard inputs. Fast filter."""
        np.copyto(state_buffer_hard, hard_inputs)
        for i, j in net:
            mi = state_buffer_hard[:, i] & state_buffer_hard[:, j]
            ma = state_buffer_hard[:, i] | state_buffer_hard[:, j]
            state_buffer_hard[:, i] = mi
            state_buffer_hard[:, j] = ma
        return np.all(state_buffer_hard[:, :-1] <= state_buffer_hard[:, 1:])

    def _odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Start with odd-even transposition
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    # Verify initial network
    if not verify_full(best_network):
        return best_network

    def get_candidate(net, rng):
        """
        Generate a candidate network using Remove, Swap, or Move operations.
        Returns None if no valid move is generated.
        """
        if not net:
            return None
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove a comparator
            idx = rng.randint(len(net))
            candidate = list(net)
            candidate.pop(idx)
            return candidate
            
        elif move_type < 0.7:
            # Swap: Change one comparator to a different random one
            idx = rng.randint(len(net))
            new_i = rng.randint(N)
            new_j = rng.randint(N)
            if new_i == new_j:
                new_j = (new_j + 1) % N
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            
            # Ensure it's not already the same comparator
            # (We don't strictly need to check uniqueness for correctness, 
            # but duplicates are wasteful. However, removing duplicates 
            # is a separate optimization step. For now, allow duplicates 
            # in intermediate steps? No, better to avoid.
            # Actually, just replace. If it creates a duplicate, that's fine, 
            # it's just redundant. We can clean up later.
            
            candidate = list(net)
            candidate[idx] = (new_i, new_j)
            return candidate
            
        else:
            # Move: Remove one, insert at random position
            idx = rng.randint(len(net))
            comp = net[idx]
            candidate = list(net)
            candidate.pop(idx)
            insert_idx = rng.randint(len(candidate) + 1)
            candidate.insert(insert_idx, comp)
            return candidate

    def search(start_net, seed, time_budget):
        """
        Supercharged local search with two-stage verification.
        """
        rng = np.random.RandomState(seed)
        current = list(start_net)
        current_count = len(current)
        best_local = list(current)
        best_local_count = current_count
        
        T = 1.0
        T_min = 0.05
        cooling_rate = 0.999
        t0 = time.time()
        last_improve = t0
        
        while time.time() < t0 + time_budget and T > T_min:
            candidate = get_candidate(current, rng)
            if candidate is None:
                break
            
            # Fast check
            if not verify_hard(candidate):
                T *= cooling_rate
                continue
                
            # Full check
            if not verify_full(candidate):
                T *= cooling_rate
                continue
                
            # Candidate is valid
            cand_count = len(candidate)
            
            # Acceptance criteria
            if cand_count < current_count:
                # Always accept improvement
                current = candidate
                current_count = cand_count
                if current_count < best_local_count:
                    best_local = list(current)
                    best_local_count = current_count
                    last_improve = time.time()
                T = 1.0 # Reset temperature on improvement? Or keep cooling? 
                        # Resetting helps escape local minima of "size"
            else:
                # Same size or worse (though we rarely generate worse in size 
                # with Remove/Swap/Move unless we add, which we don't here)
                # Actually, Swap/Move keeps size same. Remove decreases.
                # So cand_count <= current_count.
                # If same size, we might accept to explore landscape.
                # Use simulated annealing probability for same-size moves
                # to allow "neutral" moves that might lead to better removals.
                if rng.random() < np.exp(-0.1 / T): # Small penalty for non-improvement
                    current = candidate
                    # current_count unchanged
                    # Check if this enables a removal? 
                    # We don't know yet. Just accept.
                # If rejected, do nothing
            
            T *= cooling_rate
            
            # Stagnation check
            if time.time() - last_improve > 3.0:
                # Restart from best local
                current = list(best_local)
                current_count = best_local_count
                T = 1.0
                last_improve = time.time()
                
        return best_local, best_local_count

    # Multi-restart search
    global_seed = 1
    while time.time() < deadline:
        # Determine time budget for this restart
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Shorter restarts initially, longer later? 
        # Or fixed 5s restarts?
        # Let's do 5s restarts to allow exploration of different basins.
        time_budget = min(5.0, time_left)
        
        # Start from best known global, or odd-even for diversity?
        # Starting from best known is safer.
        start_net = best_network
        
        cand_net, cand_count = search(start_net, global_seed, time_budget)
        
        if cand_count < best_count:
            best_network = cand_net
            best_count = cand_count
            
        global_seed += 1

    return best_network

# EVOLVE-BLOCK-END
