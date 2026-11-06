"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    # Safety margin: Evaluator kills at 90s, task requires <60s.
    # We set deadline to 55s to ensure we return well in time, leaving room for cleanup.
    deadline = time.time() + 55.0
    
    # 1. Input Generation
    # Generate all 2^N boolean inputs efficiently using periodic tiling
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    # 2. Fast Verification Setup
    # We use two levels of verification:
    # - verify_hard: checks against a subset of 1024 random inputs for quick rejection
    # - verify_full: checks against all 2^N inputs for final confirmation
    
    rng_seed = 42
    rng_init = np.random.RandomState(rng_seed)
    hard_indices = rng_init.choice(1 << N, size=1024, replace=False)
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
        # Check sortedness: state[:, :-1] <= state[:, 1:]
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
    
    # Start with odd-even transposition network
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    if not verify_full(best_network):
        return best_network
    
    # 3. Search: Simulated Annealing with hard-filter acceleration
    # Key insight: use verify_hard for most moves, verify_full only when we think we improved
    # This allows many more search steps in the same time budget.
    
    def targeted_mutation(net, rng):
        """Generate a mutated network: change one comparator to a nearby one."""
        if len(net) == 0:
            return net
        idx = rng.randint(len(net))
        old_i, old_j = net[idx]
        strategy = rng.random()
        
        if strategy < 0.4:
            new_i = old_i
            if rng.random() < 0.5:
                new_j = old_j + rng.randint(-3, 4)
            else:
                new_j = rng.randint(old_i + 1, N) if old_i + 1 < N else old_i + 1
        elif strategy < 0.8:
            new_j = old_j
            if rng.random() < 0.5:
                new_i = old_i + rng.randint(-3, 4)
            else:
                new_i = rng.randint(0, new_j) if new_j > 0 else 0
        else:
            if rng.random() < 0.3:
                new_i = 0
                new_j = rng.randint(1, N)
            elif rng.random() < 0.3:
                new_j = N - 1
                new_i = rng.randint(0, N - 1)
            else:
                new_i = rng.randint(N)
                new_j = rng.randint(N)
                if new_i == new_j:
                    new_j = (new_j + 1) % N
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
        
        new_i = max(0, min(N - 1, new_i))
        new_j = max(0, min(N - 1, new_j))
        if new_i >= new_j:
            if new_j < N - 1:
                new_j = new_j + 1
            else:
                new_i = max(0, new_i - 1)
        if new_i >= new_j:
            return net  # invalid, return unchanged
        
        candidate = list(net)
        candidate[idx] = (new_i, new_j)
        return candidate
    
    def simulated_annealing(start_net, seed, time_budget):
        """Run SA search starting from start_net for time_budget seconds."""
        rng = np.random.RandomState(seed)
        current = list(start_net)
        current_count = len(current)
        best = list(current)
        best_count = current_count
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9998
        t0 = time.time()
        last_full_verify = t0
        full_verify_interval = 2.0  # seconds between full verifications
        
        while time.time() < t0 + time_budget and T > T_min:
            move_type = rng.random()
            
            if move_type < 0.55 and current_count > 0:
                # Try removal
                idx = rng.randint(current_count)
                candidate = list(current)
                candidate.pop(idx)
                
                # First check with hard inputs (fast)
                if verify_hard(candidate):
                    # Then verify fully (slower)
                    if verify_full(candidate):
                        current = candidate
                        current_count -= 1
                        if current_count < best_count:
                            best = list(current)
                            best_count = current_count
                            last_full_verify = time.time()
                # If hard verify fails, skip full verify (saves time)
                
            elif move_type < 0.90:
                # Try mutation (same size)
                candidate = targeted_mutation(current, rng)
                if candidate == current:
                    continue
                
                # Use hard verify for mutations (same size, so no immediate gain)
                # Only accept if it passes hard verify
                if verify_hard(candidate):
                    current = candidate
                    # Occasionally do a full verify to check if this mutation enables future removals
                    if time.time() - last_full_verify > full_verify_interval:
                        if verify_full(current):
                            last_full_verify = time.time()
                        else:
                            # Revert to best if current is broken
                            current = list(best)
                            current_count = best_count
                else:
                    pass  # Reject
                
            else:
                # Try adding a comparator (worsening move) to escape local minima
                if current_count < 200:
                    new_i = rng.randint(N)
                    new_j = rng.randint(N)
                    if new_i == new_j:
                        continue
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    # Check if already present
                    if (new_i, new_j) not in current:
                        candidate = list(current)
                        candidate.append((new_i, new_j))
                        if verify_hard(candidate):
                            # Accept with probability based on temperature
                            if rng.random() < np.exp(-1.0 / T):
                                current = candidate
                                current_count += 1
            
            T *= cooling_rate
        
        # Final verification of best
        if verify_full(best):
            return best, best_count
        else:
            # Fallback to initial network
            return list(start_net), len(start_net)
    
    # Multi-restart SA
    seed = 42
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Allocate remaining time to this restart
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        time_per_restart = min(10.0, remaining * 0.8)
        
        start_net = best_network if restarts == 1 else _odd_even_network()
        candidate_best, candidate_count = simulated_annealing(start_net, seed, time_per_restart)
        
        if candidate_count < best_count:
            best_network = candidate_best
            best_count = candidate_count
        
        seed += 100
    
    # Final safety check
    if verify_full(best_network):
        return best_network
    else:
        # Fallback to odd-even if something went wrong
        fallback = _odd_even_network()
        if verify_full(fallback):
            return fallback
        return best_network

# EVOLVE-BLOCK-END
