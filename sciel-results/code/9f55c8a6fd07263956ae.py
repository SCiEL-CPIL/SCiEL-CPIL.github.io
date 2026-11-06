"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0
    
    # Fast input generation via periodic tiling
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    state_buffer = np.empty_like(inputs)
    
    def verify(net):
        np.copyto(state_buffer, inputs)
        for i, j in net:
            mi = state_buffer[:, i] & state_buffer[:, j]
            ma = state_buffer[:, i] | state_buffer[:, j]
            state_buffer[:, i] = mi
            state_buffer[:, j] = ma
        return np.all(state_buffer[:, :-1] <= state_buffer[:, 1:])
    
    def _odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    def _batcher_sort(n):
        """Generate a Batcher sorting network for n wires."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Batcher's bitonic sort
        # Split into two halves
        half = n // 2
        # Sort each half recursively
        left = _batcher_sort(half)
        right = _batcher_sort(n - half)
        
        # Adjust right half indices
        right = [(i + half, j + half) for i, j in right]
        
        # Merge using bitonic merge
        merged = _bitonic_merge(half, n - half)
        
        return left + right + merged
    
    def _bitonic_merge(k, n):
        """Bitonic merge for merging two sorted halves of sizes k and n-k."""
        comparators = []
        # Standard bitonic merge pattern
        for i in range(k):
            if i < n - k:
                comparators.append((i, i + n - k))
        # Actually, let's use a simpler approach: recursive bitonic merge
        return _bitonic_merge_recursive(0, n, k)
    
    def _bitonic_merge_recursive(start, end, split):
        """Recursive bitonic merge."""
        comparators = []
        if end - start <= 1:
            return comparators
        if split <= 0 or split >= end - start:
            # One side is empty, just sort
            return _batcher_sort_simple(start, end)
        
        # Compare elements across the split
        for i in range(split):
            if start + i < end and start + split + i < end:
                a = start + i
                b = start + split + i
                if a < b:
                    comparators.append((a, b))
        
        # Recursively sort the two resulting halves
        # Left half: start to start+split
        # Right half: start+split to end
        left = _bitonic_merge_recursive(start, start + split, split // 2)
        right = _bitonic_merge_recursive(start + split, end, (end - start - split) // 2)
        
        return comparators + left + right
    
    def _batcher_sort_simple(start, end):
        """Simple sorting network for a range."""
        comparators = []
        for i in range(start, end):
            for j in range(i + 1, end):
                comparators.append((i, j))
        return comparators
    
    # Start with odd-even transposition network
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    if not verify(best_network):
        return best_network
    
    # Also try Batcher as a starting point
    batcher_net = _batcher_sort(N)
    if verify(batcher_net):
        if len(batcher_net) < best_count:
            best_network = batcher_net
            best_count = len(batcher_net)
    
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
        T = 1.0  # initial temperature
        T_min = 0.01
        cooling_rate = 0.9995
        t0 = time.time()
        
        while time.time() < t0 + time_budget and T > T_min:
            move_type = rng.random()
            
            if move_type < 0.5 and current_count > 0:
                # Try removal
                idx = rng.randint(current_count)
                candidate = list(current)
                candidate.pop(idx)
                if verify(candidate):
                    current = candidate
                    current_count -= 1
                    if current_count < best_count:
                        best = list(current)
                        best_count = current_count
            elif move_type < 0.85:
                # Try mutation (same size)
                candidate = targeted_mutation(current, rng)
                if candidate == current:
                    continue
                if verify(candidate):
                    current = candidate
            else:
                # Try adding a comparator (worsening move) to escape local minima
                if current_count < 200:
                    new_i = rng.randint(N)
                    new_j = rng.randint(N)
                    if new_i == new_j:
                        continue
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    if (new_i, new_j) not in current:
                        candidate = list(current)
                        candidate.append((new_i, new_j))
                        if verify(candidate):
                            if rng.random() < np.exp(-1.0 / T):
                                current = candidate
                                current_count += 1
            
            T *= cooling_rate
        
        return best, best_count
    
    # Multi-restart SA with both odd-even and Batcher starts
    seed = 42
    restarts = 0
    start_options = [best_network, _odd_even_network()]
    if batcher_net:
        start_options.append(batcher_net)
    
    while time.time() < deadline:
        restarts += 1
        time_per_restart = min(8.0, max(2.0, deadline - time.time()))
        if time.time() + time_per_restart > deadline:
            time_per_restart = max(0.5, deadline - time.time())
        
        start_net = start_options[restarts % len(start_options)]
        candidate_best, candidate_count = simulated_annealing(start_net, seed, time_per_restart)
        
        if candidate_count < best_count:
            best_network = candidate_best
            best_count = candidate_count
        
        seed += 1
    
    return best_network

# EVOLVE-BLOCK-END
