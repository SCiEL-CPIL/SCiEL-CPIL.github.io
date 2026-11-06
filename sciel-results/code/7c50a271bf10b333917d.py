import time
import numpy as np

N = 18
TIME_LIMIT = 55  # Safety margin

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Fill states
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w][block_idx] |= mask

    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        # Check sorted
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net):
        """Remove redundant comparators by trying to delete each one."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 5:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    np.random.seed(42)
    
    # Local Search with more aggressive moves
    while time.time() < deadline - 10:
        # Move 1: Try to remove a random comparator
        if len(current_network) > 0 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network))
            trial = current_network[:idx] + current_network[idx+1:]
            if _fast_verify(trial):
                current_network = trial
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                # After successful removal, try greedy reduce
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)

        # Move 2: Swap adjacent comparators
        if len(current_network) > 1 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

        # Move 3: Replace a comparator with a different one
        if len(current_network) > 0 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            # Pick a new pair that's likely to be useful
            # Try nearby wires first
            candidates = []
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < N and ni != nj:
                        if ni > nj:
                            ni, nj = nj, ni
                        if (ni, nj) != (i, j):
                            candidates.append((ni, nj))
            
            if candidates:
                new_pair = np.random.choice(candidates)
                trial = current_network.copy()
                trial[idx] = new_pair
                if _fast_verify(trial):
                    current_network = trial
                    reduced = greedy_reduce(current_network)
                    if len(reduced) < len(current_network):
                        current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)

        # Move 4: Insert a new comparator at a random position, then reduce
        if len(current_network) < 150 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network) + 1)
            # Choose a comparator that might help
            a, b = np.random.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
            reduced = greedy_reduce(trial)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

    return best_network
