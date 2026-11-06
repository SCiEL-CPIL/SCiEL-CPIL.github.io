import time
import numpy as np

N = 18
TIME_LIMIT = 55

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
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
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    def greedy_reduce(net):
        changed = True
        while changed and time.time() < deadline - 8:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 8:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(list(best_network))
    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)

    np.random.seed(12345)
    
    # Simulated Annealing approach
    current_temp = 50.0
    cooling_rate = 0.95
    min_temp = 0.1
    
    while time.time() < deadline - 12 and current_temp > min_temp:
        # Choose a move
        move_type = np.random.choice(4, p=[0.3, 0.3, 0.25, 0.15])
        trial = list(current_network)
        
        if move_type == 0:
            # Remove a random comparator
            if len(trial) > 0:
                idx = np.random.randint(len(trial))
                trial = trial[:idx] + trial[idx+1:]
        elif move_type == 1:
            # Swap adjacent comparators
            if len(trial) > 1:
                idx = np.random.randint(len(trial) - 1)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
        elif move_type == 2:
            # Mutate: replace a comparator
            if len(trial) > 0:
                idx = np.random.randint(len(trial))
                i, j = trial[idx]
                # Pick a random replacement
                ni, nj = np.random.choice(N, 2, replace=False)
                if ni > nj:
                    ni, nj = nj, ni
                if (ni, nj) != (i, j):
                    trial[idx] = (ni, nj)
        else:
            # Insert + reduce
            if len(trial) < 140:
                idx = np.random.randint(len(trial) + 1)
                a, b = np.random.choice(N, 2, replace=False)
                if a > b:
                    a, b = b, a
                trial = trial[:idx] + [(a, b)] + trial[idx:]
        
        # Verify trial
        if not _fast_verify(trial):
            current_temp *= cooling_rate
            continue
        
        # Accept or reject based on simulated annealing
        delta = len(trial) - len(current_network)
        if delta <= 0 or np.random.random() < np.exp(-delta / current_temp):
            current_network = trial
            if len(current_network) < best_len:
                best_network = list(current_network)
                best_len = len(current_network)
                # Try greedy reduction after improvement
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)
        
        current_temp *= cooling_rate
        
        # Periodic greedy reduction
        if time.time() < deadline - 12 and len(current_network) > 0:
            reduced = greedy_reduce(current_network)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

    return best_network
