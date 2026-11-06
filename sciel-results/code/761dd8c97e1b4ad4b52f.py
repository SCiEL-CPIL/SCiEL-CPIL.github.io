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

    # Precompute the sorted check mask: for each wire w, we need current[w] & ~current[w+1] == 0
    # We'll check all N-1 pairs at once by stacking them
    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        # Check sorted: for all w, current[w] & ~current[w+1] must be all zeros
        # Combine all checks into one operation
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 5
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            # Pass 1: Try removals from the end first
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > time_budget:
                    return net
                trial = net.copy()
                del trial[idx]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    break  # Restart from end after a removal
            # Pass 2: If no removal from end, try from start
            if not changed:
                for idx in range(len(net)):
                    if time.time() > time_budget:
                        return net
                    trial = net.copy()
                    del trial[idx]
                    if _fast_verify(trial):
                        net = trial
                        changed = True
                        break
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    np.random.seed(42)
    
    # Local Search with multiple move types
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10:
        # Move 1: Swap adjacent
        if len(current_network) > 1:
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
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # Move 2: Mutation (change wire pair) - with bias toward nearby wires
        if len(current_network) > 0 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            # Pick new pair - bias toward nearby wires for better locality
            a = np.random.randint(N)
            b = np.random.randint(N)
            if a == b:
                b = (a + 1) % N
            if a > b: a, b = b, a
            if a == i and b == j: continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
        # Move 3: Insert-Delete (2-opt style)
        if len(current_network) < 200 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network) + 1)
            a = np.random.randint(N)
            b = np.random.randint(N)
            if a == b:
                b = (a + 1) % N
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a,b)] + current_network[idx:]
            reduced = greedy_reduce(trial)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # Move 4: Swap two non-adjacent comparators (position swap)
        if len(current_network) > 2 and time.time() < deadline - 10:
            idx1 = np.random.randint(len(current_network))
            idx2 = np.random.randint(len(current_network))
            if idx1 != idx2:
                trial = current_network.copy()
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                if _fast_verify(trial):
                    current_network = trial
                    reduced = greedy_reduce(current_network)
                    if len(reduced) < len(current_network):
                        current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        # Move 5: 2-opt (Reverse a segment)
        if len(current_network) > 4 and time.time() < deadline - 10:
            seg_len = np.random.randint(2, min(6, len(current_network) + 1))
            idx1 = np.random.randint(len(current_network) - seg_len + 1)
            idx2 = idx1 + seg_len - 1
            trial = current_network.copy()
            trial[idx1:idx2+1] = reversed(trial[idx1:idx2+1])
            reduced = greedy_reduce(trial)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # Move 6: If stuck, try a more aggressive mutation - change multiple comparators
        if no_improve_count > max_no_improve and time.time() < deadline - 15:
            # Change 2-3 random comparators at once
            num_changes = np.random.randint(2, 4)
            trial = current_network.copy()
            for _ in range(num_changes):
                idx = np.random.randint(len(trial))
                a = np.random.randint(N)
                b = np.random.randint(N)
                if a == b:
                    b = (a + 1) % N
                if a > b: a, b = b, a
                trial[idx] = (a, b)
            if _fast_verify(trial):
                reduced = greedy_reduce(trial)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    no_improve_count = 0
                else:
                    no_improve_count = 0  # Reset to avoid infinite loop

    return best_network
