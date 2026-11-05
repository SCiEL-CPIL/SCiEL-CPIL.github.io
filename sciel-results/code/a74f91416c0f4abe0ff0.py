import time
import random

N = 16

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    """
    if not net:
        return True
    
    # Precompute initial bitsets
    if not hasattr(_verify_zero_one_fast, '_INITIAL_BITSETS'):
        _INITIAL_BITSETS = []
        for i in range(N):
            bitset = 0
            for k in range(1 << N):
                if k & (1 << i):
                    bitset |= (1 << k)
            _INITIAL_BITSETS.append(bitset)
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s margin
    
    base_net = _base_60_network()
    
    if not _verify_zero_one_fast(base_net):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = base_net
    best_len = len(base_net)
    
    # We will run multiple independent search sessions
    num_sessions = 10
    time_per_session = (deadline - start_time) / num_sessions
    
    for session in range(num_sessions):
        if time.time() > deadline:
            break
            
        current_net = base_net[:]
        rng = random.Random(session * 12345 + 6789)
        
        session_start = time.time()
        session_deadline = session_start + time_per_session
        
        while time.time() < session_deadline:
            # Try to remove a comparator
            if len(current_net) > 1:
                idx = rng.randrange(len(current_net))
                new_net = current_net[:idx] + current_net[idx + 1:]
                if _verify_zero_one_fast(new_net):
                    current_net = new_net
                    if len(current_net) < best_len:
                        best_net = current_net[:]
                        best_len = len(current_net)
                        # If we hit 53, we can stop early or continue to try for lower
                        if best_len <= 53:
                            break
            else:
                break
                
            # Try swapping two comparators
            if len(current_net) > 1:
                i, j = rng.sample(range(len(current_net)), 2)
                new_net = current_net[:]
                new_net[i], new_net[j] = new_net[j], new_net[i]
                if _verify_zero_one_fast(new_net):
                    current_net = new_net
                    if len(current_net) < best_len:
                        best_net = current_net[:]
                        best_len = len(current_net)
                        
            # Try inserting a comparator to escape local minima
            if rng.random() < 0.05:
                i, j = rng.sample(range(N), 2)
                if i > j: i, j = j, i
                idx = rng.randrange(len(current_net) + 1)
                new_net = current_net[:idx] + [(i, j)] + current_net[idx:]
                if _verify_zero_one_fast(new_net):
                    current_net = new_net

    return best_net
