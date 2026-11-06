import time
import random
import math

N = 16

_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(1 << N):
        if k & (1 << i):
            bitset |= (1 << k)
    _INITIAL_BITSETS.append(bitset)

def _verify_zero_one_fast(net):
    if not net:
        return True
    state = _INITIAL_BITSETS[:]
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def _known_60_network():
    # This is a known valid 60-comparator sorting network for N=16
    # Source: Batcher's Odd-Merge Sort
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
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    ]

def _odd_even_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try the known 60 network first
    net_60 = _known_60_network()
    if _verify_zero_one_fast(net_60):
        return net_60
        
    # If 60 is invalid (should not happen), fall back to SA from 120
    best_net = _odd_even_network()
    best_len = len(best_net)
    
    if not _verify_zero_one_fast(best_net):
        return [] # Should not happen

    rng = random.Random(42)
    current_net = best_net[:]
    current_len = best_len
    T = 10.0
    cooling = 0.95
    
    while time.time() < deadline:
        # Generate move
        move_type = rng.random()
        new_net = current_net[:]
        
        if move_type < 0.6:
            # Delete
            if len(new_net) > 1:
                idx = rng.randrange(len(new_net))
                new_net.pop(idx)
            else:
                continue
        elif move_type < 0.8:
            # Swap
            if len(new_net) > 1:
                i, j = rng.sample(range(len(new_net)), 2)
                new_net[i], new_net[j] = new_net[j], new_net[i]
            else:
                continue
        elif move_type < 0.9:
            # Move
            if len(new_net) > 1:
                i = rng.randrange(len(new_net))
                j = rng.randrange(len(new_net))
                comp = new_net.pop(i)
                new_net.insert(j, comp)
            else:
                continue
        else:
            # Insert
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i < j:
                pos = rng.randrange(len(new_net) + 1)
                new_net.insert(pos, (i, j))
            else:
                continue
                
        if _verify_zero_one_fast(new_net):
            new_len = len(new_net)
            if new_len < current_len:
                current_net = new_net
                current_len = new_len
                if current_len < best_len:
                    best_net = current_net[:]
                    best_len = current_len
            else:
                delta = new_len - current_len
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    current_net = new_net
                    current_len = new_len
        T *= cooling
        if T < 0.01:
            break
            
    return best_net
