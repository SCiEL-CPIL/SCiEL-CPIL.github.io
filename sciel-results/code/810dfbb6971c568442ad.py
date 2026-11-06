import numpy as np
import random
import time

N = 18
TOTAL_INPUTS = 1 << N
TIME_LIMIT = 85.0

def generate_inputs():
    indices = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    inputs = np.zeros((TOTAL_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (indices >> i) & 1
    return inputs

INPUTS = generate_inputs()

def verify_full(net):
    if not net:
        return N <= 1
    state = INPUTS.copy()
    for i, j in net:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    diffs = state[:, :-1] - state[:, 1:]
    return np.all(diffs <= 0)

def construct():
    start_time = time.time()
    
    # Seed: Odd-Even Transposition Network
    curr_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    curr_len = len(curr_net)
    best_net = curr_net.copy()
    best_len = curr_len
    
    if not verify_full(curr_net):
        return curr_net

    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() - start_time < TIME_LIMIT and T > min_T:
        new_net = curr_net.copy()
        move_type = random.random()
        
        if move_type < 0.5:
            if len(new_net) > 1:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
        elif move_type < 0.8:
            i = random.randint(0, N - 1)
            j = random.randint(i + 1, N - 1)
            new_net.append((i, j))
        elif move_type < 0.9:
            if len(new_net) > 1:
                i = random.randint(0, len(new_net) - 1)
                j = random.randint(0, len(new_net) - 1)
                if i != j:
                    new_net[i], new_net[j] = new_net[j], new_net[i]
        else:
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                i = random.randint(0, N - 1)
                j = random.randint(i + 1, N - 1)
                new_net[idx] = (i, j)
        
        new_len = len(new_net)
        
        if verify_full(new_net):
            delta = new_len - curr_len
            if delta < 0:
                curr_net = new_net
                curr_len = new_len
                if new_len < best_len:
                    best_net = new_net.copy()
                    best_len = new_len
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    curr_net = new_net
                    curr_len = new_len
        # else: reject
        
        T *= cooling_rate
        
    return best_net
