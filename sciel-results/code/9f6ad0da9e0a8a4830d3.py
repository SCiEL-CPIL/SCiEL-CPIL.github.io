import numpy as np
import time
import random

N = 17
M = 1 << N

# Precompute sorted targets
def _precompute():
    targets = np.zeros(M, dtype=np.int32)
    for s in range(M):
        k = bin(s).count('1')
        if k == 0:
            targets[s] = 0
        elif k == N:
            targets[s] = (1 << N) - 1
        else:
            targets[s] = ((1 << k) - 1) << (N - k)
    return targets

sorted_targets = _precompute()
states_all = np.arange(M, dtype=np.int32)

def apply_network(network, states):
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap_mask = (bi > bj)
        if np.any(swap_mask):
            states[swap_mask] ^= (1 << i) | (1 << j)
    return states

def is_valid(network):
    curr_states = states_all.copy()
    apply_network(network, curr_states)
    return np.array_equal(curr_states, sorted_targets)

def get_initial_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_network = get_initial_network()
    best_size = len(best_network)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    current_network = best_network[:]
    current_size = best_size
    
    def minimize(net, max_iters=10):
        for _ in range(max_iters):
            if len(net) == 0: break
            idxs = list(range(len(net)))
            random.shuffle(idxs)
            removed = False
            for idx in idxs:
                if idx >= len(net): continue
                test_net = net[:idx] + net[idx+1:]
                if is_valid(test_net):
                    net = test_net
                    removed = True
                    break
            if not removed:
                break
        return net

    while time.time() < deadline:
        # 1. Try to minimize current network
        minimized_current = minimize(current_network, max_iters=5)
        if len(minimized_current) < current_size:
            current_network = minimized_current
            current_size = len(current_network)
            if current_size < best_size:
                best_network = current_network[:]
                best_size = current_size
                # Deep minimize if improved
                minimized_current = minimize(current_network, max_iters=30)
                if len(minimized_current) < current_size:
                    current_network = minimized_current
                    current_size = len(current_network)
                    if current_size < best_size:
                        best_network = current_network[:]
                        best_size = current_size

        # 2. Perturbation: Insert a random comparator
        pos = random.randint(0, current_size)
        cmp = random.choice(all_comparators)
        temp_network = current_network[:pos] + [cmp] + current_network[pos:]
        
        # 3. Minimize the perturbed network
        minimized_temp = minimize(temp_network, max_iters=30)
        
        # 4. Accept if better or same size
        if len(minimized_temp) < current_size:
            current_network = minimized_temp
            current_size = len(current_network)
            if current_size < best_size:
                best_network = current_network[:]
                best_size = current_size
        elif len(minimized_temp) == current_size:
            if random.random() < 0.3:
                current_network = minimized_temp

    return best_network
