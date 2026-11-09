import numpy as np
import time

N = 17

def _generate_inputs():
    bits = np.arange(17)
    inputs = (np.arange(2**17)[:, None] >> bits) & 1
    return inputs

def _verify_network(net, inputs):
    state = inputs.copy()
    for i, j in net:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return np.all(state[:, :-1] <= state[:, 1:])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Seed Network
    net = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N-1, 2):
                net.append((i, i+1))
        else:
            for i in range(1, N-1, 2):
                net.append((i, i+1))
                
    inputs = _generate_inputs()
    
    # Greedy Removal with Forward/Reverse passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Forward pass
        for idx in range(len(net)):
            if time.time() > deadline:
                break
            candidate_net = net[:idx] + net[idx+1:]
            if _verify_network(candidate_net, inputs):
                net = candidate_net
                improved = True
                break
        
        if improved or time.time() > deadline:
            continue
            
        # Reverse pass
        for idx in range(len(net)-1, -1, -1):
            if time.time() > deadline:
                break
            candidate_net = net[:idx] + net[idx+1:]
            if _verify_network(candidate_net, inputs):
                net = candidate_net
                improved = True
                break

    return net
