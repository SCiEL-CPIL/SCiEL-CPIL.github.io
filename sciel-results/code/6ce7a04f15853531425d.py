import time
import numpy as np

N = 18
START_TIME = time.time()
TIME_LIMIT = 85  # safety margin before 90s

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        idx = np.arange(1 << N, dtype=np.uint32)
        bits = np.arange(N, dtype=np.uint32)
        _ALL_INPUTS = ((idx[:, None] >> bits[None, :]) & 1).astype(np.uint8)
    return _ALL_INPUTS

def _apply_network(network, inputs):
    result = inputs.copy()
    for i, j in network:
        a = result[:, i]
        b = result[:, j]
        result[:, i] = np.minimum(a, b)
        result[:, j] = np.maximum(a, b)
    return result

def _is_sorted_network(network):
    all_inputs = _get_all_inputs()
    outputs = _apply_network(network, all_inputs)
    diffs = np.diff(outputs, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _depth(network):
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0

def construct():
    global START_TIME
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(network)
    
    improved = True
    while improved and (time.time() - START_TIME) < TIME_LIMIT:
        improved = False
        for idx in range(len(network)):
            if (time.time() - START_TIME) > TIME_LIMIT:
                break
            test_network = network[:idx] + network[idx+1:]
            if _is_sorted_network(test_network):
                network = test_network
                best_network = list(network)
                improved = True
                break
    
    return best_network
