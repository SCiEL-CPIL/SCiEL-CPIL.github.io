"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# Precompute all 65536 zero-one inputs as a (65536, 16) uint8 array
_ALL_INPUTS = np.array([[ (mask >> i) & 1 for i in range(16) ] for mask in range(65536)], dtype=np.uint8)

def _count_violations(network):
    """
    Count the number of zero-one inputs that are NOT sorted after applying the network.
    """
    state = _ALL_INPUTS.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
    diffs = state[:, 1:] - state[:, :-1]
    violated = np.any(diffs < 0, axis=1)
    return int(np.sum(violated))

def _random_layer():
    """Generate a random non-overlapping set of 8 comparators (a perfect matching)."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for k in range(N // 2):
        i, j = wires[2*k], wires[2*k+1]
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 16 wires.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_network = odd_even_transposition()
    best_violations = _count_violations(best_network)
    assert best_violations == 0, "Odd-even transposition should sort all inputs"
    
    def try_reduce(network, deadline):
        """Try to remove redundant comparators from the network."""
        network = list(network)
        changed = True
        while changed:
            changed = False
            for idx in range(len(network)):
                if time.time() > deadline:
                    return network
                candidate = network[:idx] + network[idx+1:]
                if _count_violations(candidate) == 0:
                    network = candidate
                    changed = True
                    break
        return network
    
    reduced = try_reduce(best_network, deadline)
    if len(reduced) < len(best_network):
        best_network = reduced
    
    best_len = len(best_network)
    best_net = best_network
    
    # Try random layer-based networks
    for num_layers in range(8, min(16, best_len // 8 + 2)):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            ind = [_random_layer() for _ in range(num_layers)]
            network = [comp for layer in ind for comp in layer]
            v = _count_violations(network)
            if v == 0 and len(network) < best_len:
                best_len = len(network)
                best_net = network
                reduced2 = try_reduce(best_net, deadline)
                if len(reduced2) < best_len:
                    best_len = len(reduced2)
                    best_net = reduced2
    
    if _count_violations(best_net) != 0:
        best_net = odd_even_transposition()
    
    return best_net
# EVOLVE-BLOCK-END
