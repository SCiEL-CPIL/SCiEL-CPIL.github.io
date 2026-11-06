"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 82.0  # seconds, safety margin from 90s limit

# EVOLVE-BLOCK-START

def _all_bool_inputs():
    """Generate all 2^N boolean inputs as a (2^N, N) uint8 array."""
    arr = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(-1, 1)
    result = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        result[:, i] = ((arr >> i) & 1).astype(np.uint8)
    return result

def _simulate(network, inputs):
    """Simulate network on all inputs. Returns final state array."""
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        mask = a > b
        if np.any(mask):
            state[:, i] = np.where(mask, b, a)
            state[:, j] = np.where(mask, a, b)
    return state

def _is_sorted(state):
    """Check if all rows are sorted (non-decreasing)."""
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k + 1]):
            return False
    return True

def _verify(network, inputs):
    """Verify network sorts all inputs."""
    if not network:
        return _is_sorted(inputs)
    final = _simulate(network, inputs)
    return _is_sorted(final)

def construct():
    start_time = time.time()
    
    # Precompute all inputs
    inputs = _all_bool_inputs()
    
    # Seed: odd-even transposition network
    best_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    # Verify seed
    assert _verify(best_network, inputs), "Seed network should be valid"
    
    best_len = len(best_network)
    
    # Phase 1: Greedy deletion - repeatedly remove any removable comparator
    changed = True
    pass_count = 0
    while changed and (time.time() - start_time) < DEADLINE - 10.0:
        changed = False
        pass_count += 1
        net = list(best_network)
        for idx in range(len(net)):
            if (time.time() - start_time) > DEADLINE - 10.0:
                break
            trial = net[:idx] + net[idx+1:]
            if _verify(trial, inputs):
                best_network = trial
                best_len = len(trial)
                changed = True
                break  # Restart scan
    
    # Phase 2: Try adding a comparator then deleting two
    if (time.time() - start_time) < DEADLINE - 5.0:
        rng = np.random.RandomState(123)
        # Try many random additions followed by deletion
        for attempt in range(500):
            if (time.time() - start_time) > DEADLINE - 5.0:
                break
            i = rng.randint(N)
            j = rng.randint(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pos = rng.randint(0, len(best_network) + 1)
            trial = best_network[:pos] + [(i, j)] + best_network[pos:]
            
            # Try to delete 2 comparators
            net = list(trial)
            deleted = 0
            for idx in range(len(net) - 1, -1, -1):
                if (time.time() - start_time) > DEADLINE - 5.0:
                    break
                candidate = net[:idx] + net[idx+1:]
                if _verify(candidate, inputs):
                    net = candidate
                    deleted += 1
                    if deleted >= 2:
                        break
            
            if len(net) < len(best_network):
                best_network = net
                best_len = len(net)
    
    # Phase 3: Final deletion pass
    if (time.time() - start_time) < DEADLINE - 2.0:
        changed = True
        while changed and (time.time() - start_time) < DEADLINE - 2.0:
            changed = False
            net = list(best_network)
            for idx in range(len(net)):
                if (time.time() - start_time) > DEADLINE - 2.0:
                    break
                trial = net[:idx] + net[idx+1:]
                if _verify(trial, inputs):
                    best_network = trial
                    best_len = len(trial)
                    changed = True
                    break
    
    return best_network

# EVOLVE-BLOCK-END
