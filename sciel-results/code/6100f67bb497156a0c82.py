"""Size-efficient sorting networks on 17 wires.

Uses a structured construction + local improvement approach:
1. Build a correct sorting network via odd-even transposition or a layered
   construction that is known to be correct.
2. Aggressively prune unnecessary comparators using the 0-1 principle.
3. Attempt local improvements by swapping/replacing comparators.
"""
N = 17
import time
import random
import numpy as np

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        # Create all 2^17 boolean inputs as a (131072, 17) int8 array
        idx = np.arange(1 << N, dtype=np.int32)
        bits = np.zeros((1 << N, N), dtype=np.int8)
        for j in range(N):
            bits[:, j] = (idx >> j) & 1
        _ALL_INPUTS = bits
    return _ALL_INPUTS

def _simulate(network, inputs=None):
    """Simulate the network on all boolean inputs. Returns final state array."""
    if inputs is None:
        inputs = _get_all_inputs().copy()
    for i, j in network:
        a = inputs[:, i]
        b = inputs[:, j]
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    return inputs

def _is_valid(network):
    """Check if network sorts all boolean inputs."""
    final = _simulate(network)
    for col in range(N - 1):
        if np.any(final[:, col] > final[:, col + 1]):
            return False
    return True

def _count_unsorted(network):
    """Count total inversions across all boolean inputs after simulation."""
    final = _simulate(network)
    total = 0
    for col in range(N - 1):
        total += int(np.sum(final[:, col] > final[:, col + 1]))
    return total

def _depth_of_network(network):
    """Compute parallel depth."""
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0

def _seed_network():
    """Odd-even transposition sorting network on N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _layered_network():
    """
    Construct a sorting network using a layered approach:
    - Sort adjacent pairs in multiple passes (like bitonic or Batcher).
    This is a heuristic that often produces smaller networks than odd-even.
    """
    comps = []
    # Phase 1: sort pairs (0,1), (2,3), ...
    for i in range(0, N - 1, 2):
        comps.append((i, i + 1))
    # Phase 2: sort (1,2), (3,4), ...
    for i in range(1, N - 1, 2):
        comps.append((i, i + 1))
    # Phase 3: sort (0,2), (1,3), ...
    for i in range(0, N - 2, 2):
        comps.append((i, i + 2))
    # Phase 4: sort (2,4), (3,5), ...
    for i in range(2, N - 2, 2):
        comps.append((i, i + 2))
    # Continue with odd-even transposition to ensure correctness
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _prune(network):
    """Remove unnecessary comparators from a valid network."""
    network = list(network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(network)):
            trial = network[:idx] + network[idx+1:]
            if _is_valid(trial):
                network = trial
                changed = True
                break
    return network

def _try_remove_random(network, n_attempts=10):
    """Try to remove random comparators."""
    network = list(network)
    for _ in range(n_attempts):
        if len(network) <= 1:
            break
        idx = random.randint(0, len(network) - 1)
        trial = network[:idx] + network[idx+1:]
        if _is_valid(trial):
            network = trial
    return network

def _try_swap(network):
    """Try swapping two comparators."""
    if len(network) < 2:
        return None
    i, j = random.sample(range(len(network)), 2)
    trial = list(network)
    trial[i], trial[j] = trial[j], trial[i]
    if _is_valid(trial):
        return trial
    return None

def _try_replace(network):
    """Try replacing one comparator with a random one."""
    if len(network) < 1:
        return None
    idx = random.randint(0, len(network) - 1)
    new_i = random.randint(0, N - 2)
    new_j = random.randint(new_i + 1, N - 1)
    trial = list(network)
    trial[idx] = (new_i, new_j)
    if _is_valid(trial):
        return trial
    return None

def _insert_and_prune(network, n_inserts=1, n_prunes=10):
    """Insert random comparators then prune to try to find a smaller valid network."""
    network = list(network)
    for _ in range(n_inserts):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        network.append((i, j))
    # Prune
    changed = True
    while changed and n_prunes > 0:
        changed = False
        for idx in range(len(network)):
            trial = network[:idx] + network[idx+1:]
            if _is_valid(trial):
                network = trial
                changed = True
                n_prunes -= 1
                if n_prunes <= 0:
                    break
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with seed network
    best_network = _seed_network()
    
    if not _is_valid(best_network):
        # Should not happen, but try layered
        best_network = _layered_network()
        if not _is_valid(best_network):
            # Fallback: brute force odd-even
            best_network = _seed_network()
    
    best_size = len(best_network)
    
    # Aggressive initial pruning
    best_network = _prune(best_network)
    best_size = len(best_network)
    
    # Local search loop
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        current = list(best_network)
        
        # Try different mutation strategies
        strategy = random.choice(['remove', 'swap', 'replace', 'insert_prune', 'prune'])
        
        if strategy == 'prune':
            trial = _prune(current)
            if len(trial) < best_size:
                best_network = trial
                best_size = len(trial)
                improved = True
                no_improve = 0
                continue
        
        elif strategy == 'remove':
            trial = _try_remove_random(current, n_attempts=5)
            if len(trial) < best_size and _is_valid(trial):
                best_network = trial
                best_size = len(trial)
                improved = True
                no_improve = 0
                continue
        
        elif strategy == 'swap':
            trial = _try_swap(current)
            if trial is not None:
                # After swap, try to prune
                pruned = _prune(trial)
                if len(pruned) < best_size:
                    best_network = pruned
                    best_size = len(pruned)
                    improved = True
                    no_improve = 0
                    continue
                elif len(trial) < best_size:
                    best_network = trial
                    best_size = len(trial)
                    improved = True
                    no_improve = 0
                    continue
        
        elif strategy == 'replace':
            trial = _try_replace(current)
            if trial is not None:
                pruned = _prune(trial)
                if len(pruned) < best_size:
                    best_network = pruned
                    best_size = len(pruned)
                    improved = True
                    no_improve = 0
                    continue
                elif len(trial) < best_size:
                    best_network = trial
                    best_size = len(trial)
                    improved = True
                    no_improve = 0
                    continue
        
        elif strategy == 'insert_prune':
            trial = _insert_and_prune(current, n_inserts=1, n_prunes=20)
            if len(trial) < best_size and _is_valid(trial):
                best_network = trial
                best_size = len(trial)
                improved = True
                no_improve = 0
                continue
        
        if not improved:
            no_improve += 1
        
        # Periodically do a full prune on best
        if no_improve % 50 == 0:
            pruned = _prune(best_network)
            if len(pruned) < best_size:
                best_network = pruned
                best_size = len(pried) if False else len(pruned)
                no_improve = 0
    
    # Final prune
    final = _prune(best_network)
    if len(final) < best_size:
        best_network = final
    
    return best_network
