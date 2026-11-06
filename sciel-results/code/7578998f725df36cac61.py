"""Size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 78.0  # safety margin before 90s kill

def _apply_network_np(net, inputs):
    """Apply network to array of inputs (shape: (num_inputs, N))."""
    v = inputs.copy()
    for i, j in net:
        swap = v[:, i] > v[:, j]
        v[:, i] = np.where(swap, v[:, j], v[:, i])
        v[:, j] = np.where(swap, v[:, i], v[:, j])
    return v

def _check_sorted_np(v):
    """Check if all rows are sorted. Returns True if all sorted."""
    return np.all(v[:, :-1] <= v[:, 1:])

def _fast_check(net, inputs):
    """Check network on given inputs (numpy array)."""
    v = _apply_network_np(net, inputs)
    return _check_sorted_np(v)

def _full_check(net):
    """Full verification over all 2^18 boolean inputs using numpy."""
    # Generate all 2^18 inputs
    all_inputs = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
    v = _apply_network_np(net, all_inputs)
    return _check_sorted_np(v)

def _generate_initial_network():
    """Generate a reasonable starting network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _random_bool_input():
    return [random.randint(0, 1) for _ in range(N)]

def construct():
    start_time = time.time()
    random.seed(123)
    
    # Start with odd-even transposition (153 comparators)
    best_net = _generate_initial_network()
    best_len = len(best_net)
    
    # Verify initial
    assert _full_check(best_net), "Initial network should be correct"
    
    # Generate sample sets for fast checking
    # Hard samples: single bits, all-zero, all-one
    hard_samples_list = []
    hard_samples_list.append([0]*N)
    hard_samples_list.append([1]*N)
    for i in range(N):
        s = [0]*N
        s[i] = 1
        hard_samples_list.append(s)
    for i in range(N):
        s = [1]*N
        s[i] = 0
        hard_samples_list.append(s)
    for _ in range(200):
        hard_samples_list.append(_random_bool_input())
    hard_samples = np.array(hard_samples_list, dtype=np.int8)
    
    # Medium samples for intermediate verification
    med_samples_list = [_random_bool_input() for _ in range(500)]
    med_samples = np.array(med_samples_list, dtype=np.int8)
    
    current_net = list(best_net)
    current_len = best_len
    
    # Local search
    while time.time() - start_time < DEADLINE:
        op = random.random()
        trial_net = list(current_net)
        
        if op < 0.45 and len(trial_net) > 1:
            # Remove a random comparator
            idx = random.randint(0, len(trial_net) - 1)
            trial_net.pop(idx)
            is_removal = True
        elif op < 0.75 and len(trial_net) >= 2:
            # Swap two comparators
            i, j = random.sample(range(len(trial_net)), 2)
            trial_net[i], trial_net[j] = trial_net[j], trial_net[i]
            is_removal = False
        else:
            # Move a comparator
            if len(trial_net) < 2:
                continue
            src = random.randint(0, len(trial_net) - 1)
            dst = random.randint(0, len(trial_net) - 1)
            comp = trial_net.pop(src)
            if src < dst:
                dst -= 1
            trial_net.insert(dst, comp)
            is_removal = False
        
        # Fast check
        if not _fast_check(trial_net, hard_samples):
            continue
        
        if is_removal:
            # Medium check
            if not _fast_check(trial_net, med_samples):
                continue
            # Full check
            if _full_check(trial_net):
                current_net = trial_net
                current_len = len(current_net)
                if current_len < best_len:
                    best_net = list(current_net)
                    best_len = current_len
        else:
            # For swaps/moves, accept if fast check passes
            current_net = trial_net
    
    # Greedy removal pass on best_net
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for i in range(len(best_net)):
            if time.time() - start_time > DEADLINE:
                break
            candidate = list(best_net)
            candidate.pop(i)
            # Fast check first
            if _fast_check(candidate, hard_samples) and _fast_check(candidate, med_samples):
                if _full_check(candidate):
                    best_net = candidate
                    best_len = len(best_net)
                    improved = True
                    break
    
    return best_net
