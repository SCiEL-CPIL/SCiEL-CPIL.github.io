"""Simulated annealing with bitset-accelerated energy for 3-AP maximization without 4-APs."""
import time
import random
import numpy as np

N = 256

# Precompute all 3-APs (a, b, c) with a < b < c, b-a = c-b
_3APS = []
_3AP_INDEX = {}  # (a, c) -> list of b values (for fast counting)
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        c = a + 2 * d
        if c <= N:
            _3APS.append((a - 1, b - 1, c - 1))
            _3AP_INDEX.setdefault((a - 1, c - 1), []).append(b - 1)

# Precompute all 4-APs (a, b, c, d) with a < b < c < d, b-a = c-b = d-c
_4APS = []
for a in range(1, N + 1):
    for d in range(1, N):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e <= N:
            _4APS.append((a - 1, b - 1, c - 1, e - 1))

# For each element i, precompute:
# - pairs3[i]: list of (j, k) such that {i, j, k} forms a 3-AP
# - quads4[i]: list of 4-AP tuples containing i
_pairs3 = [[] for _ in range(N)]
for (a, b, c) in _3APS:
    _pairs3[a].append((b, c))
    _pairs3[b].append((a, c))
    _pairs3[c].append((a, b))

_quads4 = [[] for _ in range(N)]
for (a, b, c, d) in _4APS:
    _quads4[a].append((b, c, d))
    _quads4[b].append((a, c, d))
    _quads4[c].append((a, b, d))
    _quads4[d].append((a, b, c))

PENALTY = 10000  # Large penalty for each 4-AP


def _count_3aps(state):
    """Count 3-APs in the given state (numpy bool array)."""
    count = 0
    for a in range(N):
        if state[a]:
            for (b, c) in _pairs3[a]:
                if state[b] and state[c]:
                    count += 1
    return count


def _count_4aps(state):
    """Count 4-APs in the given state."""
    count = 0
    for a in range(N):
        if state[a]:
            for (b, c, d) in _quads4[a]:
                if state[b] and state[c] and state[d]:
                    count += 1
    return count


def _energy(state):
    """Energy = -3AP_count + PENALTY * 4AP_count. Lower is better."""
    return -_count_3aps(state) + PENALTY * _count_4aps(state)


def _greedy_initial():
    """Greedy construction: add elements that maximize 3-AP gain without creating 4-APs."""
    state = np.zeros(N, dtype=bool)
    order = list(range(N))
    random.shuffle(order)
    for i in order:
        # Try adding i
        state[i] = True
        # Check if any 4-AP is created
        ok = True
        for (b, c, d) in _quads4[i]:
            if state[b] and state[c] and state[d]:
                ok = False
                break
        if not ok:
            state[i] = False
        else:
            # Check if this creates too many 4-APs (it shouldn't if ok)
            pass
    return state


def _local_search(state, time_budget, deadline):
    """Iterated local search with simulated annealing."""
    best_state = state.copy()
    best_energy = _energy(state)
    cur_energy = best_energy
    cur_state = state.copy()
    
    # SA parameters
    T_start = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    t = T_start
    step_count = 0
    
    while time.time() < deadline and t > T_min:
        # Choose a random move: flip one element
        i = random.randint(0, N - 1)
        old_val = cur_state[i]
        cur_state[i] = not old_val
        
        # Compute energy delta
        # Delta in 3-AP count:
        d3 = 0
        for (b, c) in _pairs3[i]:
            if cur_state[b] and cur_state[c]:
                if old_val:
                    d3 -= 1  # removed a 3-AP
                else:
                    d3 += 1  # added a 3-AP
        
        # Delta in 4-AP count:
        d4 = 0
        for (b, c, d) in _quads4[i]:
            if cur_state[b] and cur_state[c] and cur_state[d]:
                if old_val:
                    d4 -= 1
                else:
                    d4 += 1
        
        delta_e = -d3 + PENALTY * d4
        new_energy = cur_energy + delta_e
        
        # Accept?
        if new_energy < cur_energy or (t > T_min and random.random() < np.exp(-delta_e / t)):
            cur_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = cur_state.copy()
        else:
            cur_state[i] = old_val
        
        step_count += 1
        t *= cooling_rate
        
        # Occasionally do a larger perturbation (restart-like)
        if step_count % 500 == 0 and time.time() < deadline - 5:
            # Perturb: flip 3-5 random elements
            for _ in range(random.randint(3, 5)):
                j = random.randint(0, N - 1)
                cur_state[j] = not cur_state[j]
            cur_energy = _energy(cur_state)
            # Reset temperature slightly
            t = max(t, 10.0)
    
    return best_state, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    random.seed(42)
    np.random.seed(42)
    
    # Multiple restarts with different initializations
    best_state = None
    best_energy = float('inf')
    
    # Strategy 1: Greedy initial + SA
    if time.time() < deadline - 10:
        state = _greedy_initial()
        s, e = _local_search(state, 10, deadline - 10)
        if e < best_energy:
            best_energy = e
            best_state = s
    
    # Strategy 2: Random initial + SA
    if time.time() < deadline - 10:
        state = np.zeros(N, dtype=bool)
        # Random subset of size ~100
        indices = random.sample(range(N), 100)
        for i in indices:
            state[i] = True
        # Remove 4-APs greedily
        for (a, b, c, d) in _4APS:
            if state[a] and state[b] and state[c] and state[d]:
                state[random.choice([a, b, c, d])] = False
        s, e = _local_search(state, 10, deadline - 10)
        if e < best_energy:
            best_energy = e
            best_state = s
    
    # Strategy 3: Another greedy variant with different order
    if time.time() < deadline - 10:
        state = np.zeros(N, dtype=bool)
        # Add elements in order of "density" - prefer middle elements
        order = sorted(range(N), key=lambda x: abs(x - N // 2))
        random.shuffle(order)
        for i in order:
            state[i] = True
            ok = True
            for (b, c, d) in _quads4[i]:
                if state[b] and state[c] and state[d]:
                    ok = False
                    break
            if not ok:
                state[i] = False
        s, e = _local_search(state, 10, deadline - 10)
        if e < best_energy:
            best_energy = e
            best_state = s
    
    # Strategy 4: More SA iterations on the best found
    if time.time() < deadline - 5 and best_state is not None:
        s, e = _local_search(best_state, 10, deadline - 5)
        if e < best_energy:
            best_energy = e
            best_state = s
    
    # Final verification and cleanup: ensure no 4-APs
    if best_state is None:
        best_state = np.zeros(N, dtype=bool)
        best_state[0] = True
    
    # Remove any remaining 4-APs (safety)
    for (a, b, c, d) in _4APS:
        if best_state[a] and best_state[b] and best_state[c] and best_state[d]:
            best_state[d] = False
    
    # Convert to sorted list
    result = [i + 1 for i in range(N) if best_state[i]]
    
    if not result:
        result = [1]
    
    return result
