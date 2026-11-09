"""Size-efficient sorting network on 17 wires."""
import time
import random

N = 17
DEADLINE = 85.0

_INITIAL_STATES = None
_EXPECTED_STATES = None

def _build_initial_states():
    n = N
    total = 1 << n
    wire_states = [0] * n
    for w in range(n):
        val = 0
        for k in range(total):
            if (k >> w) & 1:
                val |= (1 << k)
        wire_states[w] = val
    return wire_states

def _build_expected_states():
    n = N
    total = 1 << n
    expected = [0] * n
    for w in range(n):
        threshold = n - w
        val = 0
        for k in range(total):
            if k.bit_count() >= threshold:
                val |= (1 << k)
        expected[w] = val
    return expected

def _get_initial_states():
    global _INITIAL_STATES
    if _INITIAL_STATES is None:
        _INITIAL_STATES = _build_initial_states()
    return _INITIAL_STATES

def _get_expected_states():
    global _EXPECTED_STATES
    if _EXPECTED_STATES is None:
        _EXPECTED_STATES = _build_expected_states()
    return _EXPECTED_STATES

def _verify_network(comparators):
    n = N
    wire_states = list(_get_initial_states())
    expected = _get_expected_states()
    
    for (i, j) in comparators:
        swap_mask = wire_states[i] & ~wire_states[j]
        if swap_mask:
            wire_states[i] ^= swap_mask
            wire_states[j] ^= swap_mask
    
    for w in range(n):
        if wire_states[w] != expected[w]:
            return False
    return True

def _odd_even_transposition():
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _greedy_remove(comparators, deadline):
    comps = list(comparators)
    start_time = time.time()
    i = 0
    while i < len(comps) and (time.time() - start_time) < deadline:
        trial = comps[:i] + comps[i+1:]
        if _verify_network(trial):
            comps = trial
        else:
            i += 1
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = random.Random(42)
    
    _get_initial_states()
    _get_expected_states()
    
    comps = _odd_even_transposition()
    
    # Phase 1: Greedy removal
    remaining = deadline - time.time() - 15.0
    if remaining > 1:
        comps = _greedy_remove(comps, remaining)
    
    # Phase 2: Multiple random restarts of greedy removal
    best = comps
    best_len = len(comps)
    
    while (deadline - time.time()) > 20:
        rng2 = random.Random(rng.randint(0, 2**31))
        trial = _greedy_remove(list(comps), min(deadline - time.time() - 20, 15.0))
        if len(trial) < best_len:
            best = trial
            best_len = len(trial)
            comps = trial
    
    # Phase 3: Local search
    remaining = deadline - time.time() - 10.0
    if remaining > 1:
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        while (time.time() - start_time) < deadline - 10:
            improved = False
            for i in range(len(comps)):
                if (time.time() - start_time) > deadline - 10:
                    break
                trial = comps[:i] + comps[i+1:]
                if _verify_network(trial):
                    comps = trial
                    improved = True
                    if len(comps) < best_len:
                        best = list(comps)
                        best_len = len(comps)
                    break
            if not improved:
                for _ in range(20):
                    if (time.time() - start_time) > deadline - 10:
                        break
                    if len(comps) < 2:
                        break
                    i, j = rng.sample(range(len(comps)), 2)
                    trial = comps[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    if _verify_network(trial):
                        for k in range(len(trial)):
                            if (time.time() - start_time) > deadline - 10:
                                break
                            trial2 = trial[:k] + trial[k+1:]
                            if _verify_network(trial2):
                                comps = trial2
                                if len(comps) < best_len:
                                    best = list(comps)
                                    best_len = len(comps)
                                improved = True
                                break
                    if improved:
                        break
    
    if not _verify_network(best):
        return _odd_even_transposition()
    
    return best
