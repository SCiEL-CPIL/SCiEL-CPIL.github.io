"""
Size-efficient sorting network search for 17 wires.
Combines big-integer bitmask verification (fast) with greedy removal and local search.
"""
import time
import random
import numpy as np

N = 17
TOTAL_BOOL_INPUTS = 1 << N  # 131072
SEED_COMPARATORS = 136  # odd-even transposition for 17 wires

def _build_seed_network():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _init_wires():
    """Initialize wire bitmasks for all Boolean inputs.
    
    wires[i] is a bitmask of length 2^N where bit k is 1 iff input k has a 1 on wire i.
    """
    total = 1 << N
    wires = []
    for i in range(N):
        val = 0
        block = 1 << i
        period = 2 * block
        for start in range(0, total, period):
            mask = ((1 << block) - 1) << (start + block)
            val |= mask
        wires.append(val)
    return wires

def _compute_expected():
    """Compute expected final wire values after sorting.
    
    After sorting, wire i (0-indexed, 0 is smallest) should contain 1s for all inputs
    where the (N-i)-th smallest value is 1.
    Since we're dealing with boolean inputs, the sorted output on wire i is 1 iff
    the input has at least (N-i) ones.
    """
    expected = []
    total = 1 << N
    for i in range(N):
        val = 0
        threshold = N - i
        for k in range(total):
            if k.bit_count() >= threshold:
                val |= (1 << k)
        expected.append(val)
    return expected

# Pre-compute wires_init and expected for speed
_wires_init = _init_wires()
_expected = _compute_expected()

def _verify_network_bitmask(comparators):
    """Verify network sorts all Boolean inputs using bitmask simulation.
    
    This is the fastest verification method using Python big integers.
    Each wire is represented as a single big integer where bit k indicates
    the value on that wire for input k.
    """
    wires = _wires_init[:]
    for (a, b) in comparators:
        ma = wires[a] & wires[b]
        mb = wires[a] | wires[b]
        wires[a] = ma
        wires[b] = mb
    
    for i in range(N):
        if wires[i] != _expected[i]:
            return False
    return True

def _greedy_remove(comps, deadline):
    """Repeatedly try to remove each comparator until a full pass removes none.
    Returns the reduced list."""
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify_network_bitmask(new_comps):
                comps = new_comps
                improved = True
                break  # restart scan from the end
    return comps

def _local_search(comps, deadline, max_iter=50):
    """
    Local search: try swapping adjacent comparators, or moving a comparator
    to a different position, if it maintains validity.
    """
    current = list(comps)
    best = list(current)
    best_len = len(current)
    rng = random.Random(42)
    
    for _ in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        improved = False
        n = len(current)
        
        # Try random swaps (reorder)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            if n < 2:
                break
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            cand = current[:]
            cand[i], cand[j] = cand[j], cand[i]
            if _verify_network_bitmask(cand):
                reduced = _greedy_remove(cand, deadline - 10)
                if len(reduced) < len(current):
                    current = reduced
                    improved = True
                    break
                elif len(reduced) == len(current) and rng.random() < 0.3:
                    # accept a same-size reorder to change the landscape
                    current = reduced
                    improved = True
                    break
        
        # Try inserting a comparator then re-removing (escape move)
        if not improved and time.time() < deadline - 10:
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                a = rng.randint(0, N - 1)
                b = rng.randint(a + 1, N)
                pos = rng.randint(0, len(current) + 1)
                cand = current[:pos] + [(a, b)] + current[pos:]
                if _verify_network_bitmask(cand):
                    reduced = _greedy_remove(cand, deadline - 10)
                    if len(reduced) < len(current):
                        current = reduced
                        improved = True
                        break
                    elif len(reduced) == len(current) and rng.random() < 0.2:
                        current = reduced
                        improved = True
                        break
        
        if len(current) < best_len:
            best = current[:]
            best_len = len(current)
        
        if not improved:
            # No improvement in this iteration, try a few more random perturbations
            perturbations = 0
            while not improved and perturbations < 5 and time.time() < deadline - 10:
                perturbations += 1
                n = len(current)
                if n < 2:
                    break
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                cand = current[:]
                cand[i], cand[j] = cand[j], cand[i]
                if _verify_network_bitmask(cand):
                    reduced = _greedy_remove(cand, deadline - 10)
                    if len(reduced) < len(current):
                        current = reduced
                        improved = True
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Start with seed network
    comps = _build_seed_network()
    best = list(comps)
    best_len = len(comps)
    
    # Phase 1: Greedy removal of redundant comparators
    if time.time() < deadline - 30:
        removed = _greedy_remove(comps, deadline - 30)
        if len(removed) < best_len:
            best = removed
            best_len = len(removed)
        comps = removed
    
    # Phase 2: Local search for further improvements
    if time.time() < deadline - 15:
        searched = _local_search(comps, deadline - 15, max_iter=50)
        if len(searched) < best_len:
            best = searched
            best_len = len(searched)
    
    return best
