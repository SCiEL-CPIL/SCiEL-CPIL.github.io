"""Combined: fast bitmask verify + iterative local search with restarts."""
import numpy as np
import time
import random
from math import comb

N = 18
NUM_BOOLS = 1 << N
NUM_WORDS = NUM_BOOLS // 64

_INITIAL_MASKS = None
_EXPECTED_MASKS = None

def _precompute_masks(n=N):
    global _INITIAL_MASKS, _EXPECTED_MASKS
    if _INITIAL_MASKS is not None:
        return
    num_words = (1 << n) // 64
    _INITIAL_MASKS = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        for w in range(num_words):
            base = 64 * w
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            _INITIAL_MASKS[i, w] = word
    _EXPECTED_MASKS = np.zeros(n, dtype=np.int64)
    for i in range(n):
        count = 0
        for j in range(n - i, n + 1):
            count += comb(n, j)
        _EXPECTED_MASKS[i] = count

def _verify_fast(comps, n=N):
    _precompute_masks(n)
    num_words = (1 << n) // 64
    masks = _INITIAL_MASKS.copy()
    for (a, b) in comps:
        masks[a] &= masks[b]
        masks[b] = masks[a] | masks[b]
    for i in range(n):
        bit_count = 0
        for w in masks[i]:
            bit_count += bin(w).count('1')
        if bit_count != _EXPECTED_MASKS[i]:
            return False
    for i in range(n - 1):
        if np.any(masks[i] & ~masks[i + 1]):
            return False
    return True

def _build_ooe_network(n):
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _build_batcher_n18():
    """A known 77-comparator Batcher-style network for 18 wires (approximate)."""
    # Use odd-even as fallback base; we'll optimize from there.
    return _build_ooe_network(n=18)

def _local_search_remove(comps, deadline, rng):
    """Greedy removal with random ordering. Returns (comps, improved_count)."""
    improved = 0
    while time.time() < deadline:
        n = len(comps)
        if n == 0:
            break
        order = list(range(n))
        rng.shuffle(order)
        found = False
        for idx in order:
            if time.time() > deadline:
                return comps, improved
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, N):
                comps = trial
                improved += 1
                found = True
                break
        if not found:
            break
    return comps, improved

def _local_search_swap_remove(comps, deadline, rng):
    """Try swapping adjacent comparators then removing. Returns (comps, improved)."""
    improved = 0
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = rng.randint(0, max(0, len(comps) - 2))
            trial = comps[:]
            trial[idx], trial[idx + 1] = trial[idx + 1], trial[idx]
            if _verify_fast(trial, N):
                # Try removing from the swapped network
                for j in range(len(trial)):
                    if time.time() > deadline:
                        break
                    trial2 = trial[:j] + trial[j+1:]
                    if _verify_fast(trial2, N):
                        comps = trial2
                        improved += 1
                        progress = True
                        break
                if progress:
                    break
    return comps, improved

def _local_search_move(comps, deadline, rng):
    """Move a comparator to a different position (if it doesn't violate wire dependencies)."""
    improved = 0
    while time.time() < deadline:
        n = len(comps)
        if n < 2:
            break
        src = rng.randint(0, n - 1)
        dst = rng.randint(0, n - 1)
        if src == dst:
            continue
        trial = comps[:]
        elem = trial.pop(src)
        trial.insert(dst, elem)
        if _verify_fast(trial, N):
            # Try to remove something
            for j in range(len(trial)):
                if time.time() > deadline:
                    break
                trial2 = trial[:j] + trial[j+1:]
                if _verify_fast(trial2, N):
                    comps = trial2
                    improved += 1
                    break
            if improved > 0:
                continue
            # Accept the move even without immediate removal (explore)
            comps = trial
            improved += 0  # track as neutral move
        if improved >= 50:
            break
    return comps, improved

def _perturb(comps, rng, num_ops=3):
    """Apply random perturbations: move, add, remove."""
    trial = list(comps)
    for _ in range(num_ops):
        op = rng.randint(0, 2)
        if op == 0 and len(trial) > 0:
            # Remove a random comparator
            idx = rng.randint(0, len(trial) - 1)
            trial.pop(idx)
        elif op == 1 and len(trial) > 1:
            # Move a comparator
            src = rng.randint(0, len(trial) - 1)
            dst = rng.randint(0, len(trial) - 1)
            elem = trial.pop(src)
            trial.insert(dst, elem)
        elif op == 2:
            # Add a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(trial))
            trial.insert(pos, (i, j))
    return trial

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = N
    _precompute_masks(n)
    
    # Start with odd-even transposition
    comps = _build_ooe_network(n)
    
    if not _verify_fast(comps, n):
        return comps
    
    best = list(comps)
    best_len = len(best)
    
    rng = random.Random(12345)
    
    # Phase 1: Initial aggressive greedy removal
    comps, _ = _local_search_remove(comps, min(deadline, start_time + 30.0), rng)
    if len(comps) < best_len:
        best = list(comps)
        best_len = len(best)
    
    # Phase 2: Iterative local search with restarts
    # Alternate between different local search strategies and occasional perturbations
    while time.time() < deadline:
        # Determine which local search to use
        strategy = rng.randint(0, 3)
        
        if strategy == 0:
            # Pure removal
            new_comps, imp = _local_search_remove(comps, min(deadline, time.time() + 5.0), rng)
        elif strategy == 1:
            # Swap + remove
            new_comps, imp = _local_search_swap_remove(comps, min(deadline, time.time() + 5.0), rng)
        elif strategy == 2:
            # Move + remove
            new_comps, imp = _local_search_move(comps, min(deadline, time.time() + 5.0), rng)
        else:
            # Perturb then remove
            new_comps = _perturb(comps, rng, num_ops=rng.randint(1, 4))
            new_comps, imp = _local_search_remove(new_comps, min(deadline, time.time() + 5.0), rng)
        
        if len(new_comps) < best_len:
            best = list(new_comps)
            best_len = len(best)
            comps = list(new_comps)
        elif len(new_comps) == len(comps) and imp > 0:
            # No improvement but some moves were accepted; keep exploring
            comps = list(new_comps)
        else:
            # Stuck; try a stronger perturbation
            if rng.random() < 0.3:
                comps = _perturb(comps, rng, num_ops=rng.randint(2, 6))
                # Re-verify; if invalid, revert to best
                if not _verify_fast(comps, N):
                    comps = list(best)
    
    return best
