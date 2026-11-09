"""Size-efficient sorting network on 17 wires: fast verify + aggressive prune."""
import time
import random
import numpy as np

N = 17
TOTAL = 1 << N  # 131072
SORTED = np.arange(TOTAL, dtype=np.int32)


def _build_seed():
    """Odd-even transposition network (136 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify_fast(comp):
    """Zero-one principle over all 2^N Boolean inputs. Returns True iff sorted."""
    if not comp:
        return True
    state = SORTED.copy()
    for i, j in comp:
        vi = (state >> i) & 1
        vj = (state >> j) & 1
        # swap where vi=1 and vj=0
        swap_mask = vi & (1 - vj)
        if np.any(swap_mask):
            state ^= (swap_mask << i) | (swap_mask << j)
    return bool(np.array_equal(state, SORTED))


def _prune_aggressive(comp, deadline):
    """Aggressively prune redundant comparators."""
    comp = list(comp)
    best_len = len(comp)
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for idx in range(len(comp) - 1, -1, -1):
            if time.time() > deadline:
                break
            cand = comp[:idx] + comp[idx + 1:]
            if _verify_fast(cand):
                comp = cand
                best_len = len(comp)
                progress = True
    return comp


def _local_search(comp, deadline):
    """Local search: try removals, swaps, and insertions."""
    comp = list(comp)
    rnd = random.Random(42)
    all_pairs = [(i, j) for j in range(N) for i in range(j)]
    
    while time.time() < deadline:
        improved = False
        # Try removing each comparator
        for idx in range(len(comp) - 1, -1, -1):
            if time.time() > deadline:
                return comp
            cand = comp[:idx] + comp[idx + 1:]
            if _verify_fast(cand):
                comp = cand
                improved = True
                break
        if improved:
            continue
        
        # Try swapping adjacent comparators (if they don't share a wire)
        swapped = False
        for idx in range(len(comp) - 1):
            if time.time() > deadline:
                return comp
            (i1, j1), (i2, j2) = comp[idx], comp[idx + 1]
            if len({i1, j1, i2, j2}) == 4:  # independent, can swap
                cand = comp[:idx] + [comp[idx + 1], comp[idx]] + comp[idx + 2:]
                if _verify_fast(cand):
                    comp = cand
                    swapped = True
                    break
        if swapped:
            continue
        
        # Try inserting a random comparator at a random position
        if time.time() > deadline - 2.0:
            return comp
        c = rnd.choice(all_pairs)
        pos = rnd.randint(0, len(comp))
        cand = comp[:pos] + [c] + comp[pos:]
        if _verify_fast(cand):
            comp = _prune_aggressive(cand, deadline)
    
    return comp


def construct():
    t_start = time.time()
    deadline = t_start + 82.0  # safety margin before 90s kill
    comp = _build_seed()
    
    # Verify seed
    if not _verify_fast(comp):
        return comp
    
    # Phase 1: Aggressive pruning from seed
    comp = _prune_aggressive(comp, deadline)
    best = comp
    best_len = len(comp)
    
    # Phase 2: Multiple local search restarts
    if time.time() < deadline - 8.0:
        seeds = [1, 7, 13, 23, 42, 99, 123, 456, 789, 1000]
        for seed_val in seeds:
            if time.time() > deadline - 8.0:
                break
            trial = list(best)
            rnd = random.Random(seed_val)
            rnd.shuffle(trial)
            # Also try random removals before local search
            for _ in range(min(5, len(trial) // 4)):
                if time.time() > deadline - 8.0:
                    break
                idx = rnd.randint(0, len(trial) - 1)
                trial = trial[:idx] + trial[idx + 1:]
                if _verify_fast(trial):
                    continue
                else:
                    trial = list(best)
                    rnd.shuffle(trial)
                    break
            trial = _local_search(trial, deadline - 5.0)
            if len(trial) < best_len and _verify_fast(trial):
                best = trial
                best_len = len(trial)
    
    # Phase 3: Final pruning pass
    if time.time() < deadline:
        best = _prune_aggressive(best, deadline)
    
    return best
