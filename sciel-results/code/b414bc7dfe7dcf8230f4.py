"""Combined: bitmask precompute + conflict counting + multi-restart greedy + local-search polish.

Maximizes a 4-uniform sunflower-free family on [1,16].
A sunflower triple: three distinct sets with all three pairwise intersections equal.
"""
import time
import random
from itertools import combinations

_ALL_SETS = None
_MASKS = None
_N_TOTAL = 0
_MASK_TO_IDX = None
_CC = None
_CHOSEN = None
_CHOSEN_SET = None


def _build_tables():
    global _ALL_SETS, _MASKS, _N_TOTAL, _MASK_TO_IDX, _CC, _CHOSEN, _CHOSEN_SET
    if _ALL_SETS is not None:
        return
    _ALL_SETS = list(combinations(range(1, 17), 4))
    _MASKS = []
    for s in _ALL_SETS:
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        _MASKS.append(m)
    _N_TOTAL = len(_MASKS)
    _MASK_TO_IDX = {m: i for i, m in enumerate(_MASKS)}
    _CC = [0] * _N_TOTAL
    _CHOSEN = []
    _CHOSEN_SET = set()


def _forbidden_indices(core_mask, pool_mask):
    """Indices of 4-sets C with C & (core|pool) == core (C contains core, disjoint from pool)."""
    c = bin(core_mask).count("1")
    need = 4 - c
    if need < 0:
        return
    all16 = (1 << 16) - 1
    remaining_mask = all16 & ~core_mask & ~pool_mask
    rem_bits = [i for i in range(16) if (remaining_mask >> i) & 1]
    if need == 0:
        cm = core_mask
        idx = _MASK_TO_IDX.get(cm)
        if idx is not None:
            yield idx
        return
    for T in combinations(rem_bits, need):
        tm = 0
        for b in T:
            tm |= (1 << b)
        cm = core_mask | tm
        idx = _MASK_TO_IDX.get(cm)
        if idx is not None:
            yield idx


def _add_set(idx):
    global _CC
    m = _MASKS[idx]
    _CHOSEN.append(idx)
    _CHOSEN_SET.add(idx)
    for b in _CHOSEN:
        if b == idx:
            continue
        mb = _MASKS[b]
        core = m & mb
        pool = (m | mb) & ~core
        for cidx in _forbidden_indices(core, pool):
            _CC[cidx] += 1


def _remove_set(idx):
    global _CC
    m = _MASKS[idx]
    for b in _CHOSEN:
        if b == idx:
            continue
        mb = _MASKS[b]
        core = m & mb
        pool = (m | mb) & ~core
        for cidx in _forbidden_indices(core, pool):
            _CC[cidx] -= 1
    _CHOSEN.remove(idx)
    _CHOSEN_SET.discard(idx)


def _reset():
    global _CC, _CHOSEN, _CHOSEN_SET
    _CC = [0] * _N_TOTAL
    _CHOSEN = []
    _CHOSEN_SET = set()


def construct():
    _build_tables()
    deadline = time.time() + 80.0
    rng = random.Random(20240517)

    best = 0
    best_chosen = []

    # Phase 1: multi-restart greedy
    for restart in range(8):
        if time.time() > deadline - 25.0:
            break
        _reset()
        order = list(range(_N_TOTAL))
        if restart > 0:
            rng.shuffle(order)
        for idx in order:
            if time.time() > deadline - 25.0:
                break
            if _CC[idx] == 0:
                _add_set(idx)
        cur = len(_CHOSEN)
        if cur > best:
            best = cur
            best_chosen = list(_CHOSEN)

    # Phase 2: local-search polish (grow via remove-then-add + pure additions)
    _reset()
    for idx in best_chosen:
        _add_set(idx)

    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False
        n = len(_CHOSEN)
        # pure additions first
        for j in range(_N_TOTAL):
            if time.time() > deadline - 8.0:
                break
            if j not in _CHOSEN_SET and _CC[j] == 0:
                _add_set(j)
                improved = True
        # swap-based growth: remove one, add several
        for ri in range(len(_CHOSEN)):
            if time.time() > deadline - 8.0:
                break
            removed = _CHOSEN[ri]
            _remove_set(removed)
            added_any = False
            for j in range(_N_TOTAL):
                if time.time() > deadline - 8.0:
                    break
                if j not in _CHOSEN_SET and _CC[j] == 0:
                    _add_set(j)
                    added_any = True
            if not added_any:
                _add_set(removed)  # restore
            else:
                improved = True

    cur = len(_CHOSEN)
    if cur > best:
        best = cur
        best_chosen = list(_CHOSEN)

    return [sorted(_ALL_SETS[i]) for i in best_chosen]
