"""Maximize triples on 16 points avoiding K(2,2,2) transversal completion."""
import time
import random
import itertools
import numpy as np

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
TRIPLE_TO_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}
NT = len(ALL_TRIPLES)  # 560

def _build_constraint_lists():
    lists = [[] for _ in range(NT)]
    for i, (a, b, c) in enumerate(ALL_TRIPLES):
        others = [x for x in range(N) if x not in (a, b, c)]
        seen = set()
        for a_p in others:
            for b_p in others:
                if b_p == a_p:
                    continue
                for c_p in others:
                    if c_p == a_p or c_p == b_p:
                        continue
                    for t in [(a, b, c_p), (a, b_p, c), (a, b_p, c_p),
                              (a_p, b, c), (a_p, b, c_p), (a_p, b_p, c), (a_p, b_p, c_p)]:
                        j = TRIPLE_TO_IDX[tuple(sorted(t))]
                        if j not in seen:
                            seen.add(j)
                            lists[i].append(j)
        lists[i] = sorted(seen)
    return lists

CONSTRAINTS = _build_constraint_lists()

def _greedy_construct(rng):
    present = np.zeros(NT, dtype=bool)
    block_count = np.zeros(NT, dtype=np.int32)
    order = list(range(NT))
    rng.shuffle(order)
    count = 0
    for i in order:
        if block_count[i] < 7:
            present[i] = True
            count += 1
            for j in CONSTRAINTS[i]:
                block_count[j] += 1
    return present, block_count, count

def _try_add(present, block_count, count, i, rng):
    """Try to add triple i, possibly via a swap. Returns (present, block_count, count, improved)."""
    if block_count[i] < 7:
        present[i] = True
        for j in CONSTRAINTS[i]:
            block_count[j] += 1
        return present, block_count, count + 1, True
    else:
        # Need to remove a partner
        partners_present = [j for j in CONSTRAINTS[i] if present[j]]
        if len(partners_present) < 7:
            return present, block_count, count, False
        # Try removing one partner
        rng.shuffle(partners_present)
        for r in partners_present:
            # After removing r, block_count[i] decreases by 1
            if block_count[i] - 1 < 7:
                present[r] = False
                for j in CONSTRAINTS[r]:
                    block_count[j] -= 1
                present[i] = True
                for j in CONSTRAINTS[i]:
                    block_count[j] += 1
                return present, block_count, count, True
        return present, block_count, count, False

def _local_search(present, block_count, count, deadline, rng):
    best_present = present.copy()
    best_block_count = block_count.copy()
    best_count = count
    stall = 0
    while time.time() < deadline:
        improved = False
        missing = np.where(~present)[0]
        if len(missing) == 0:
            break
        # Try random missing triples
        n_tries = min(200, len(missing))
        for _ in range(n_tries):
            i = missing[rng.randrange(len(missing))]
            present, block_count, count, ok = _try_add(present, block_count, count, i, rng)
            if ok:
                improved = True
                if count > best_count:
                    best_present = present.copy()
                    best_block_count = block_count.copy()
                    best_count = count
                    stall = 0
                else:
                    stall += 1
        if not improved:
            stall += 10
        if stall > 200:
            break
    return best_present, best_block_count, best_count

def construct():
    t0 = time.time()
    deadline = t0 + 140
    rng = random.Random(12345)
    
    best_present = None
    best_count = 0
    best_block_count = None
    
    # Phase 1: multiple greedy restarts
    while time.time() < deadline - 40:
        present, block_count, count = _greedy_construct(rng)
        if count > best_count:
            best_count = count
            best_present = present.copy()
            best_block_count = block_count.copy()
    
    # Phase 2: local search
    if best_present is not None and time.time() < deadline - 5:
        best_present, best_block_count, best_count = _local_search(
            best_present, best_block_count, best_count, deadline - 2, rng)
    
    result = [list(ALL_TRIPLES[i]) for i in range(NT) if best_present[i]]
    return result
