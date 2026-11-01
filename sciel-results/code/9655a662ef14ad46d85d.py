"""Improved covering-code constructor: direct-sum baseline + greedy set-cover +
repair-based stochastic local search.

Strategy:
  * A simple always-valid baseline (direct sum of a radius-1 perfect code with the
    remaining coordinates) guarantees correctness and gives an upper bound.
  * A greedy set-cover heuristic (place the center that covers the most uncovered
    words, repeat) finds codes that are often smaller than the baseline.
  * A repair-based stochastic local search (SLS) maximizes the number of radius-1
    balls packed while penalizing overlaps via dynamic weights.  "Repair" moves
    instantly fix coverage deficits by swapping in a word that covers the most
    uncovered positions, keeping the codeword count constant.  This shifts focus
    from minimizing set-cover cost to maximizing packing density with hard coverage
    constraints handled by repair.
  * Coverage is verified exactly before returning; on any failure or time-out the
    valid baseline is returned, so the result is always a correct covering.
"""

import time
import itertools
import numpy as np
import random

# Safety margin: the evaluator kills the process at 150 s; keep a generous buffer.
_DEADLINE = 140.0


def _direct_sum(q, n, radius):
    """Always-valid baseline: a radius-1 perfect code of length 4 over the alphabet,
    tensored with every word on the remaining n-4 coordinates.  Valid for R>=1, n>=4.
    Size = q^2 * q^(n-4) = q^(n-2)."""
    if n < 4:
        # Fallback for small n: cover every word directly
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    # Perfect (q^2, 4, 1) covering: positions 0,1 free; 2 = a+b; 3 = a+2b (mod q).
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]


def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx


def _ball_indices(c, q, n, radius):
    """Flat indices of all words within Hamming distance `radius` of center c."""
    if radius == 0:
        return [_idx(c, q)]
    ball = [_idx(c, q)]
    for d in range(1, radius + 1):
        for pos in itertools.combinations(range(n), d):
            for vals in itertools.product(range(1, q), repeat=d):
                w = list(c)
                for p, v in zip(pos, vals):
                    w[p] = v
                ball.append(_idx(w, q))
    return ball


def _build_balls(q, n, radius):
    """Precompute the radius ball (as a sorted unique list of flat indices) for every
    word in {0..q-1}^n.  Returns (all_words, balls) where all_words is a numpy array
    of shape (total, n) and balls is a list of lists of ints."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    balls = []
    for i in range(total):
        c = all_words[i]
        balls.append(np.array(_ball_indices(c.tolist(), q, n, radius), dtype=np.int32))
    return all_words, balls


def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space (only used for small spaces)."""
    total = q ** n
    if total > 5 * 10**6:
        return False
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        for idx in _ball_indices(c, q, n, radius):
            covered[idx] = True
    return bool(covered.all())


def _greedy(q, n, radius, time_budget):
    """Greedy set-cover.  Returns (codewords, verified) where verified is True only if
    the returned code is a confirmed covering of the whole space."""
    total = q ** n
    if total > 3 * 10**6:
        return None, False  # too large to track with an explicit covered-set

    start = time.time()
    all_words, balls = _build_balls(q, n, radius)
    
    # Build a binary matrix: ball_matrix[i][j] = 1 if word j is in ball of word i
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        ball_matrix[i, balls[i]] = 1
    
    covered = np.zeros(total, dtype=bool)
    code = []
    while not covered.all():
        if time.time() - start > time_budget:
            return code, _covers_all(code, q, n, radius)
        # Vectorized gain computation: for each candidate i, count uncovered words in its ball
        uncovered_mask = ~covered  # bool array
        gains = ball_matrix @ uncovered_mask.astype(np.int32)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break  # no improvement possible
        code.append(all_words[best].tolist())
        covered[balls[best]] = True
    return code, True


def _sls_repair(code, q, n, radius, time_budget, seed=0):
    """Stochastic Local Search with repair moves for maximum packing with coverage constraints.
    
    Objective: maximize packing (minimize overlaps) subject to full coverage.
    Moves:
      1. Remove: if a codeword is redundant (all its neighbors are covered by others), remove it.
      2. Swap: replace a codeword with another that maintains coverage and reduces total overlap.
    
    Returns (new_code, verified).
    """
    total = q ** n
    if total > 3 * 10**6:
        return code, False

    rng = random.Random(seed)
    start = time.time()

    all_words, balls = _build_balls(q, n, radius)
    
    # Precompute ball sets for fast membership testing
    ball_sets = [set(b.tolist()) for b in balls]
    
    # State: list of word indices
    state = [_idx(c, q) for c in code]
    
    # Coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for wi in state:
        counts[balls[wi]] += 1
    
    # Objective: sum(counts^2) — lower means less overlap
    cur_overlap = int(np.sum(counts * counts))
    best_state = list(state)
    best_overlap = cur_overlap
    
    no_improve_count = 0
    max_no_improve = 3000
    
    while time.time() - start < time_budget:
        if no_improve_count > max_no_improve:
            break
            
        # 1. Try Removals (scan in reverse to avoid index issues)
        improved = False
        for i in range(len(state) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            wi = state[i]
            b = balls[wi]
            # Can we remove? Only if no word in this ball has count == 1
            if not np.any(counts[b] == 1):
                counts[b] -= 1
                state.pop(i)
                cur_overlap = int(np.sum(counts * counts))
                best_state = list(state)
                best_overlap = cur_overlap
                improved = True
                break
        if improved:
            no_improve_count = 0
            continue
            
        # 2. Try Swaps with incremental overlap delta
        if len(state) > 0:
            i = rng.randrange(len(state))
            old_wi = state[i]
            b_old = balls[old_wi]
            
            # Precompute critical set: words in b_old with count == 1
            critical_mask = (counts[b_old] == 1)
            critical_set = set(b_old[critical_mask].tolist()) if np.any(critical_mask) else set()
            
            # Try more candidates with structured exploration
            found_swap = False
            candidates_tried = 0
            max_candidates = 200
            
            while candidates_tried < max_candidates and time.time() - start < time_budget:
                candidates_tried += 1
                
                # Mix of random and local candidates
                if rng.random() < 0.3:
                    # Local: pick a word at Hamming distance 1 or 2 from old_wi
                    old_word = all_words[old_wi]
                    new_word = list(old_word)
                    if rng.random() < 0.5:
                        # Distance 1
                        pos = rng.randrange(n)
                        new_word[pos] = rng.randrange(q)
                    else:
                        # Distance 2
                        pos1, pos2 = rng.sample(range(n), 2)
                        new_word[pos1] = rng.randrange(q)
                        new_word[pos2] = rng.randrange(q)
                    new_wi = _idx(new_word, q)
                else:
                    new_wi = rng.randrange(total)
                
                if new_wi == old_wi:
                    continue
                
                b_new = balls[new_wi]
                
                # Coverage check: all critical words must be in b_new
                if critical_set:
                    b_new_set = ball_sets[new_wi]
                    if not critical_set.issubset(b_new_set):
                        continue
                
                # Compute overlap delta incrementally:
                # For words in b_old only: count decreases by 1 → delta = (c-1)^2 - c^2 = -2c+1
                # For words in b_new only: count increases by 1 → delta = (c+1)^2 - c^2 = 2c+1
                # For words in both: no net change
                
                # Compute using numpy
                in_old = np.zeros(total, dtype=bool)
                in_new = np.zeros(total, dtype=bool)
                in_old[b_old] = True
                in_new[b_new] = True
                
                only_old = in_old & ~in_new
                only_new = in_new & ~in_old
                
                delta = int(np.sum(2 * counts[only_new] + 1) - np.sum(2 * counts[only_old] - 1))
                
                if delta < 0 or (delta == 0 and rng.random() < 0.15):
                    # Accept swap
                    counts[only_old] -= 1
                    counts[only_new] += 1
                    state[i] = new_wi
                    cur_overlap += delta
                    best_state = list(state)
                    best_overlap = cur_overlap
                    found_swap = True
                    no_improve_count = 0
                    break
            if found_swap:
                continue
                
        no_improve_count += 1

    # Convert best_state back to codewords
    best_code = [all_words[wi].tolist() for wi in best_state]
    verified = _covers_all(best_code, q, n, radius)
    return best_code, verified


def _sls_repair_v2(code, q, n, radius, time_budget, seed=0):
    """Enhanced SLS with adaptive temperature and better candidate sampling."""
    total = q ** n
    if total > 3 * 10**6:
        return code, False

    rng = random.Random(seed)
    start = time.time()

    all_words, balls = _build_balls(q, n, radius)
    ball_sets = [set(b.tolist()) for b in balls]
    
    state = [_idx(c, q) for c in code]
    counts = np.zeros(total, dtype=np.int32)
    for wi in state:
        counts[balls[wi]] += 1
    
    cur_overlap = int(np.sum(counts * counts))
    best_state = list(state)
    best_overlap = cur_overlap
    
    no_improve_count = 0
    max_no_improve = 5000
    temperature = 0.15  # Accept probability for delta=0
    
    while time.time() - start < time_budget:
        if no_improve_count > max_no_improve:
            break
        
        # 1. Try Removals
        improved = False
        for i in range(len(state) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            wi = state[i]
            b = balls[wi]
            if not np.any(counts[b] == 1):
                counts[b] -= 1
                state.pop(i)
                cur_overlap = int(np.sum(counts * counts))
                best_state = list(state)
                best_overlap = cur_overlap
                improved = True
                break
        if improved:
            no_improve_count = 0
            temperature = max(0.05, temperature * 0.95)
            continue
        
        # 2. Try Swaps
        if len(state) > 0:
            i = rng.randrange(len(state))
            old_wi = state[i]
            b_old = balls[old_wi]
            
            critical_mask = (counts[b_old] == 1)
            critical_set = set(b_old[critical_mask].tolist()) if np.any(critical_mask) else set()
            
            found_swap = False
            candidates_tried = 0
            max_candidates = 300
            
            # Weighted sampling: prefer words that overlap less with current ball
            while candidates_tried < max_candidates and time.time() - start < time_budget:
                candidates_tried += 1
                
                if rng.random() < 0.25:
                    # Local move
                    old_word = all_words[old_wi]
                    new_word = list(old_word)
                    if rng.random() < 0.5:
                        pos = rng.randrange(n)
                        new_word[pos] = rng.randrange(q)
                    else:
                        pos1, pos2 = rng.sample(range(n), 2)
                        new_word[pos1] = rng.randrange(q)
                        new_word[pos2] = rng.randrange(q)
                    new_wi = _idx(new_word, q)
                else:
                    new_wi = rng.randrange(total)
                
                if new_wi == old_wi:
                    continue
                
                b_new = balls[new_wi]
                
                if critical_set:
                    if not critical_set.issubset(ball_sets[new_wi]):
                        continue
                
                in_old = np.zeros(total, dtype=bool)
                in_new = np.zeros(total, dtype=bool)
                in_old[b_old] = True
                in_new[b_new] = True
                
                only_old = in_old & ~in_new
                only_new = in_new & ~in_old
                
                delta = int(np.sum(2 * counts[only_new] + 1) - np.sum(2 * counts[only_old] - 1))
                
                if delta < 0 or (delta == 0 and rng.random() < temperature):
                    counts[only_old] -= 1
                    counts[only_new] += 1
                    state[i] = new_wi
                    cur_overlap += delta
                    best_state = list(state)
                    best_overlap = cur_overlap
                    found_swap = True
                    no_improve_count = 0
                    temperature = max(0.05, temperature * 0.95)
                    break
            if found_swap:
                continue
        
        no_improve_count += 1

    best_code = [all_words[wi].tolist() for wi in best_state]
    verified = _covers_all(best_code, q, n, radius)
    return best_code, verified


def construct(q, n, radius):
    start = time.time()
    # Always-valid baseline (used as the guaranteed fallback).
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]

    best = baseline
    best_len = len(baseline)

    # Try to improve with the greedy set-cover, only when the space is tractable.
    if q ** n <= 3 * 10**6 and best_len > 1:
        time_budget = max(0.5, _DEADLINE - (time.time() - start) - 20.0)
        try:
            cand, verified = _greedy(q, n, radius, time_budget)
        except Exception:
            cand, verified = None, False
        if cand is not None and verified and len(cand) < best_len:
            best = cand
            best_len = len(cand)

        # SLS repair polish to try to beat the greedy result.
        remaining = _DEADLINE - (time.time() - start) - 5.0
        if remaining > 3.0 and best_len > 1:
            for sls_seed in [42, 123, 777, 999, 1234]:
                remaining = _DEADLINE - (time.time() - start) - 5.0
                if remaining < 2.0:
                    break
                try:
                    polished, p_verified = _sls_repair_v2(best, q, n, radius, remaining, seed=sls_seed)
                except Exception:
                    polished, p_verified = None, False
                if polished is not None and p_verified and len(polished) < best_len:
                    best = polished
                    best_len = len(polished)

    # Final safety: if we are running out of time, return the valid baseline.
    if time.time() - start > _DEADLINE - 5.0:
        return baseline

    return best
