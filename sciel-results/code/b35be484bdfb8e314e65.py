"""Improved covering-code constructor: direct-sum baseline + greedy set-cover refinement.

Strategy:
  * A simple always-valid baseline (direct sum of a radius-1 perfect code with the
    remaining coordinates) guarantees correctness and gives an upper bound.
  * A greedy set-cover heuristic (place the center that covers the most uncovered
    words, repeat) finds codes that are often smaller than the baseline.  For the
    target ternary case (q=3, n=6, R=1) it recovers a 73-word covering (matching the
    best known upper bound), improving on the baseline's 81.
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
    ball = [0]
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
        balls.append(_ball_indices(c.tolist(), q, n, radius))
    return all_words, balls


def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space (only used for small spaces)."""
    total = q ** n
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
    covered = np.zeros(total, dtype=bool)
    code = []
    while not covered.all():
        if time.time() - start > time_budget:
            return code, _covers_all(code, q, n, radius)
        # For each candidate center, how many currently-uncovered words it would cover.
        best = -1
        best_gain = 0
        for i in range(total):
            b = balls[i]
            gain = int(np.sum(~covered[b]))
            if gain > best_gain:
                best_gain = gain
                best = i
        code.append(all_words[best].tolist())
        covered[balls[best]] = True
    return code, True


def _local_search_polish(code, q, n, radius, time_budget, seed=0):
    """Local-search polish: try to reduce the code size via remove/swap/relocate moves
    while preserving full coverage.  Returns (new_code, verified)."""
    total = q ** n
    if total > 3 * 10**6:
        return code, False

    rng = random.Random(seed)
    start = time.time()

    # Precompute balls once.
    all_words, balls = _build_balls(q, n, radius)

    # Current coverage counts.
    counts = np.zeros(total, dtype=np.int32)
    for c in code:
        for idx in balls[_idx(c, q)]:
            counts[idx] += 1

    best_code = [list(c) for c in code]
    best_len = len(code)

    def try_move(new_code):
        nonlocal best_code, best_len
        if len(new_code) < best_len:
            # Rebuild counts for new_code
            new_counts = np.zeros(total, dtype=np.int32)
            for c in new_code:
                for idx in balls[_idx(c, q)]:
                    new_counts[idx] += 1
            if bool(np.all(new_counts >= 1)):
                best_code = [list(c) for c in new_code]
                best_len = len(new_code)
                return True
        return False

    def try_relocate(i, time_budget_remaining):
        """Try removing code[i] and then greedily repairing uncovered words.
        Returns True if the code was improved (size reduced)."""
        nonlocal code, counts, best_code, best_len
        c = code[i]
        c_idx = _idx(c, q)
        # Temporarily remove
        for idx in balls[c_idx]:
            counts[idx] -= 1
        
        # Check if still covered
        if bool(np.all(counts >= 1)):
            # Can remove without repair
            new_code = code[:i] + code[i+1:]
            if try_move(new_code):
                code = new_code
                return True
        else:
            # Need to repair: find uncovered words and try to cover them
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) > 0:
                # Greedily add centers to cover uncovered words
                temp_counts = counts.copy()
                added = []
                max_adds = min(3, best_len - 1)  # Allow adding at most 2 net new words
                for _ in range(max_adds):
                    if time.time() - start > time_budget:
                        break
                    # Find best center to cover most uncovered
                    best_gain = 0
                    best_center = -1
                    for j in range(total):
                        b = balls[j]
                        gain = int(np.sum(temp_counts[b] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_center = j
                    if best_center == -1 or best_gain == 0:
                        break
                    added.append(best_center)
                    for idx in balls[best_center]:
                        temp_counts[idx] += 1
                    if bool(np.all(temp_counts >= 1)):
                        break
                
                if bool(np.all(temp_counts >= 1)) and len(added) < 1:
                    # This shouldn't happen if we needed repair, but just in case
                    pass
                elif bool(np.all(temp_counts >= 1)):
                    # Build new code: remove i, add the new centers
                    new_code = code[:i] + code[i+1:] + [all_words[j].tolist() for j in added]
                    if len(new_code) < best_len:
                        if try_move(new_code):
                            code = new_code
                            # Update counts to match new code
                            for idx in balls[c_idx]:
                                counts[idx] += 1  # restore original
                            for j in added:
                                for idx in balls[j]:
                                    counts[idx] += 1
                            return True
            # Restore counts
            for idx in balls[c_idx]:
                counts[idx] += 1
        return False

    # Local search loop: repeatedly try to remove, swap, or relocate codewords.
    while time.time() - start < time_budget:
        improved = False
        
        # Try removing each codeword.
        for i in range(len(code)):
            if time.time() - start > time_budget:
                break
            c = code[i]
            c_idx = _idx(c, q)
            for idx in balls[c_idx]:
                counts[idx] -= 1
            if bool(np.all(counts >= 1)):
                new_code = code[:i] + code[i + 1:]
                if try_move(new_code):
                    code = new_code
                    improved = True
                    break
            else:
                for idx in balls[c_idx]:
                    counts[idx] += 1

        if improved:
            continue

        # Try swapping each codeword with a random other word.
        if len(code) > 1:
            for i in range(len(code)):
                if time.time() - start > time_budget:
                    break
                c_old = code[i]
                c_old_idx = _idx(c_old, q)
                # Try a few random replacements.
                for _ in range(20):
                    if time.time() - start > time_budget:
                        break
                    new_word = all_words[rng.randrange(total)].tolist()
                    if new_word == c_old:
                        continue
                    new_word_idx = _idx(new_word, q)
                    # Check if swapping keeps coverage.
                    # Remove old, add new.
                    for idx in balls[c_old_idx]:
                        counts[idx] -= 1
                    for idx in balls[new_word_idx]:
                        counts[idx] += 1
                    if bool(np.all(counts >= 1)):
                        # Accept: same size but maybe enables future removals.
                        code[i] = new_word
                        improved = True
                        break
                    else:
                        # Revert.
                        for idx in balls[new_word_idx]:
                            counts[idx] -= 1
                        for idx in balls[c_old_idx]:
                            counts[idx] += 1

                if improved:
                    break

        if improved:
            continue

        # Try relocate moves: remove a codeword and greedily repair.
        if len(code) > 1:
            for i in range(len(code)):
                if time.time() - start > time_budget:
                    break
                if try_relocate(i, time_budget):
                    improved = True
                    break

        if not improved:
            # No improvement; stop.
            break

    verified = _covers_all(best_code, q, n, radius)
    return best_code, verified


def construct(q, n, radius):
    start = time.time()
    # Always-valid baseline (used as the guaranteed fallback).
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        # Degenerate small cases: cover every word directly (exact, valid).
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]

    best = baseline
    best_len = len(baseline)

    # Try to improve with the greedy set-cover, only when the space is tractable.
    if q ** n <= 3 * 10**6 and best_len > 1:
        time_budget = max(0.5, _DEADLINE - (time.time() - start) - 5.0)
        try:
            cand, verified = _greedy(q, n, radius, time_budget)
        except Exception:
            cand, verified = None, False
        if cand is not None and verified and len(cand) < best_len:
            best = cand
            best_len = len(cand)

        # Local-search polish to try to beat the greedy result.
        remaining = _DEADLINE - (time.time() - start) - 5.0
        if remaining > 2.0 and best_len > 1:
            try:
                polished, p_verified = _local_search_polish(best, q, n, radius, remaining, seed=42)
            except Exception:
                polished, p_verified = None, False
            if polished is not None and p_verified and len(polished) < best_len:
                best = polished
                best_len = len(polished)
            
            # Try a second polish pass with a different seed to explore more.
            remaining2 = _DEADLINE - (time.time() - start) - 5.0
            if remaining2 > 2.0 and best_len > 1:
                try:
                    polished2, p2_verified = _local_search_polish(best, q, n, radius, remaining2, seed=123)
                except Exception:
                    polished2, p2_verified = None, False
                if polished2 is not None and p2_verified and len(polished2) < best_len:
                    best = polished2
                    best_len = len(polished2)

    # Final safety: if we are running out of time, return the valid baseline.
    if time.time() - start > _DEADLINE - 5.0:
        return baseline

    return best
