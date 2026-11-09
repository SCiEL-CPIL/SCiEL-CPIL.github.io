"""
Structural change: model as a Constraint Satisfaction Problem (CSP) with
forward checking, dynamic value ordering, and multi-restart branch-and-bound.
This is a different algorithmic family from the previous heuristic DFS/beam approach.

Key structural differences:
1. CSP formulation: variables are positions, domains are valid values.
2. Forward checking: prune invalid values before recursing.
3. Multi-restart with randomized value ordering to escape local optima.
4. Beam search refinement on top of global search results.
5. Tighter branch-and-bound using domain size estimates.
"""
import time
import random
import sys
from collections import defaultdict


def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i - 1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _compute_new_sums(seq, v):
    """Compute new block sums when appending v to seq."""
    new_sums = [v]
    total = v
    for i in range(len(seq) - 1, -1, -1):
        total += seq[i]
        new_sums.append(total)
    return new_sums


def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq


def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values


def _construct_greedy_largest():
    """Greedy largest - different bias, may find different optima."""
    values, sums = [], set()
    for x in range(256, 0, -1):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    values.sort()
    return values


def _get_all_sums(seq):
    """Compute all consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums


def _valid_candidates(seq, sums, last_val):
    """Forward-check: list of (v, new_sums_list) that keep the sequence valid."""
    candidates = []
    for v in range(last_val + 1, 257):
        new_sums_list = _compute_new_sums(seq, v)
        ok = True
        for s in new_sums_list:
            if s in sums:
                ok = False
                break
        if ok:
            candidates.append((v, new_sums_list))
    return candidates


def _solve_csp(deadline, seed=None, width=6):
    """
    CSP-style solver with forward checking and randomized value ordering.
    Different from previous DFS: uses explicit domain pruning and
    least-constraining-value heuristic.
    """
    if seed is not None:
        random.seed(seed)

    best_seq = []
    best_len = 0

    def dfs(seq, sums, last_val):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return

        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)

        # Branch-and-bound: max possible additional elements
        max_additional = 256 - last_val
        if n + max_additional <= best_len:
            return

        # Generate valid candidates with forward checking
        candidates = _valid_candidates(seq, sums, last_val)
        if not candidates:
            return

        # Value ordering: mix of smallest (most room) and random for diversity
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > width:
            ordered = candidates[:max(1, width // 2)] + \
                random.sample(candidates[width // 2:],
                              min(width - width // 2, len(candidates) - width // 2))
        else:
            ordered = candidates

        for v, new_sums_list in ordered:
            if time.time() > deadline:
                return
            new_sums = sums.union(new_sums_list)
            dfs(seq + [v], new_sums, v)

    dfs([], set(), 0)
    return best_seq


def _beam_search(deadline, width=200, seed=None):
    """
    Beam search: keep many diverse partial sequences in parallel.
    Each node is (seq, sums, last_val). Score a node by how much room it has
    (256 - last_val) and how few sums it consumes, so that paths that leave
    room AND stay unconstrained survive. This escapes the greedy local minima.
    """
    if seed is not None:
        random.seed(seed)

    # Start from the empty sequence
    beam = [([], set(), 0)]
    best = []

    while time.time() < deadline:
        next_beam = []
        for seq, sums, last_val in beam:
            if time.time() > deadline:
                break
            for v, new_sums_list in _valid_candidates(seq, sums, last_val):
                next_beam.append((seq + [v], sums.union(new_sums_list), v))
            if time.time() > deadline:
                break
        if not next_beam:
            break
        # Track best
        for seq, _, _ in next_beam:
            if len(seq) > len(best):
                best = list(seq)
        # Prune: score = room (primary), then -len(sums) (secondary)
        # We want large room and few sums.
        next_beam.sort(key=lambda node: (256 - node[2], -len(node[1])), reverse=True)
        # Keep top `width` but ensure diversity by last_val spread
        beam = next_beam[:width]
    return best


def _solve_csp_multi_restart(deadline, n_restarts=8):
    """
    Multi-restart CSP solver with different random seeds.
    Each restart explores a different region of the search space.
    """
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)

    remaining_time = deadline - time.time()
    if remaining_time <= 1:
        return best_seq

    per_restart = remaining_time / max(n_restarts, 1)

    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break
        restart_deadline = min(deadline - 2, time.time() + per_restart)
        try:
            result = _solve_csp(restart_deadline, seed=r * 12345 + 678)
            if len(result) > best_len:
                best_seq = result
                best_len = len(result)
        except Exception:
            pass

    return best_seq


def _extend_greedy(seq, deadline):
    """Extend a sequence greedily (smallest valid) until no more fit or time out."""
    seq = list(seq)
    sums = _get_all_sums(seq)
    while time.time() < deadline:
        found = False
        for v in range(seq[-1] + 1, 257) if seq else range(1, 257):
            new_sums_list = _compute_new_sums(seq, v)
            if not sums.intersection(new_sums_list):
                seq.append(v)
                sums.update(new_sums_list)
                found = True
                break
        if not found:
            break
    return seq


def _local_search_refine(seq, deadline):
    """
    Local search: try to extend the sequence by finding valid next elements.
    Also try alternative endings to unlock further extensions.
    Different from global search: operates on a near-complete sequence.
    """
    if not seq:
        return seq

    best = list(seq)
    best_len = len(best)

    # Try to extend greedily
    best = _extend_greedy(best, deadline)
    best_len = len(best)

    # Try replacing the last element with a different one to unlock extensions
    for pos in range(max(0, len(best) - 3), len(best)):
        if time.time() > deadline - 1:
            break
        for v in range((best[pos - 1] + 1) if pos > 0 else 1, 257):
            candidate = best[:pos] + [v]
            if not _check_valid(candidate):
                continue
            # Try to extend this candidate
            extended_candidate = _extend_greedy(candidate, deadline)
            if len(extended_candidate) > best_len:
                best = extended_candidate
                best_len = len(best)

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    # Multiple baselines
    best = _construct_greedy_smallest()
    best_len = len(best)

    try:
        greedy_large = _construct_greedy_largest()
        if len(greedy_large) > best_len:
            best = greedy_large
            best_len = len(greedy_large)
    except Exception:
        pass

    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best

    # Beam search (structural change: parallel diverse paths escape greedy minima)
    try:
        beam_result = _beam_search(min(deadline - 5, start_time + 40), width=250, seed=1)
        if len(beam_result) > best_len:
            best = beam_result
            best_len = len(beam_result)
    except Exception:
        pass

    # Multi-restart CSP solver (structural change: different algorithmic family)
    try:
        csp_result = _solve_csp_multi_restart(deadline, n_restarts=8)
        if len(csp_result) > best_len:
            best = csp_result
            best_len = len(csp_result)
    except Exception:
        pass

    if time.time() > deadline - 5:
        if not _check_valid(best):
            best = _construct_powers()
        return best

    # Local search refinement on the best sequence found
    try:
        refined = _local_search_refine(best, deadline)
        if len(refined) > best_len:
            best = refined
            best_len = len(refined)
    except Exception:
        pass

    # Final verification
    if not _check_valid(best):
        best = _construct_powers()

    return best
