"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _block_sums(seq):
    """Return set of all nonempty consecutive block sums for seq (list)."""
    s = set()
    n = len(seq)
    for i in range(n):
        t = 0
        for j in range(i, n):
            t += seq[j]
            s.add(t)
    return s

def _try_insert(seq, x, pos):
    """Try inserting x at pos (0..len(seq)). Return new seq if valid (strictly inc + distinct block sums), else None."""
    n = len(seq)
    # check strictly increasing
    if pos > 0 and seq[pos-1] >= x:
        return None
    if pos < n and seq[pos] <= x:
        return None
    new_seq = seq[:pos] + [x] + seq[pos:]
    # compute block sums
    sums = set()
    m = len(new_seq)
    for i in range(m):
        t = 0
        for j in range(i, m):
            t += new_seq[j]
            if t in sums:
                return None
            sums.add(t)
    return new_seq

def _greedy_base():
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _extend_by_insertion(seq, deadline):
    """Repeatedly try to insert a new element anywhere to grow seq. Return best seq found."""
    best = list(seq)
    best_len = len(best)
    # maintain current sums
    cur_sums = _block_sums(best)
    # We'll iterate: find smallest x in [1,256] not in best, try all positions
    # To be efficient, precompute for each x the set of "new tails" it would create at each pos
    # But positions depend on current seq. Just loop.
    while time.time() < deadline - 1.0:
        improved = False
        used = set(best)
        for x in range(1, 257):
            if x in used:
                continue
            # try inserting at each position
            for pos in range(len(best) + 1):
                if pos > 0 and best[pos-1] >= x:
                    continue
                if pos < len(best) and best[pos] <= x:
                    continue
                ns = _try_insert(best, x, pos)
                if ns is not None:
                    best = ns
                    cur_sums = _block_sums(best)
                    improved = True
                    if len(best) > best_len:
                        best_len = len(best)
                    break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    base = _greedy_base()
    best = list(base)
    best_len = len(best)

    # Phase 1: greedy insertion extension
    cand = _extend_by_insertion(base, deadline)
    if len(cand) > best_len:
        best = cand
        best_len = len(cand)

    # Phase 2: try random restarts with different greedy orders to find longer seeds
    if time.time() < deadline - 2.0:
        random.seed(12345)
        for trial in range(50):
            if time.time() > deadline - 2.0:
                break
            # Build a seed by greedy but with a shuffled candidate order bias
            # Actually just perturb: start from base, remove a random element, re-extend
            if len(best) >= 2:
                seq2 = list(best)
                # remove a random middle element
                idx = random.randint(1, len(seq2) - 2)
                seq2.pop(idx)
                cand2 = _extend_by_insertion(seq2, deadline)
                if len(cand2) > best_len:
                    best = cand2
                    best_len = len(cand2)

    return best

# EVOLVE-BLOCK-END
