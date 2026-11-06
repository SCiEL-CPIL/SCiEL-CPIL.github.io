"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _block_sums_set(values):
    """Return set of all consecutive block sums."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _greedy(candidates):
    """Standard greedy: add candidates in order if no block sum conflict."""
    values = []
    sums = set()
    for x in candidates:
        if values and x <= values[-1]:
            continue
        tails = []
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _local_search(seq, max_rounds=50):
    """Try to insert elements to lengthen the sequence."""
    seq = list(seq)
    for _ in range(max_rounds):
        improved = False
        used = set(seq)
        for x in range(1, 257):
            if x in used:
                continue
            for pos in range(len(seq) + 1):
                if pos > 0 and x <= seq[pos - 1]:
                    continue
                if pos < len(seq) and x >= seq[pos]:
                    continue
                new_seq = seq[:pos] + [x] + seq[pos:]
                sums = _block_sums_set(new_seq)
                if len(sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                    seq = new_seq
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Baseline greedy
    best = _greedy(range(1, 257))
    best_len = len(best)
    
    # Local search from baseline
    improved = _local_search(best)
    if len(improved) > best_len:
        best = improved
        best_len = len(best)
    
    # Randomized restarts
    random.seed(42)
    while time.time() < deadline:
        cands = list(range(1, 257))
        random.shuffle(cands)
        seq = _greedy(cands)
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)
        elif len(seq) >= best_len - 2:
            improved = _local_search(seq)
            if len(improved) > best_len:
                best = improved
                best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
