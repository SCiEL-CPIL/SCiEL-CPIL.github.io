"""Optimized search for increasing sequences with unique consecutive-block sums."""
import time
import random

def _block_sums_for_append(values, x):
    """Compute the new block sums if we append x to values."""
    tails = []
    total = x
    tails.append(total)
    for a in reversed(values):
        total += a
        tails.append(total)
    return tails

def _greedy_build():
    """Fast greedy construction."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = _block_sums_for_append(values, x)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _random_greedy(rng, skip_prob=0.0):
    """Randomized greedy: in increasing order, randomly skip some candidates."""
    values, sums = [], set()
    for x in range(1, 257):
        if skip_prob > 0 and rng.random() < skip_prob:
            continue
        tails = _block_sums_for_append(values, x)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _local_search_extend(values, sums, rng, deadline):
    """Try to extend the sequence by trying to insert elements or modify."""
    # Try to find a longer sequence by backtracking from the end
    # Remove last few elements and try different choices
    n = len(values)
    if n < 2:
        return values, sums
    
    # Try removing last k elements and rebuilding
    for k in range(1, min(4, n)):
        if time.time() >= deadline:
            break
        prefix = values[:-k]
        prefix_sums = set()
        # Recompute sums for prefix
        m = len(prefix)
        for i in range(m):
            s = 0
            for j in range(i, m):
                s += prefix[j]
                prefix_sums.add(s)
        
        # Now try to extend from prefix with different choices
        # Try all possible next elements
        last = prefix[-1] if prefix else 0
        best_ext = []
        best_sums = set()
        
        # Try greedy from here with some randomness
        for trial in range(20):
            vals, sm = list(prefix), set(prefix_sums)
            for x in range(last + 1, 257):
                if trial > 0 and rng.random() < 0.1:
                    continue
                tails = _block_sums_for_append(vals, x)
                if not sm.intersection(tails):
                    vals.append(x)
                    sm.update(tails)
            if len(vals) > len(best_ext):
                best_ext = vals
                best_sums = sm
        
        if len(best_ext) > len(values):
            return best_ext, best_sums
    
    return values, sums

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Start with greedy baseline
    best = _greedy_build()
    best_len = len(best)
    
    # Compute sums for best
    def compute_sums(seq):
        s = set()
        n = len(seq)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += seq[j]
                s.add(acc)
        return s
    
    best_sums = compute_sums(best)
    
    rng = random.Random(12345)
    
    # Try randomized greedy with various skip probabilities
    for skip_prob in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        if time.time() >= deadline - 5:
            break
        for trial in range(50):
            if time.time() >= deadline - 5:
                break
            vals = _random_greedy(rng, skip_prob)
            if len(vals) > best_len:
                best = vals
                best_len = len(vals)
                best_sums = compute_sums(best)
    
    # Local search on best
    for _ in range(100):
        if time.time() >= deadline - 5:
            break
        new_best, new_sums = _local_search_extend(best, best_sums, rng, deadline)
        if len(new_best) > best_len:
            best = new_best
            best_len = len(new_best)
            best_sums = new_sums
    
    # Final verification
    if not _valid(best):
        # Fallback to greedy
        best = _greedy_build()
    
    return best

def _valid(seq):
    """Check strictly increasing in [1,256] and all consecutive-block sums distinct."""
    if not seq:
        return False
    if len(set(seq)) != len(seq):
        return False
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for a, b in zip(seq, seq[1:]):
        if a >= b:
            return False
    n = len(seq)
    expected = n * (n + 1) // 2
    return len(compute_sums(seq)) == expected

def compute_sums(seq):
    s = set()
    n = len(seq)
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += seq[j]
            s.add(acc)
    return s
