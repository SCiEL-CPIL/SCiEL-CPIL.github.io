import time
import random

def _check_add(values, x, sums):
    """Check if x can be appended to values without duplicate consecutive-block sums."""
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    if sums.intersection(tails):
        return False
    return True

def _greedy_search(values, sums, start, deadline, skip_prob=0.0):
    """Greedy search from current state with optional random skipping."""
    values = list(values)
    sums = set(sums)
    x = start
    while x <= 256:
        if time.time() > deadline:
            break
        if random.random() < skip_prob:
            x += 1
            continue
        if _check_add(values, x, sums):
            values.append(x)
            tails = [x]
            total = x
            for a in reversed(values[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
        x += 1
    return values

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best = []
    
    # First, run the deterministic greedy to get a baseline
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if _check_add(values, x, sums):
            values.append(x)
            tails = [x]
            total = x
            for a in reversed(values[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
    best = list(values)
    
    if time.time() > deadline:
        return best
    
    # Now do randomized restarts with varying skip probabilities
    random.seed(42)
    while time.time() < deadline:
        skip_prob = random.uniform(0.0, 0.3)
        result = _greedy_search([], set(), 1, deadline, skip_prob=skip_prob)
        if len(result) > len(best):
            best = result
    
    return best
