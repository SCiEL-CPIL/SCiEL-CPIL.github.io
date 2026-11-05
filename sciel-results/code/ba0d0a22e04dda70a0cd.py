"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _greedy_build(candidate_order, used_sums=None, deadline=None):
    """Build a valid sequence by greedily selecting from candidate_order."""
    values = []
    sums = set()
    for x in candidate_order:
        if deadline and time.time() > deadline:
            break
        # Compute all new consecutive block sums if we append x
        new_sums = [x]
        total = x
        for a in reversed(values):
            total += a
            new_sums.append(total)
        # Check for conflicts
        conflict = False
        for s in new_sums:
            if s in sums:
                conflict = True
                break
        if not conflict:
            values.append(x)
            sums.update(new_sums)
    return values

def _greedy_forward():
    """Standard greedy: try candidates in ascending order."""
    return _greedy_build(range(1, 257))

def _greedy_backward():
    """Try candidates in descending order first, then fill gaps."""
    candidates = list(range(256, 0, -1))
    return _greedy_build(candidates)

def _randomized_trial(seed_val, deadline=None):
    """Randomized trial with shuffled candidate order."""
    random.seed(seed_val)
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    return _greedy_build(candidates, deadline=deadline)

def _biased_randomized_trial(seed_val, bias_low, bias_high, deadline=None):
    """Trial with biased sampling: prefer values in [bias_low, bias_high]."""
    random.seed(seed_val)
    candidates = list(range(1, 257))
    # Create weighted order: values in preferred range come first
    preferred = [x for x in candidates if bias_low <= x <= bias_high]
    others = [x for x in candidates if not (bias_low <= x <= bias_high)]
    random.shuffle(preferred)
    random.shuffle(others)
    candidates = preferred + others
    return _greedy_build(candidates, deadline=deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # First, get the baseline greedy result
    best = _greedy_forward()
    best_len = len(best)
    
    # Try backward greedy
    result = _greedy_backward()
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Try multiple randomized trials
    trial_count = 0
    while time.time() < deadline:
        trial_count += 1
        # Alternate between pure random and biased approaches
        if trial_count % 3 == 1:
            result = _randomized_trial(trial_count, deadline)
        elif trial_count % 3 == 2:
            # Bias toward lower values (smaller sums, more room)
            result = _biased_randomized_trial(trial_count + 1000, 1, 80, deadline)
        else:
            # Bias toward higher values
            result = _biased_randomized_trial(trial_count + 2000, 180, 256, deadline)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best
# EVOLVE-BLOCK-END
