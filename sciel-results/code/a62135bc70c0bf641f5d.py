"""Rewritten program: multi-strategy randomized greedy for unique consecutive-block sums."""
# EVOLVE-BLOCK-START
import time
import random

def _valid(seq):
    """Check strictly increasing and all consecutive-block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
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

def _greedy(order, start=0, end=257):
    """Greedy: iterate candidates in given order, add if it keeps block sums distinct."""
    values = []
    sums = set()
    for x in order:
        # compute new suffix sums
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_largest():
    # bias toward large values: spreads sums, keeps low range sparse
    return _greedy(range(256, 0, -1))

def _greedy_smallest():
    return _greedy(range(1, 257))

def _greedy_random(seed):
    rng = random.Random(seed)
    order = list(range(1, 257))
    rng.shuffle(order)
    return _greedy(order)

def _greedy_weighted(seed, bias):
    # weighted sampling: higher weight for values near bias (1=small, -1=large)
    rng = random.Random(seed)
    vals = list(range(1, 257))
    if bias >= 0:
        weights = [1.0 + bias * (256 - v) / 256.0 for v in vals]
    else:
        weights = [1.0 + (-bias) * v / 256.0 for v in vals]
    order = []
    pool = vals[:]
    weights_pool = weights[:]
    while pool:
        total = sum(weights_pool)
        r = rng.random() * total
        acc = 0
        idx = len(pool) - 1
        for i, w in enumerate(weights_pool):
            acc += w
            if r <= acc:
                idx = i
                break
        order.append(pool.pop(idx))
        weights_pool.pop(idx)
    return _greedy(order)

def _beam_search(beam_width=50, max_depth=20, deadline=None):
    """Beam search to explore non-greedy branches."""
    if deadline is None:
        deadline = time.time() + 45.0
        
    # State: (seq, sums_set)
    beams = [([], set())]
    
    for _ in range(max_depth):
        if time.time() > deadline:
            break
        next_beams = []
        for seq, sums in beams:
            used = set(seq)
            # Try all unused numbers
            # Optimization: We can iterate 1..256
            for x in range(1, 257):
                if x in used:
                    continue
                # Check validity: new suffix sums must not intersect existing sums
                total = x
                valid = True
                if total in sums:
                    valid = False
                else:
                    for a in reversed(seq):
                        total += a
                        if total in sums:
                            valid = False
                            break
                
                if valid:
                    # Construct new sums set
                    new_sums = set(sums)
                    total = x
                    new_sums.add(total)
                    for a in reversed(seq):
                        total += a
                        new_sums.add(total)
                    
                    # Heuristic: Prefer larger x to spread sums (based on knowledge)
                    score = x
                    next_beams.append((score, seq + [x], new_sums))
        
        if not next_beams:
            break
        
        # Sort by score (descending) and keep top beam_width
        # Using a set to deduplicate identical sequences (though unlikely with unique x)
        # next_beams contains (score, seq, sums)
        next_beams.sort(key=lambda t: t[0], reverse=True)
        
        # Keep top beam_width
        beams = [(seq, sums) for _, seq, sums in next_beams[:beam_width]]
        
        if not beams:
            break
    
    # Return the longest sequence found
    if not beams:
        return []
    best_seq = max(beams, key=lambda t: len(t[0]))[0]
    return best_seq

def construct():
    # Set deadline to 50s to ensure we return before the 60s kill
    deadline = time.time() + 50.0

    best = []
    best_len = 0

    def consider(seq):
        nonlocal best, best_len
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)

    # Deterministic strategies first
    consider(_greedy_smallest())
    consider(_greedy_largest())

    # Structural Change: Beam Search
    if time.time() < deadline:
        consider(_beam_search(beam_width=30, max_depth=15, deadline=deadline))
        
    # Weighted biases (small bias and large bias)
    for seed in range(2):
        if time.time() > deadline:
            break
        consider(_greedy_weighted(seed, 0.5))   # favor large
        consider(_greedy_weighted(seed, -0.5))  # favor small

    # Randomized restarts with diverse sampling
    rng = random.Random(12345)
    seed = 0
    while time.time() < deadline - 5.0:  # Stop 5s before deadline to allow cleanup
        # alternate sampling strategies
        if seed % 3 == 0:
            seq = _greedy_largest()
        elif seed % 3 == 1:
            seq = _greedy_random(seed)
        else:
            seq = _greedy_weighted(seed, rng.uniform(-1.0, 1.0))
        consider(seq)
        seed += 1

    # Ensure a valid result is returned
    if not best:
        best = _greedy_smallest()

    # Final safety: validate
    if not _valid(best):
        # fall back to a guaranteed valid construction
        best = _greedy_smallest()

    return best
# EVOLVE-BLOCK-END
