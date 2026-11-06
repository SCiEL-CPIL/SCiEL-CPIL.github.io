"""Rewritten program: multi-start lookahead greedy with time guard."""
import time
import random

def _check_valid(seq):
    """Verify strictly increasing, in [1,256], and all consecutive-block sums distinct."""
    n = len(seq)
    if n == 0:
        return False
    for i in range(n):
        if seq[i] < 1 or seq[i] > 256:
            return False
        if i > 0 and seq[i] <= seq[i - 1]:
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

def _greedy_build(preferred_order, lookahead=2):
    """
    Build a strictly increasing sequence with distinct consecutive-block sums.
    preferred_order: list of candidate values in the order we want to try them.
    lookahead: how many future candidates to examine when choosing the next element.
    Returns the constructed sequence.
    """
    values = []
    sums = set()
    # Index into preferred_order for candidates not yet tried
    candidates = list(preferred_order)
    
    while candidates:
        best_val = None
        best_sums = None
        # Look ahead up to `lookahead` candidates to find the one that
        # leaves the most flexibility (fewest future conflicts)
        for li in range(min(lookahead, len(candidates))):
            x = candidates[li]
            # Check if x can be appended
            if values and x <= values[-1]:
                continue
            # Compute new tails
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if sums.intersection(tails):
                continue
            # x is valid. Score it by how many future candidates it blocks.
            # A candidate y > x is blocked if adding y after x would create a sum collision.
            blocked = 0
            for y in candidates[li + 1:]:
                if y <= x:
                    continue
                # Simulate adding x then y
                # After adding x, values' becomes values + [x], sums' = sums | tails
                # Then check y against values+[x] and sums|tails
                tails_y = [y]
                total_y = y
                # reversed(values + [x]) = [x] + reversed(values)
                for a in [x] + list(reversed(values)):
                    total_y += a
                    tails_y.append(total_y)
                if (sums | set(tails)).intersection(tails_y):
                    blocked += 1
            # Prefer candidate with fewer blocked future options
            if best_val is None or blocked < best_sums:
                best_val = x
                best_sums = blocked
                if blocked == 0:
                    break  # Can't do better
        
        if best_val is None:
            # No candidate in the lookahead window works; try the rest
            for x in candidates:
                if values and x <= values[-1]:
                    continue
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    best_val = x
                    best_sums = 0
                    break
        
        if best_val is None:
            break
        
        # Commit best_val
        values.append(best_val)
        tails = [best_val]
        total = best_val
        for a in reversed(values[:-1]):
            total += a
            tails.append(total)
        sums.update(tails)
        
        # Remove best_val from candidates
        candidates.remove(best_val)
    
    return values

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    best_seq = []
    
    # Strategy 1: smallest-first greedy (baseline)
    seq1 = _greedy_build(list(range(1, 257)), lookahead=1)
    if len(seq1) > len(best_seq):
        best_seq = seq1
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 2: smallest-first with lookahead
    seq2 = _greedy_build(list(range(1, 257)), lookahead=3)
    if len(seq2) > len(best_seq):
        best_seq = seq2
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 3: largest-first greedy
    seq3 = _greedy_build(list(range(256, 0, -1)), lookahead=1)
    if len(seq3) > len(best_seq):
        best_seq = seq3
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 4: largest-first with lookahead
    seq4 = _greedy_build(list(range(256, 0, -1)), lookahead=3)
    if len(seq4) > len(best_seq):
        best_seq = seq4
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 5: random restarts with different orderings
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        rng.shuffle(order)
        seq = _greedy_build(order, lookahead=2)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # Final validation
    if _check_valid(best_seq):
        return best_seq
    
    # Fallback: simplest valid sequence
    values = [1]
    sums = {1}
    for x in range(2, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values
