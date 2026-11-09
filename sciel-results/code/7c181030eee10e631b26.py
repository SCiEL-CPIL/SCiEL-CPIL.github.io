# EVOLVE-BLOCK-START
import time
import sys

def construct(n):
    """
    Construct a superpermutation for n symbols.
    Returns a string of digits 1..n such that every permutation appears as a substring of length n.
    """
    deadline = time.time() + 115  # safety margin before 120s kill
    start_time = time.time()
    
    # For small n, use known optimal constructions
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return _standard_construction(4)
    if n == 5:
        return _standard_construction(5)
    if n == 6:
        return _standard_construction(6)
    
    # For n >= 7: build up to n-1 using standard construction,
    # then optimize the final stage with local search
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        word = _expand_step(word, m)
    
    # Final stage: add symbol n
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    if not perms:
        return "".join(map(str, word))
    
    # Build with standard order first
    baseline_word = _build_final_stage(perms, m)
    baseline_len = len(baseline_word)
    
    # Local search: try swapping adjacent permutations to reduce length
    if n <= 7 and time.time() < deadline - 10:
        best_order = list(perms)
        best_len = baseline_len
        best_word = baseline_word
        improved = True
        max_rounds = 200  # limit total rounds to stay within time
        
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(len(best_order) - 1):
                if time.time() > deadline - 10:
                    break
                # Try swapping i and i+1
                best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
                candidate = _build_final_stage(best_order, m)
                if len(candidate) < best_len:
                    best_len = len(candidate)
                    best_word = candidate
                    improved = True
                    # Don't break; continue trying more swaps in this round
                else:
                    # Swap back
                    best_order[i], best_order[i + 1] = best_order[i + 1], best_order[i]
            
            if not improved:
                break
        
        return "".join(map(str, best_word))
    else:
        return "".join(map(str, baseline_word))


def _standard_construction(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        word = _expand_step(word, m)
    return "".join(map(str, word))


def _expand_step(word, m):
    """Expand word from level m-1 to level m using standard construction."""
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        overlap = _find_overlap(new, block)
        new += block[overlap:]
    return new


def _build_final_stage(order, m):
    """Build the final stage word given an ordering of (m-1)-permutations."""
    new = []
    for p in order:
        block = list(p) + [m] + list(p)
        overlap = _find_overlap(new, block)
        new += block[overlap:]
    return new


def _find_overlap(current, block):
    """Find the maximum overlap between end of current and start of block."""
    if not current:
        return 0
    max_overlap = min(len(current), len(block))
    for o in range(max_overlap, -1, -1):
        if o == 0:
            return 0
        if current[len(current) - o:] == block[:o]:
            return o
    return 0
# EVOLVE-BLOCK-END
