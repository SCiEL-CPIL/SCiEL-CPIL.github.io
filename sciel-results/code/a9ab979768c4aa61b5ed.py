"""Improved program using bitset-based scoring for high-throughput SA."""
import time
import random

# EVOLVE-BLOCK-START

def _score_bitset(bits):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d all in set,
    where bits is a Python int with bit i set iff element i is in the set.
    Elements are in 1..512, so we use bits 1..512 (bit 0 unused).
    
    For each d in 1..256, check if (bits & (bits >> d) & (bits >> (2*d))) != 0.
    """
    count = 0
    b = bits
    # d ranges from 1 to 256 (since max element is 512, a+2d <= 512 means d <= 255, but let's go to 256 to be safe)
    for d in range(1, 257):
        if (b & (b >> d) & (b >> (2 * d))) != 0:
            count += 1
    return count


def _set_to_bits(s):
    """Convert a set of integers to a bitset integer."""
    bits = 0
    for x in s:
        bits |= (1 << x)
    return bits


def _bits_to_set(bits):
    """Convert a bitset integer back to a set of integers."""
    s = set()
    while bits:
        # Find lowest set bit
        lb = bits & (-bits)
        idx = lb.bit_length() - 1
        s.add(idx)
        bits ^= lb
    return s


def _greedy_construct():
    """
    Greedy construction: start with empty set, add elements that maximize score increment.
    """
    bits = 0
    s = set()
    for _ in range(32):
        best_elem = -1
        best_score = -1
        for candidate in range(1, 513):
            if candidate in s:
                continue
            new_bits = bits | (1 << candidate)
            sc = _score_bitset(new_bits)
            if sc > best_score:
                best_score = sc
                best_elem = candidate
        s.add(best_elem)
        bits |= (1 << best_elem)
    return s, _score_bitset(bits)


def _local_search(bits, s, current_score, deadline, rng):
    """
    Fast local search using bitset scoring.
    Tries swapping one element for another.
    """
    best_bits = bits
    best_s = set(s)
    best_score = current_score
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random element to remove
        elems = list(s)
        idx = rng.randrange(32)
        removed = elems[idx]
        
        # Pick a random element to add
        new_elem = rng.randint(1, 512)
        if new_elem == removed:
            continue
        if new_elem in s:
            continue
        
        new_bits = bits & ~(1 << removed) | (1 << new_elem)
        new_score = _score_bitset(new_bits)
        
        if new_score > best_score:
            best_score = new_score
            best_bits = new_bits
            best_s = set(s)
            best_s.discard(removed)
            best_s.add(new_elem)
            bits = new_bits
            s = best_s
            no_improve = 0
        else:
            no_improve += 1
    
    return best_bits, best_s, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    s, score = _greedy_construct()
    bits = _set_to_bits(s)
    
    best_bits = bits
    best_s = set(s)
    best_score = score
    
    # Phase 2: Local search with restarts
    remaining_phases = 5
    phase_time = (deadline - time.time()) / remaining_phases
    
    for phase in range(remaining_phases):
        phase_deadline = min(time.time() + phase_time, deadline)
        
        # Perturb: randomly swap a few elements
        perturbed_s = set(s)
        for _ in range(rng.randint(2, 5)):
            elems = list(perturbed_s)
            idx = rng.randrange(32)
            old_elem = elems[idx]
            new_elem = rng.randint(1, 512)
            if new_elem not in perturbed_s:
                perturbed_s.discard(old_elem)
                perturbed_s.add(new_elem)
        
        perturbed_bits = _set_to_bits(perturbed_s)
        perturbed_score = _score_bitset(perturbed_bits)
        
        # Local search from perturbed state
        ls_bits, ls_s, ls_score = _local_search(
            perturbed_bits, perturbed_s, perturbed_score, phase_deadline, rng
        )
        
        if ls_score > best_score:
            best_score = ls_score
            best_bits = ls_bits
            best_s = ls_s
            s = ls_s
            bits = ls_bits
        else:
            # Keep the best for next restart
            s = best_s
            bits = best_bits
    
    # Final polish with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        final_bits, final_s, final_score = _local_search(
            bits, s, best_score, deadline, rng
        )
        if final_score > best_score:
            best_score = final_score
            best_bits = final_bits
            best_s = final_s
    
    return sorted(best_s)

# EVOLVE-BLOCK-END
