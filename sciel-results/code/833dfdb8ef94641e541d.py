import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    # Bitmask representation: bit i is set if i is in the set
    # We use 1..512, so we need bits 1..512.
    # Using a Python int as the bitmask.

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        count = 0
        for d in range(1, 256):
            # Shift mask right by d and 2d
            # If the bitwise AND of the original, shifted by d, and shifted by 2d is non-zero,
            # then there is at least one triplet.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        # Iterate through bits 1 to 512
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Generate initial candidates
    candidates = []

    # Random starts
    for _ in range(50):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        candidates.append(lst)

    # Structured starts
    # Arithmetic progression
    candidates.append(list(range(1, 1 + 31 * 16, 16)))
    # Dense at start
    candidates.append(list(range(1, 33)))
    # Dense at end
    candidates.append(list(range(481, 513)))
    # Another structured: powers of 2 (first 32 that fit? 2^0..2^31 doesn't fit)
    # Let's try a mix of low and high numbers
    low_half = random.sample(range(1, 257), 16)
    high_half = random.sample(range(257, 513), 16)
    candidates.append(low_half + high_half)

    # Evaluate initial candidates
    best_score = -1
    best_list = None
    best_mask = 0

    for lst in candidates:
        m = list_to_mask(lst)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m

    # Simulated Annealing
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask

    # Set for O(1) membership checks to speed up neighbor generation
    current_set = set(current_list)

    T = 10.0  # Initial temperature
    cooling_rate = 0.9995

    # Precompute a list of all valid values for faster sampling?
    # No, just sample and check.

    while time.time() < deadline:
        # Generate a neighbor
        # Pick a random index to change
        idx = random.randint(0, N - 1)
        old_val = current_list[idx]

        # Pick a new value
        # Sample from 1..512 until we get a valid one not in the set
        # Since set size is 32 out of 512, rejection sampling is very fast
        while True:
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val:
                continue
            if new_val in current_set:
                continue
            break

        # Calculate delta score
        # Remove old_val, add new_val.
        new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)

        new_score = score_mask(new_mask)
        delta = new_score - current_score

        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # Boltzmann factor
            if T < 1e-10:
                accept = False
            else:
                # exp(delta / T)
                # delta is negative, T is positive
                # If delta is very large negative, exp is 0
                accept = random.random() < math.exp(delta / T)

        if accept:
            # Update state
            current_list[idx] = new_val
            current_set.remove(old_val)
            current_set.add(new_val)
            current_score = new_score
            current_mask = new_mask

            if current_score > best_score:
                best_score = current_score
                best_list = current_list[:]
                best_mask = current_mask

        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_list
# EVOLVE-BLOCK-END
