"""Optimized construction for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Bitmask representation: 512 bits, bit i set means (i+1) is in the set
    # We use a Python integer as a 512-bit mask

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << (x - 1))
        return mask

    def mask_to_list(mask):
        return [i + 1 for i in range(512) if (mask >> i) & 1]

    def score_mask(mask):
        """Count distinct d>0 such that a, a+d, a+2d all in S, using bitmask."""
        # For each d from 1 to 255, check if there exists a such that
        # bits a, a+d, a+2d are all set (where a is 0-indexed)
        count = 0
        for d in range(1, 256):
            # a ranges from 0 to 511-2d
            # Check: (mask & (mask >> d) & (mask >> (2*d))) != 0
            # But we need to ensure a+2d < 512, so a < 512-2d
            # The shift handles this: if a+2d >= 512, that bit won't be set
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    # Initial candidates
    candidates = []

    # Candidate 1: First 32 integers
    candidates.append(list(range(1, 33)))

    # Candidate 2: Structured spread
    c2 = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23,
          32, 33, 34, 35, 36, 37, 38, 39, 64, 65, 66, 67, 68, 69, 70, 71]
    candidates.append(c2)

    # Candidate 3: Mix
    c3 = list(range(1, 17)) + [i * 16 + 8 for i in range(16)]
    c3 = [x for x in c3 if 1 <= x <= 512]
    if len(c3) == 32:
        candidates.append(c3)

    # Evaluate initial candidates
    best_mask = None
    best_score = 0
    for cand in candidates:
        if len(cand) == 32 and all(1 <= x <= 512 for x in cand) and len(set(cand)) == 32:
            m = list_to_mask(cand)
            s = score_mask(m)
            if s > best_score:
                best_score = s
                best_mask = m

    if best_mask is None:
        best_mask = list_to_mask(list(range(1, 33)))
        best_score = score_mask(best_mask)

    random.seed(42)

    # Simulated Annealing
    current_mask = best_mask
    current_score = best_score
    best_mask_sa = best_mask
    best_score_sa = best_score

    T = 10.0  # initial temperature
    T_min = 0.01
    alpha = 0.9997  # cooling rate
    max_iter = 500000

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Generate a random move: remove one element, add another
        # Pick a random position to remove
        rem_bit = random.randint(0, 511)
        if not (current_mask >> rem_bit) & 1:
            # Not set, pick a different approach: pick a set bit
            set_bits = [i for i in range(512) if (current_mask >> i) & 1]
            if not set_bits:
                break
            rem_bit = random.choice(set_bits)

        # Pick a random position to add
        add_bit = random.randint(0, 511)
        if (current_mask >> add_bit) & 1:
            # Already set, try again or skip
            if random.random() < 0.5:
                continue

        new_mask = current_mask & ~(1 << rem_bit)
        new_mask |= (1 << add_bit)

        # Check uniqueness (should be 32 bits set)
        if bin(new_mask).count('1') != 32:
            continue

        new_score = score_mask(new_mask)
        delta = new_score - current_score

        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, T_min)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_mask_sa = current_mask

        T *= alpha
        if T < T_min:
            T = T_min

        # Occasionally restart with a slightly perturbed version of best
        if it % 10000 == 0 and it > 0:
            if random.random() < 0.3:
                # Perturb best
                pm = best_mask_sa
                for _ in range(random.randint(1, 3)):
                    rb = random.randint(0, 511)
                    if (pm >> rb) & 1:
                        pm &= ~(1 << rb)
                    else:
                        # Find a set bit to remove
                        set_bits = [i for i in range(512) if (pm >> i) & 1]
                        if set_bits:
                            rm = random.choice(set_bits)
                            pm &= ~(1 << rm)
                        else:
                            break
                    ab = random.randint(0, 511)
                    if not (pm >> ab) & 1:
                        pm |= (1 << ab)
                if bin(pm).count('1') == 32:
                    current_mask = pm
                    current_score = score_mask(current_mask)

    # Final polish: try targeted moves on the best solution
    if time.time() < deadline - 3:
        current_mask = best_mask_sa
        current_score = best_score_sa
        S_set = set(mask_to_list(current_mask))

        for trial in range(5000):
            if time.time() > deadline - 1:
                break

            # Pick an element to remove
            elems = mask_to_list(current_mask)
            rem_idx = random.randint(0, 31)
            rem_val = elems[rem_idx]
            rem_bit = rem_val - 1

            # Find candidate additions that would create new 3-APs
            candidates_add = set()
            for a in elems:
                if a == rem_val:
                    continue
                for b in elems:
                    if b == rem_val or b == a:
                        continue
                    # c = 2b - a
                    c = 2 * b - a
                    if 1 <= c <= 512 and c not in S_set and c != rem_val:
                        candidates_add.add(c)
                    # c = (a+b)/2
                    if (a + b) % 2 == 0:
                        c = (a + b) // 2
                        if 1 <= c <= 512 and c not in S_set and c != rem_val:
                            candidates_add.add(c)
                    # c = 2a - b
                    c = 2 * a - b
                    if 1 <= c <= 512 and c not in S_set and c != rem_val:
                        candidates_add.add(c)

            # Try a few candidates
            for add_val in list(candidates_add)[:15]:
                new_mask = current_mask & ~(1 << rem_bit)
                new_mask |= (1 << (add_val - 1))
                if bin(new_mask).count('1') != 32:
                    continue
                s = score_mask(new_mask)
                if s > current_score:
                    current_mask = new_mask
                    current_score = s
                    S_set = set(mask_to_list(current_mask))
                    if current_score > best_score_sa:
                        best_score_sa = current_score
                        best_mask_sa = current_mask
                    break

    # Return the best set found
    best_set = mask_to_list(best_mask_sa)
    if len(best_set) != 32:
        best_set = list(range(1, 33))
    return best_set
