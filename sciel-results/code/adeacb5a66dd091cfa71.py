import time
import itertools
import random

def _deadline_hit(deadline, margin=0.0):
    return time.time() >= deadline - margin

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    n = 24
    # LCM of 1..24
    # 1..24 LCM is 5354228880
    L = 5354228880

    # Weights
    w = [L // i for i in range(1, n + 1)]

    # Precompute sums for all masks
    # This might be slow in pure python if not optimized
    # 2^24 = 16777216
    # We can use a list of size 2^24
    # But memory? 16M integers. ~128MB. OK.

    # To speed up sum calculation, we can use DP
    # sum_w[m] = sum_w[m & (m-1)] + w[bit]

    # However, iterating 16M times in Python takes time.
    # Let's try to do it efficiently.

    # Alternative:
    # We only need to find masks that sum to w[x] for x in 1..24.
    # We can use a dictionary: sum -> list of masks.
    # But we still need to generate all masks to populate it.

    # Let's try the DP approach for sums.
    # It's a single loop of 16M.

    # Actually, we can just iterate masks and calculate sum on the fly?
    # No, we need to group by sum.

    # Let's use the dictionary approach.
    # sum_to_masks = {}
    # for m in range(1, 1 << n):
    #    # Calculate sum
    #    # This is O(n) per mask if done naively.
    #    # O(n) * 2^n = 24 * 16M = 400M ops. Too slow.

    # We need the O(1) update.
    sum_w = [0] * (1 << n)
    # sum_w[0] = 0
    for m in range(1, 1 << n):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        sum_w[m] = sum_w[m ^ lsb] + w[idx]

    # Check time
    if time.time() > deadline - 10:
        return list(range(13, 25))

    # Group masks by sum
    sum_to_masks = {}
    for m in range(1, 1 << n):
        s = sum_w[m]
        if s in sum_to_masks:
            sum_to_masks[s].append(m)
        else:
            sum_to_masks[s] = [m]

    # Find forbidden masks
    forbidden = set()
    for x in range(1, n + 1):
        target = w[x - 1]
        if target in sum_to_masks:
            for m in sum_to_masks[target]:
                # Check if x is not in m
                if (m & (1 << (x - 1))) == 0:
                    # Forbidden set is m union {x}
                    fm = m | (1 << (x - 1))
                    forbidden.add(fm)

    # Remove supersets
    # Sort by popcount
    forbidden_list = sorted(forbidden, key=lambda x: bin(x).count('1'))
    minimal_forbidden = []
    for fm in forbidden_list:
        is_superset = False
        for mf in minimal_forbidden:
            if (fm & mf) == mf:
                is_superset = True
                break
        if not is_superset:
            minimal_forbidden.append(fm)

    # Now search for max valid set
    # Iterate sizes from n down to 1
    best_set = None

    # Optimization: Precompute bit counts?
    # We are using combinations, so we know the size.

    for k in range(n, 0, -1):
        if time.time() > deadline - 5:
            break

        # Generate combinations of size k
        # Using itertools.combinations on indices 0..23
        for comb in itertools.combinations(range(n), k):
            # Create mask
            mask = 0
            for i in comb:
                mask |= (1 << i)

            # Check validity
            valid = True
            for fm in minimal_forbidden:
                if (mask & fm) == fm:
                    valid = False
                    break

            if valid:
                best_set = comb
                break

        if best_set:
            break

    # Max-SAT / WalkSAT local search on the forbidden-clause structure
    # Variables: x_i in {0,1}, i=0..23. Clause: not(all bits in fm are 1).
    # Objective: maximize number of 1s subject to no clause being fully satisfied.
    # We start from the best_set found above (or a greedy seed) and do local flips.
    if best_set is None:
        best_set = tuple(range(n))  # all ones as a starting point for search

    best_mask = 0
    for i in best_set:
        best_mask |= (1 << i)

    # Helper: count violated clauses for a mask
    def violated_count(mask):
        vc = 0
        for fm in minimal_forbidden:
            if (mask & fm) == fm:
                vc += 1
        return vc

    # Helper: popcount
    def popcount(x):
        return bin(x).count('1')

    current_mask = best_mask
    current_vc = violated_count(current_mask)
    current_size = popcount(current_mask)

    # If current is valid, we already have a valid set; try to grow it via local search
    # If not valid, try to fix violations while maximizing size.
    # WalkSAT-style: flip variables to reduce violations, with bias toward increasing size.
    random.seed(12345)
    max_iters = 200000
    p_flip_off = 0.3  # probability to flip 1->0 when choosing a random flip among violated clauses

    for it in range(max_iters):
        if _deadline_hit(deadline, margin=3.0):
            break

        if current_vc == 0:
            # Valid: try to add a variable (flip 0->1) that doesn't create violations
            improved = False
            # Random order for diversity
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                if _deadline_hit(deadline, margin=3.0):
                    break
                if (current_mask & (1 << i)) == 0:
                    new_mask = current_mask | (1 << i)
                    new_vc = violated_count(new_mask)
                    if new_vc == 0:
                        current_mask = new_mask
                        current_vc = 0
                        current_size += 1
                        improved = True
                        break
            if not improved:
                # No single add works; break (local optimum for valid sets)
                break
        else:
            # Pick a random violated clause and flip a variable in it
            # Find violated clauses
            violated = [fm for fm in minimal_forbidden if (current_mask & fm) == fm]
            if not violated:
                # Shouldn't happen if current_vc > 0, but guard
                break
            fm = random.choice(violated)
            # Variables in this clause
            bits = []
            m = fm
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                bits.append(idx)
                m ^= lsb
            # Choose a variable to flip
            # Prefer flipping 1->0 to break the clause
            ones_in_clause = [i for i in bits if (current_mask & (1 << i))]
            zeros_in_clause = [i for i in bits if (current_mask & (1 << i)) == 0]

            if ones_in_clause:
                # Flip one of the 1s to 0
                i = random.choice(ones_in_clause)
                new_mask = current_mask & ~(1 << i)
            else:
                # All zeros? Then clause isn't violated. Pick a random bit and set to 1? No, that would create violation.
                # Actually if all zeros, clause is not violated. So this case shouldn't occur.
                # Just flip a random bit to 1 (might not help)
                i = random.choice(bits)
                new_mask = current_mask | (1 << i)

            new_vc = violated_count(new_mask)
            new_size = popcount(new_mask)

            # Accept if it reduces violations, or if same violations but increases size
            if new_vc < current_vc or (new_vc == current_vc and new_size > current_size):
                current_mask = new_mask
                current_vc = new_vc
                current_size = new_size
            else:
                # WalkSAT random walk: flip a random variable in the clause
                # Choose from all bits in clause
                i = random.choice(bits)
                if current_mask & (1 << i):
                    new_mask = current_mask & ~(1 << i)
                else:
                    new_mask = current_mask | (1 << i)
                new_vc = violated_count(new_mask)
                new_size = popcount(new_mask)
                # Accept with some probability to escape local minima
                if new_vc < current_vc or (new_vc == current_vc and new_size >= current_size):
                    current_mask = new_mask
                    current_vc = new_vc
                    current_size = new_size

    # After local search, if valid, update best
    if current_vc == 0:
        if current_size > popcount(best_mask):
            best_mask = current_mask
        elif popcount(best_mask) == 0:
            best_mask = current_mask

    # Convert best_mask to list
    if best_mask:
        result = [i + 1 for i in range(n) if (best_mask & (1 << i))]
        if result:
            return result

    # Fallback
    return [1]
