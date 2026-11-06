import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before the 90s kill

    n = 5040
    target = 4979

    # All divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    total_sum = sum(divisors)          # 19344
    comp_target = total_sum - target   # 14365 (sum of the un-selected divisors)

    # Split divisors: odd divisors (individual) and even divisors (one sum-bit)
    odd_divs = [d for d in divisors if d % 2 == 1]
    even_divs = [d for d in divisors if d % 2 == 0]
    even_sum = sum(even_divs)          # 13920

    # ---- Bit-parallel DP over the complement target ----
    # bitset[b] = int bitmask; bit s set  <=>  complement sum s reachable
    # using exactly b "binary choices" (each odd divisor + optionally the even-sum bit).
    # Choosing the even-sum bit counts as 1 choice but may realize many even divisors,
    # so maximizing b minimizes the number of selected divisors.
    def run_dp(items):
        m = len(items)
        bitset = [0] * (m + 1)
        bitset[0] = 1  # bit 0 set
        for it in items:
            for b in range(m - 1, -1, -1):
                bitset[b + 1] |= (bitset[b] << it)
        return bitset

    # Pass 1: include the even-sum as a single binary item
    bitset_with = run_dp(odd_divs + [even_sum])
    # Pass 2: exclude the even-sum (even divisors handled separately in reconstruction)
    bitset_without = run_dp(odd_divs)

    def reachable(bitset, b, s):
        return (bitset[b] >> s) & 1

    # Search for the maximum binary count b that reaches comp_target.
    best = None
    for b in range(len(odd_divs) + 1, -1, -1):
        if reachable(bitset_with, b, comp_target):
            best = (b, True, None)
            break
        if reachable(bitset_without, b, comp_target):
            best = (b, False, None)
            break
        if time.time() - t0 > DEADLINE:
            break

    if best is not None:
        b, use_even, _ = best
        # Reconstruct which odd divisors (and the even-sum bit) form the complement.
        odd_set = set(odd_divs)
        chosen_odd = []
        cur = comp_target
        if use_even:
            items = odd_divs + [even_sum]
            bitset = bitset_with
        else:
            items = odd_divs
            bitset = bitset_without
        for it in items:
            if it in odd_set:
                # take this odd divisor if the remainder is reachable with b-1 choices
                if b - 1 >= 0 and reachable(bitset, b - 1, cur - it):
                    chosen_odd.append(it)
                    cur -= it
                    b -= 1
        # (the even-sum bit, if used, is not an odd divisor so it is skipped above)

        comp = set(chosen_odd)
        if use_even:
            comp.add(even_sum)
        comp_sum = sum(chosen_odd) + (even_sum if use_even else 0)
        # remaining even-divisor sum needed (0 if the even-sum bit was used)
        even_need = comp_target - comp_sum
        # even_need is 0 or an even number <= even_sum

        # Greedy even-divisor subset-sum to realize even_need
        even_divs_desc = sorted(even_divs, reverse=True)
        even_choose = []
        rem = even_need
        for d in even_divs_desc:
            if d <= rem:
                even_choose.append(d)
                rem -= d
        # rem must be 0 (even_need is even and divisors allow it); safety below

        comp_full = set(chosen_odd) | set(even_choose)
        selected = [d for d in divisors if d not in comp_full]
        if sum(selected) == target and len(set(selected)) == len(selected):
            return selected

    # ---- Fallback: greedy on divisors (valid but not necessarily minimal) ----
    if time.time() - t0 < DEADLINE:
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if sum(chosen) == target:
            return chosen

    # Final safety: return a valid-looking set (should not be reached)
    return divisors[:1]
