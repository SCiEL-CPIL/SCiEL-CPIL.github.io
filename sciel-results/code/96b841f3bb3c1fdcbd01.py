import time
import random

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 78.0  # safety margin under 90s kill

    # Precompute all 3-APs as (a,b,c) with a<b<c, d>0
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))
    m3 = len(aps_3)

    # For each element x, list of 3-AP indices containing x
    x_aps = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(aps_3):
        x_aps[a].append(i)
        x_aps[b].append(i)
        x_aps[c].append(i)

    # Bitset for 4-AP checks: mask[x][d] has bits at positions of the 4 terms
    # We'll use Python ints as bitsets. Bit i represents element i (1-indexed).
    # Precompute for each (x,d) the 4 masks: the 4-APs where x is term 1,2,3,4
    # Actually simpler: for each x and d, the 4-AP containing x at position p
    # We'll store masks[x] = list of (d, [mask1,mask2,mask3,mask4]) for the 4-APs
    # where x is the 1st,2nd,3rd,4th term respectively.

    # For a candidate x, to check if adding x creates a 4-AP:
    # For each d, check the 4-APs where x appears:
    #   x as term1: need S & mask == mask (all 3 others in S)
    #   x as term2: need S & mask == mask
    #   etc.
    # Where mask = bits of the other 3 terms.

    # Precompute: for each x, for each d, the bitmask of the 3 other terms in each 4-AP containing x
    x_4ap_masks = [[] for _ in range(n + 1)]
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            terms = [a, a + d, a + 2 * d, a + 3 * d]
            for pos in range(4):
                x = terms[pos]
                other_mask = 0
                for j in range(4):
                    if j != pos:
                        other_mask |= (1 << terms[j])
                x_4ap_masks[x].append(other_mask)

    def has_4ap(S_bits):
        # S_bits is a Python int bitset
        # Check if any 4-AP is fully contained
        # For each x in S, check its 4-AP masks
        # But that's expensive. Instead, for a single candidate addition, we check incrementally.
        pass

    def is_valid_add(x, S_bits):
        """Check if adding x to S_bits creates no 4-AP."""
        for mask in x_4ap_masks[x]:
            # If all 3 other terms are in S_bits, then adding x completes a 4-AP
            if (S_bits & mask) == mask:
                return False
        return True

    def gain_of_add(x, S_bits):
        """Count new 3-APs formed by adding x to S_bits."""
        g = 0
        for i in x_aps[x]:
            a, b, c = aps_3[i]
            # Check if the other two terms are in S_bits
            other1 = a if a != x else (b if b != x else c)
            other2 = b if b != x else (c if c != x else a)
            # Simpler: just check all three, x is not in S_bits yet
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                # All three in S_bits? But x not in S_bits, so this is 0
                pass
            # x is one of a,b,c. The other two must be in S_bits.
            # Since x not in S_bits, check:
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    g += 1
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1):
                    g += 1
            else:  # c == x
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1):
                    g += 1
        return g

    def score_set(S_bits):
        """Count 3-APs in S_bits."""
        cnt = 0
        for a, b, c in aps_3:
            if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                cnt += 1
        return cnt

    def greedy_scored(order):
        S_bits = 0
        for x in order:
            if not is_valid_add(x, S_bits):
                continue
            # Compute gain
            g = gain_of_add(x, S_bits)
            # Add if valid (even with 0 gain, to enable future gains)
            S_bits |= (1 << x)
        return S_bits

    def local_search(S_bits):
        """Add valid elements and do swaps to improve score."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False

            # Phase 1: Add any valid element
            for x in range(1, n + 1):
                if (S_bits >> x) & 1:
                    continue
                if is_valid_add(x, S_bits):
                    S_bits |= (1 << x)
                    improved = True

            # Phase 2: Swap - remove y, add x
            # Try to find a swap that improves score
            # For each x not in S, compute gain if we could add it (ignoring 4-AP)
            # For each y in S, compute loss if we remove it
            # Then try swaps where gain > loss and the swap is valid

            # First, compute for each x not in S: the gain of adding x (to current S)
            # and whether it's valid
            add_info = {}
            for x in range(1, n + 1):
                if (S_bits >> x) & 1:
                    continue
                if is_valid_add(x, S_bits):
                    add_info[x] = gain_of_add(x, S_bits)

            # Compute for each y in S: the loss of removing y
            # Loss = number of 3-APs that y is part of (in current S)
            # But removing y might also make some additions possible.
            # Simple loss: count 3-APs containing y in S
            S_list = [i for i in range(1, n + 1) if (S_bits >> i) & 1]
            loss_info = {}
            for y in S_list:
                loss = 0
                for i in x_aps[y]:
                    a, b, c = aps_3[i]
                    if (S_bits >> a) & 1 and (S_bits >> b) & 1 and (S_bits >> c) & 1:
                        loss += 1
                loss_info[y] = loss

            # Try swaps: for each x in add_info, for each y in S_list
            # where loss_info[y] < add_info[x]
            # Check if after removing y, x is still valid (it should be, since removing y
            # only reduces S, so 4-AP risk decreases)
            # Actually removing y can only help validity, so if x was valid before,
            # it's valid after removing y.

            # But we also need to check: does removing y and adding x actually improve?
            # Net change = add_info[x] - loss_info[y] + (new 3-APs from x that use y? No, y is removed)
            # Wait: add_info[x] counts 3-APs using x and two elements in S.
            # After removing y, some of those might not exist anymore.
            # So the actual gain of adding x to (S \ {y}) might be less.
            # This is getting complicated. Let's just try the swap and recompute.

            # To limit time, only try a subset of swaps
            if time.time() < deadline - 10:
                # Sort add_info by gain descending
                sorted_adds = sorted(add_info.items(), key=lambda kv: -kv[1])
                # Sort S_list by loss ascending
                sorted_rems = sorted(S_list, key=lambda y: loss_info[y])

                # Try top candidates
                best_swap_gain = 0
                best_swap = None
                for x, gx in sorted_adds[:20]:  # top 20 additions
                    if gx <= 0:
                        break
                    for y in sorted_rems[:20]:  # bottom 20 losses
                        ly = loss_info[y]
                        if gx <= ly:
                            break
                        # Try the swap
                        S_new = (S_bits & ~(1 << y)) | (1 << x)
                        # Check validity: x must be valid in S\{y}
                        # Since S\{y} is a subset of S, and x was valid in S, it's valid in S\{y}
                        # But wait, we need to check 4-APs in S_new.
                        # The only new 4-APs would involve x. Since x was valid in S,
                        # and S\{y} ⊆ S, x is valid in S\{y}.
                        # So S_new is valid.
                        s_new = score_set(S_new)
                        s_old = score_set(S_bits)
                        if s_new > s_old:
                            delta = s_new - s_old
                            if delta > best_swap_gain:
                                best_swap_gain = delta
                                best_swap = (x, y)

                if best_swap:
                    x, y = best_swap
                    S_bits = (S_bits & ~(1 << y)) | (1 << x)
                    improved = True

        return S_bits

    # Main: multiple random restarts
    best_S_bits = 0
    best_score = 0

    # Natural order greedy
    S1 = greedy_scored(range(1, n + 1))
    s1 = score_set(S1)
    if s1 > best_score:
        best_score = s1
        best_S_bits = S1

    # Random restarts
    rng = random.Random(42)
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S_cand = greedy_scored(order)
        S_cand = local_search(S_cand)
        s_cand = score_set(S_cand)
        if s_cand > best_score:
            best_score = s_cand
            best_S_bits = S_cand

    # Final local search on best
    if time.time() < deadline - 5:
        best_S_bits = local_search(best_S_bits)
        best_score = score_set(best_S_bits)

    result = [i for i in range(1, n + 1) if (best_S_bits >> i) & 1]
    return sorted(result)
