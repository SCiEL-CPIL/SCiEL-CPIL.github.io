import time
import random

def construct():
    deadline = time.time() + 85
    n = 256

    # Precompute conflicts as bitmasks for O(1) conflict checks
    # conflicts[i] is an integer bitmask where bit j is set iff (i*j)%(i+j)==0
    conflicts = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)

    def mask_size(m):
        return bin(m).count('1')

    def greedy_fill(s_mask, order):
        """Add elements from order to s_mask if valid (no conflict)."""
        for e in order:
            bit = 1 << e
            if s_mask & bit:
                continue
            if not (conflicts[e] & s_mask):
                s_mask |= bit
        return s_mask

    def local_search(s_mask, max_iters=200):
        """Local search: add, 1-swap, and 2-swap moves."""
        best_mask = s_mask
        best_size = mask_size(s_mask)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # 1. Try to add any non-member (no conflict)
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s_mask & bit_b:
                    continue
                if not (conflicts[b] & s_mask):
                    s_mask |= bit_b
                    sz = mask_size(s_mask)
                    if sz > best_size:
                        best_size = sz
                        best_mask = s_mask
                    improved = True
                    break
            if improved:
                continue

            # 2. Try 1-swap: remove a, add b (b conflicts with exactly one element in S)
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s_mask & bit_b:
                    continue
                cwb = conflicts[b] & s_mask
                if cwb == 0:
                    continue
                if cwb & (cwb - 1) == 0:  # exactly one bit set
                    a_bit = cwb
                    new_mask = (s_mask & ~a_bit) | bit_b
                    # Verify b doesn't conflict with new_mask minus b
                    if not (conflicts[b] & (new_mask & ~bit_b)):
                        sz = mask_size(new_mask)
                        if sz > best_size:
                            s_mask = new_mask
                            best_size = sz
                            best_mask = new_mask
                            improved = True
                            break
                        elif sz == best_size:
                            # Accept sideways move occasionally for diversity
                            if random.random() < 0.1:
                                s_mask = new_mask
                                improved = True
                                break
            if improved:
                continue

            # 3. Try 2-swap: remove a, remove c, add b (b conflicts with exactly {a,c})
            for b in range(1, n + 1):
                bit_b = 1 << b
                if s_mask & bit_b:
                    continue
                cwb = conflicts[b] & s_mask
                if cwb == 0:
                    continue
                # Check if exactly 2 bits set
                if cwb & (cwb - 1) == 0:
                    continue
                if (cwb & (cwb - 1)) & ((cwb & (cwb - 1)) - 1) == 0:
                    # Exactly two bits: extract them
                    a_bit = cwb & (-cwb)
                    c_bit = cwb ^ a_bit
                    new_mask = (s_mask & ~a_bit & ~c_bit) | bit_b
                    if not (conflicts[b] & (new_mask & ~bit_b)):
                        sz = mask_size(new_mask)
                        if sz > best_size:
                            s_mask = new_mask
                            best_size = sz
                            best_mask = new_mask
                            improved = True
                            break
            if improved:
                continue

            # 4. Random kick: remove 1-3 random elements, then greedily refill
            if random.random() < 0.3:
                elems = [i for i in range(1, n + 1) if s_mask & (1 << i)]
                if elems:
                    k = random.randint(1, 3)
                    to_remove = random.sample(elems, min(k, len(elems)))
                    for r in to_remove:
                        s_mask &= ~(1 << r)
                    # Greedily refill from all elements in random order
                    remaining = [e for e in range(1, n + 1) if not (s_mask & (1 << e))]
                    random.shuffle(remaining)
                    s_mask = greedy_fill(s_mask, remaining)
                    sz = mask_size(s_mask)
                    if sz > best_size:
                        best_size = sz
                        best_mask = s_mask
                    improved = True

            if not improved:
                break

        return best_mask, best_size

    all_elements = list(range(1, n + 1))
    odds = set(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))

    best_mask = 0
    best_size = 0

    # Phase 1: Structured greedy starts
    # Start with all odds, add evens sorted by conflict count with odds
    odds_mask = 0
    for o in odds:
        odds_mask |= (1 << o)

    def conflict_count_with_mask(e, m):
        return mask_size(conflicts[e] & m)

    evens_sorted = sorted(evens, key=lambda e: conflict_count_with_mask(e, odds_mask))
    s_mask = greedy_fill(odds_mask, evens_sorted)
    s_mask, sz = local_search(s_mask, max_iters=300)
    if sz > best_size:
        best_size = sz
        best_mask = s_mask

    # Phase 2: Randomized greedy restarts from odds
    random.seed(42)
    for trial in range(80):
        if time.time() > deadline - 5:
            break
        evens_shuffled = list(evens)
        random.shuffle(evens_shuffled)
        s_mask = greedy_fill(odds_mask, evens_shuffled)
        s_mask, sz = local_search(s_mask, max_iters=100)
        if sz > best_size:
            best_size = sz
            best_mask = s_mask

    # Phase 3: Randomized kicks from best
    for trial in range(60):
        if time.time() > deadline - 5:
            break
        s_mask = best_mask
        # Kick: remove K random elements
        elems = [i for i in range(1, n + 1) if s_mask & (1 << i)]
        K = random.randint(3, 12)
        to_remove = random.sample(elems, min(K, len(elems)))
        for r in to_remove:
            s_mask &= ~(1 << r)
        # Greedily refill
        remaining = [e for e in all_elements if not (s_mask & (1 << e))]
        random.shuffle(remaining)
        s_mask = greedy_fill(s_mask, remaining)
        s_mask, sz = local_search(s_mask, max_iters=150)
        if sz > best_size:
            best_size = sz
            best_mask = s_mask

    # Phase 4: Final aggressive local search on best
    if time.time() < deadline - 5:
        s_mask, sz = local_search(best_mask, max_iters=500)
        if sz > best_size:
            best_size = sz
            best_mask = s_mask

    # Convert best_mask to list
    result = [i for i in range(1, n + 1) if best_mask & (1 << i)]
    if not result:
        result = [1]
    return result
