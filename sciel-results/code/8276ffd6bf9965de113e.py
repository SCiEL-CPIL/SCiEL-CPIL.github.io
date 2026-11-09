import time
import random

def _can_add_bitmask(mask, x):
    """Check if x can be added to the set represented by bitmask without creating a forbidden triple."""
    # Part 1: Check triples (a, b, x) where a,b in S, a+x in S, b+x in S, a+b in S or a+b==x
    max_val = 128 - x
    if max_val >= 1:
        # valid_a: a in S, a+x in S, a <= max_val
        valid_a = (mask >> x) & mask & ((1 << (max_val + 1)) - 2)  # bits 1..max_val
        if valid_a:
            # For any pair a,b in valid_a with a<b: check if a+b in S or a+b==x
            # (mask >> a) has bit b set iff mask has bit (a+b) set
            # We need: exists a,b in valid_a, a<b, such that (mask>>a) has bit b set OR b==x-a
            # Equivalently: (valid_a & (mask >> a)) has a bit set above a, for some a in valid_a
            va = valid_a
            while va:
                a_bit = va & -va
                a = a_bit.bit_length() - 1
                va ^= a_bit
                # b must be in valid_a, b > a
                b_mask = valid_a & ~((1 << (a + 1)) - 1)  # bits above a
                if b_mask == 0:
                    continue
                # Check if any a+b is in S: (mask >> a) has bit b set iff mask has bit (a+b) set
                if ((mask >> a) & b_mask) != 0:
                    return False
                # Check if a+b == x: b = x-a
                b_val = x - a
                if b_val > a and b_val <= 128:
                    if (b_mask >> b_val) & 1:
                        return False

    # Part 2: Check if x completes a sum: a+b==x with a,b in S, and exists c in S\{a,b} with a+c in S and b+c in S
    for a in range(1, x // 2 + 1):
        b = x - a
        if b <= a or b > 128:
            continue
        if not ((mask >> a) & 1) or not ((mask >> b) & 1):
            continue
        # c must be in S, c != a, c != b, a+c in S, b+c in S
        max_c = 128 - b
        if max_c < 1:
            continue
        c_mask = mask & ((1 << (max_c + 1)) - 2)  # c in S, 1<=c<=max_c
        c_mask &= ~((1 << a) | (1 << b))  # exclude a and b
        if c_mask == 0:
            continue
        # Check if exists c in c_mask with a+c in S AND b+c in S
        if ((mask >> a) & (mask >> b) & c_mask) != 0:
            return False

    return True

def _is_valid_bitmask(mask):
    """Check if the set represented by bitmask is valid (no forbidden triple)."""
    elems = []
    for i in range(1, 129):
        if (mask >> i) & 1:
            elems.append(i)
    
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i + 1, n):
            b = elems[j]
            ab = a + b
            if ab > 128:
                continue
            if not ((mask >> ab) & 1):
                continue
            for k in range(j + 1, n):
                c = elems[k]
                ac = a + c
                bc = b + c
                if ac > 128 or bc > 128:
                    continue
                if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit

    def time_left():
        return deadline - time.time()

    best_mask = 0
    best_size = 0
    rng = random.Random(42)

    def mask_size(mask):
        return bin(mask).count('1')

    def greedy_fill(mask, order, time_limit=None):
        """Greedily add elements from order to mask if valid."""
        if time_limit is None:
            time_limit = deadline - 3
        cur = mask
        for x in order:
            if time.time() > time_limit:
                break
            if (cur >> x) & 1:
                continue
            if _can_add_bitmask(cur, x):
                cur |= (1 << x)
        return cur

    def local_search(mask, max_iter=100000):
        """Local search: try additions, then 1-swaps, then 2-swaps."""
        cur = mask
        cur_size = mask_size(cur)
        no_improve = 0
        iterations = 0

        while time_left() > 8 and iterations < max_iter:
            iterations += 1
            made_progress = False

            # Try to add an element
            for x in range(1, 129):
                if (cur >> x) & 1:
                    continue
                if _can_add_bitmask(cur, x):
                    cur |= (1 << x)
                    cur_size += 1
                    made_progress = True
                    break

            if made_progress:
                no_improve = 0
                continue

            # Try single swaps
            elems = [i for i in range(1, 129) if (cur >> i) & 1]
            found_swap = False
            for r in elems:
                if time_left() < 8:
                    break
                without = cur & ~(1 << r)
                best_a = -1
                for a in range(1, 129):
                    if (without >> a) & 1:
                        continue
                    if _can_add_bitmask(without, a):
                        test_mask = without | (1 << a)
                        # Check if we can add more after swap
                        can_add_more = False
                        for x2 in range(1, 129):
                            if (test_mask >> x2) & 1:
                                continue
                            if _can_add_bitmask(test_mask, x2):
                                can_add_more = True
                                break
                        if can_add_more:
                            best_a = a
                            break
                        if a > r:
                            if best_a == -1 or a > best_a:
                                best_a = a
                if best_a != -1:
                    cur = without | (1 << best_a)
                    cur_size = mask_size(cur)
                    made_progress = True
                    found_swap = True
                    break

            if made_progress:
                no_improve = 0
                continue

            # Try 2-swaps: remove two, add two
            if len(elems) >= 2:
                found_2swap = False
                for i in range(len(elems)):
                    if time_left() < 8:
                        break
                    r1 = elems[i]
                    for j in range(i + 1, len(elems)):
                        if time_left() < 8:
                            break
                        r2 = elems[j]
                        without2 = cur & ~(1 << r1) & ~(1 << r2)
                        # Try to find two replacements
                        best_pair = None
                        for a in range(1, 129):
                            if (without2 >> a) & 1:
                                continue
                            if not _can_add_bitmask(without2, a):
                                continue
                            mask1 = without2 | (1 << a)
                            for b in range(1, 129):
                                if b == a or (mask1 >> b) & 1:
                                    continue
                                if _can_add_bitmask(mask1, b):
                                    test_mask = mask1 | (1 << b)
                                    # Check if we can add more
                                    can_add_more = False
                                    for x2 in range(1, 129):
                                        if (test_mask >> x2) & 1:
                                            continue
                                        if _can_add_bitmask(test_mask, x2):
                                            can_add_more = True
                                            break
                                    if can_add_more:
                                        best_pair = (a, b)
                                        break
                            if best_pair:
                                break
                        if best_pair:
                            cur = without2 | (1 << best_pair[0]) | (1 << best_pair[1])
                            cur_size = mask_size(cur)
                            made_progress = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break

            if made_progress:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    break

        return cur, cur_size

    # Strategy 1: Upper half base + greedy lower (natural order)
    base_mask = 0
    for i in range(65, 129):
        base_mask |= (1 << i)

    lower_elements = list(range(1, 65))

    # Natural order
    cur_mask = greedy_fill(base_mask, lower_elements, deadline - 5)
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask

    if time_left() > 20:
        cur_mask, cur_size = local_search(cur_mask, max_iter=50000)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask

    # Random orderings
    for trial in range(30):
        if time_left() < 12:
            break
        order = lower_elements[:]
        rng.shuffle(order)
        cur_mask = greedy_fill(base_mask, order, deadline - 5)
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask

        if time_left() > 15:
            cur_mask, cur_size = local_search(cur_mask, max_iter=30000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask

    # Strategy 2: Different base ranges
    for start in range(55, 66):
        if time_left() < 8:
            break
        base_mask = 0
        for i in range(start, 129):
            base_mask |= (1 << i)

        candidates = list(range(1, start))
        rng.shuffle(candidates)

        cur_mask = greedy_fill(base_mask, candidates, deadline - 5)
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask

        if time_left() > 12:
            cur_mask, cur_size = local_search(cur_mask, max_iter=30000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask

    # Strategy 3: Odds/evens
    for pattern in [1, 2]:
        if time_left() < 8:
            break
        cur_mask = 0
        for i in range(pattern, 129, 2):
            cur_mask |= (1 << i)
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask

        if time_left() > 12:
            cur_mask, cur_size = local_search(cur_mask, max_iter=30000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask

    # Strategy 4: Aggressive local search with removals on best
    if time_left() > 12:
        current_mask = best_mask
        current_size = best_size
        improved = True
        while time_left() > 10:
            improved = False
            elems = [i for i in range(1, 129) if (current_mask >> i) & 1]
            for r in elems:
                if time_left() < 10:
                    break
                without = current_mask & ~(1 << r)
                for a in range(1, 129):
                    if (without >> a) & 1:
                        continue
                    if _can_add_bitmask(without, a):
                        new_mask = without | (1 << a)
                        can_grow = False
                        for x2 in range(1, 129):
                            if (new_mask >> x2) & 1:
                                continue
                            if _can_add_bitmask(new_mask, x2):
                                can_grow = True
                                break
                        if can_grow or a > r:
                            current_mask = new_mask
                            current_size = mask_size(current_mask)
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                break

        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask

    # Final polish
    if time_left() > 5:
        for x in range(1, 129):
            if time_left() < 3:
                break
            if (best_mask >> x) & 1:
                continue
            if _can_add_bitmask(best_mask, x):
                best_mask |= (1 << x)
                best_size += 1

    # Final verification
    result = [i for i in range(1, 129) if (best_mask >> i) & 1]

    if not _is_valid_bitmask(best_mask):
        result = list(range(65, 129))

    return [int(x) for x in result]
