"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128
    FULL = (1 << (N + 1)) - 1
    FULL &= ~1  # clear bit 0

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # For each x and each d, mask of y such that gcd(x,y) = d
    gcd_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            d = gcd_table[x][y]
            gcd_mask[x][d] |= (1 << y)

    # Precompute: for each pair (a,b) with gcd(a,b)=d, the mask of x where gcd(x,a)=d AND gcd(x,b)=d
    # If a and b are both in S, then no x in this mask can be in S.
    # Store as: pair_block[(a,b)] = mask of forbidden x (excluding a and b themselves)
    pair_block = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = gcd_table[a][b]
            # x must satisfy gcd(x,a)=d and gcd(x,b)=d
            m = gcd_mask[a][d] & gcd_mask[b][d]
            m &= ~(1 << a)
            m &= ~(1 << b)
            if m:
                pair_block[(a, b)] = m
                pair_block[(b, a)] = m

    def can_add(x, selected_mask):
        """Check if x can be added to selected_mask without creating a bad triple."""
        # A bad triple {x, a, b} exists if gcd(x,a)=gcd(x,b)=gcd(a,b)
        # For each a in selected_mask, let d = gcd(x,a). Then we need no b in selected_mask
        # with gcd(x,b)=d and gcd(a,b)=d.
        # Equivalently: for each a in selected_mask, x must not be in pair_block[(a,b)] for any b in selected_mask.
        # Faster: for each a in selected, check if there's b in selected with gcd(x,a)=gcd(x,b)=gcd(a,b).
        # Use: for each a in selected, d = gcd(x,a), then check if (selected & gcd_mask[x][d] & gcd_mask[a][d]) has any bit.
        # But that includes a itself potentially. Let's be careful.
        # Actually: for each a in selected (a != x), d = gcd(x,a).
        # We need: no b in selected (b != x, b != a) such that gcd(x,b)=d and gcd(a,b)=d.
        # gcd(x,b)=d means b in gcd_mask[x][d].
        # gcd(a,b)=d means b in gcd_mask[a][d].
        # So check: (selected & gcd_mask[x][d] & gcd_mask[a][d] & ~(1<<a) & ~(1<<x)) != 0
        # But since x is not in selected yet, ~(1<<x) is automatic.
        # And b != a is handled by ~(1<<a).
        sel = selected_mask
        while sel:
            lb = sel & (-sel)
            a = lb.bit_length() - 1
            sel ^= lb
            d = gcd_table[x][a]
            # Check if any b in selected has gcd(x,b)=d and gcd(a,b)=d
            m = gcd_mask[x][d] & gcd_mask[a][d] & ~(1 << a)
            if m & selected_mask:
                return False
        return True

    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << i)]

    def greedy(order):
        m = 0
        for x in order:
            if can_add(x, m):
                m |= (1 << x)
        return m

    # Multi-start greedy with diverse orderings
    rng = random.Random(12345)
    best_mask = 0
    best_size = 0

    # Deterministic orderings
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        m = greedy(order)
        s = bin(m).count('1')
        if s > best_size:
            best_size = s
            best_mask = m

    # Random orderings
    for _ in range(50):
        if time.time() > deadline - 40:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        m = greedy(order)
        s = bin(m).count('1')
        if s > best_size:
            best_size = s
            best_mask = m

    # Local search with add, remove, swap operations
    def local_search(start_mask, time_limit):
        best = start_mask
        best_size_local = bin(best).count('1')
        no_improve = 0

        while time.time() < time_limit:
            if no_improve > 50:
                break
            improved = False

            # Try adding a single vertex
            for x in range(1, N + 1):
                if best & (1 << x):
                    continue
                if can_add(x, best):
                    best |= (1 << x)
                    best_size_local += 1
                    improved = True
                    break
            if improved:
                no_improve = 0
                continue

            # Try 1-out 2-in
            found_swap = False
            for i in range(1, N + 1):
                if not (best & (1 << i)):
                    continue
                temp = best & ~(1 << i)
                added = 0
                for x in range(1, N + 1):
                    if temp & (1 << x) or x == i:
                        continue
                    if can_add(x, temp):
                        temp |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    best = temp
                    best_size_local += added - 1
                    found_swap = True
                    improved = True
                    break
            if improved:
                no_improve = 0
                continue

            # Try 2-out 3-in
            found_swap2 = False
            for i in range(1, N + 1):
                if not (best & (1 << i)):
                    continue
                for j in range(i + 1, N + 1):
                    if not (best & (1 << j)):
                        continue
                    temp = best & ~(1 << i) & ~(1 << j)
                    added = 0
                    for x in range(1, N + 1):
                        if temp & (1 << x) or x == i or x == j:
                            continue
                        if can_add(x, temp):
                            temp |= (1 << x)
                            added += 1
                            if added >= 3:
                                break
                    if added >= 3:
                        best = temp
                        best_size_local += added - 2
                        found_swap2 = True
                        break
                if found_swap2:
                    break
            if found_swap2:
                no_improve = 0
                continue

            no_improve += 1

        return best, best_size_local

    if time.time() < deadline - 30:
        best_mask, best_size = local_search(best_mask, deadline - 30)

    # Multi-restart local search from perturbed solutions
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Perturb: remove a few random elements, then re-add greedily
        perturbed = best_mask
        elems = mask_to_list(perturbed)
        if len(elems) > 5:
            to_remove = rng.sample(elems, min(3, len(elems) // 4))
            for e in to_remove:
                perturbed &= ~(1 << e)
        # Re-add greedily
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        for x in remaining:
            if can_add(x, perturbed):
                perturbed |= (1 << x)
        # Local search
        m, s = local_search(perturbed, min(deadline - 10, time.time() + 3.0))
        if s > best_size:
            best_size = s
            best_mask = m

    # Final exhaustive add pass
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if best_mask & (1 << x):
                continue
            if can_add(x, best_mask):
                best_mask |= (1 << x)
                best_size += 1
                changed = True

    result = mask_to_list(best_mask)
    return result if result else [1]
# EVOLVE-BLOCK-END
