def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 82  # safety margin under 90s

    N = 128
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict structure:
    # CONFLICT[x] = list of (a, b) with a<b, a!=x, b!=x such that
    # gcd(x,a) == gcd(x,b) == gcd(a,b)
    CONFLICT = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_table[x][b] == g and gcd_table[a][b] == g:
                    CONFLICT[x].append((a, b))

    # Precompute bitmask representations
    # For each x, CONFLICT_MASKS[x] = list of (1<<a, 1<<b) for each conflict pair
    CONFLICT_MASKS = {x: [(1 << a, 1 << b) for (a, b) in CONFLICT[x]] for x in range(1, N + 1)}

    def can_add_bitmask(x, S):
        """Check if x can be added to bitmask S without creating a bad triple."""
        for ma, mb in CONFLICT_MASKS[x]:
            if (S & ma) and (S & mb):
                return False
        return True

    def is_valid_bitmask(S):
        """Check if bitmask S represents a valid (triple-free) set."""
        # Check all elements in S
        for x in range(1, N + 1):
            if S & (1 << x):
                # Remove x from S and check conflicts
                S_without_x = S & ~(1 << x)
                for ma, mb in CONFLICT_MASKS[x]:
                    if (S_without_x & ma) and (S_without_x & mb):
                        return False
        return True

    def count_elements(S):
        """Count bits set in bitmask S."""
        return bin(S).count('1')

    def bitmask_to_list(S):
        """Convert bitmask to list of elements."""
        return [x for x in range(1, N + 1) if S & (1 << x)]

    def list_to_bitmask(lst):
        """Convert list to bitmask."""
        S = 0
        for x in lst:
            S |= (1 << x)
        return S

    # Greedy initialization with multiple orderings
    random.seed(42)

    best_S = 0
    best_size = 0

    # Try multiple random orderings for greedy
    for trial in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)

        S = 0
        for x in order:
            if can_add_bitmask(x, S):
                S |= (1 << x)

        size = count_elements(S)
        if size > best_size:
            best_S = S
            best_size = size

    # Local search: 1-remove-2-add exhaustive
    current_S = best_S
    current_size = best_size

    # Precompute all pairs of elements not in current set for 2-add moves
    # We'll do this dynamically

    T = 5.0
    T_min = 0.01
    cooling = 0.999

    for iteration in range(300000):
        if time.time() > deadline:
            break

        move_type = random.random()

        if move_type < 0.25:
            # Try to add one element
            candidates = [x for x in range(1, N + 1) if not (current_S & (1 << x))]
            if not candidates:
                continue
            x = random.choice(candidates)
            if can_add_bitmask(x, current_S):
                new_S = current_S | (1 << x)
                new_size = current_size + 1
                if new_size > best_size:
                    best_S = new_S
                    best_size = new_size
                current_S = new_S
                current_size = new_size

        elif move_type < 0.45:
            # Try to remove one element
            in_set = [x for x in range(1, N + 1) if current_S & (1 << x)]
            if not in_set:
                continue
            x = random.choice(in_set)
            new_S = current_S & ~(1 << x)
            current_S = new_S
            current_size -= 1

        elif move_type < 0.75:
            # 1-remove-2-add: remove one, try all pairs of additions
            in_set = [x for x in range(1, N + 1) if current_S & (1 << x)]
            out_set = [x for x in range(1, N + 1) if not (current_S & (1 << x))]
            if not in_set or len(out_set) < 2:
                continue

            r = random.choice(in_set)
            S_without_r = current_S & ~(1 << r)

            # Try all pairs from out_set
            best_pair = None
            best_pair_valid = False
            for i in range(len(out_set)):
                x1 = out_set[i]
                if not can_add_bitmask(x1, S_without_r):
                    continue
                S_with_x1 = S_without_r | (1 << x1)
                for j in range(i + 1, len(out_set)):
                    x2 = out_set[j]
                    if can_add_bitmask(x2, S_with_x1):
                        # Valid pair
                        if best_pair is None:
                            best_pair = (x1, x2)
                            best_pair_valid = True
                        # Keep looking for more? No, just take first valid
                        break
                if best_pair_valid:
                    break

            if best_pair_valid:
                x1, x2 = best_pair
                new_S = S_without_r | (1 << x1) | (1 << x2)
                new_size = current_size + 1
                if new_size > best_size:
                    best_S = new_S
                    best_size = new_size
                current_S = new_S
                current_size = new_size
            else:
                # No valid 2-add after removing r; just remove r
                current_S = S_without_r
                current_size -= 1

        else:
            # 1-for-1 swap
            in_set = [x for x in range(1, N + 1) if current_S & (1 << x)]
            out_set = [x for x in range(1, N + 1) if not (current_S & (1 << x))]
            if not in_set or not out_set:
                continue
            x_out = random.choice(in_set)
            x_in = random.choice(out_set)
            S_without_out = current_S & ~(1 << x_out)
            if can_add_bitmask(x_in, S_without_out):
                new_S = S_without_out | (1 << x_in)
                current_S = new_S
                # size unchanged
                if current_size > best_size:
                    best_S = current_S
                    best_size = current_size

        T *= cooling
        if T < T_min:
            T = T_min

    # Final exhaustive 1-remove-2-add pass on best_S
    if time.time() < deadline - 5:
        S = best_S
        size = best_size
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            in_set = [x for x in range(1, N + 1) if S & (1 << x)]
            out_set = [x for x in range(1, N + 1) if not (S & (1 << x))]
            if len(out_set) < 2:
                break
            for r in in_set:
                S_without_r = S & ~(1 << r)
                for i in range(len(out_set)):
                    x1 = out_set[i]
                    if not can_add_bitmask(x1, S_without_r):
                        continue
                    S_with_x1 = S_without_r | (1 << x1)
                    for j in range(i + 1, len(out_set)):
                        x2 = out_set[j]
                        if can_add_bitmask(x2, S_with_x1):
                            new_S = S_with_x1 | (1 << x2)
                            new_size = size + 1
                            if new_size > best_size:
                                best_S = new_S
                                best_size = new_size
                            S = new_S
                            size = new_size
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final validity check
    if is_valid_bitmask(best_S):
        return bitmask_to_list(best_S)
    else:
        # Fallback
        return [1]
