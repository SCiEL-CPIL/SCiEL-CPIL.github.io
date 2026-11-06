"""Bitmask simulated annealing for equal-gcd triple avoidance."""
import time
import random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    N = 128
    UNIVERSE = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute for each element x, for each gcd value g,
    # the bitmask of elements y such that gcd(x,y) == g
    # This lets us quickly check conflicts when adding x.
    # For each x, group other elements by gcd(x, y)
    gcd_groups = [dict() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            g = gcd_mat[x][y]
            if g not in gcd_groups[x]:
                gcd_groups[x][g] = 0
            gcd_groups[x][g] |= (1 << y)

    # For conflict detection: a triple (a,b,c) is bad iff
    # gcd(a,b) == gcd(a,c) == gcd(b,c) == g for some g.
    # When we add element x to set S, a conflict arises iff
    # there exist a, b in S such that gcd(x,a) == gcd(x,b) == gcd(a,b).
    # Precompute: for each x, for each g, the set of elements with gcd(x,.)=g
    # Then for adding x: for each g, let C_g = {y in S : gcd(x,y)=g}
    #   conflict iff there exist a,b in C_g with gcd(a,b)=g.
    # Even better: precompute for each pair (x, g) the bitmask of y where gcd(x,y)=g AND for
    # any two such y1,y2 in S, we need gcd(y1,y2)=g.
    # 
    # Alternative efficient approach: precompute all bad triples as bitmasks.
    # For each element x, precompute the set of (a,b) pairs in S that would form a bad triple with x.
    # 
    # Let's use a different precomputation:
    # For each x, and each g, let M[x][g] = bitmask of y where gcd(x,y) == g
    # When adding x to S: for each g, count elements in S ∩ M[x][g].
    #   If count >= 2, we need to check if any pair among them has gcd == g.
    #   But that's still expensive. 
    #
    # Better: Precompute for each x, the set of "conflict partners" - elements that
    # cannot coexist with x in any valid set. Actually that's not right either.
    #
    # Most efficient: enumerate all bad triples once, store as a set of frozensets.
    # Then for a bitmask S, count violations by checking each bad triple.
    # But there could be many bad triples.
    #
    # Let's count bad triples:
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    # Store as bitmasks for fast intersection
    bad_triple_masks = [((1 << a) | (1 << b) | (1 << c)) for (a, b, c) in bad_triples]
    num_bad_triples = len(bad_triple_masks)

    def count_violations(mask):
        """Count number of bad triples fully contained in mask."""
        count = 0
        for bm in bad_triple_masks:
            if (mask & bm) == bm:
                count += 1
        return count

    def popcount(x):
        return bin(x).count('1')

    # Greedy construction for initial solution
    def greedy_construct(seed_val):
        random.seed(seed_val)
        candidates = list(UNIVERSE)
        random.shuffle(candidates)
        mask = 0
        for x in candidates:
            bit = 1 << x
            new_mask = mask | bit
            if count_violations(new_mask) == 0:
                mask = new_mask
        return mask

    # Simulated annealing
    def anneal(initial_mask, seed_val, max_time=60.0):
        random.seed(seed_val)
        t_start = time.time()
        current_mask = initial_mask
        current_violations = count_violations(current_mask)
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        best_violations = current_violations

        T = 2.0
        T_min = 0.001
        cooling = 0.9997

        step = 0
        while time.time() - t_start < max_time and T > T_min:
            step += 1
            if step % 500 == 0 and time.time() - t_start > max_time:
                break

            # Choose a random move
            move_type = random.random()
            if move_type < 0.4:
                # Add a random element
                x = random.randint(1, N)
                bit = 1 << x
                if current_mask & bit:
                    continue
                new_mask = current_mask | bit
                new_violations = count_violations(new_mask)
                new_size = current_size + 1
            elif move_type < 0.7:
                # Remove a random element
                if current_size <= 1:
                    continue
                x = random.randint(1, N)
                bit = 1 << x
                if not (current_mask & bit):
                    continue
                new_mask = current_mask & ~bit
                new_violations = count_violations(new_mask)
                new_size = current_size - 1
            else:
                # Swap: remove one, add another
                if current_size <= 1:
                    continue
                x_remove = random.randint(1, N)
                bit_r = 1 << x_remove
                if not (current_mask & bit_r):
                    continue
                x_add = random.randint(1, N)
                bit_a = 1 << x_add
                if current_mask & bit_a:
                    continue
                new_mask = (current_mask & ~bit_r) | bit_a
                new_violations = count_violations(new_mask)
                new_size = current_size

            # Acceptance
            delta_v = new_violations - current_violations
            delta_s = new_size - current_size
            # Objective: minimize violations, maximize size
            # Energy = violations * 100 - size (so violations dominate)
            delta_energy = delta_v * 100.0 - delta_s

            if delta_energy <= 0 or random.random() < pow(2.718281828, -delta_energy / T):
                current_mask = new_mask
                current_violations = new_violations
                current_size = new_size

                if current_violations == 0 and current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                    best_violations = 0

            T *= cooling

        return best_mask, best_size

    # Multi-restart SA
    best_mask = 0
    best_size = 0

    # Initial greedy solutions
    for seed_val in [0, 42, 123, 777]:
        if time.time() > DEADLINE - 10:
            break
        g_mask = greedy_construct(seed_val)
        g_size = popcount(g_mask)
        if g_size > best_size:
            best_size = g_size
            best_mask = g_mask

    # SA from best greedy
    if time.time() < DEADLINE - 5:
        sa_mask, sa_size = anneal(best_mask, 0, max_time=min(30, DEADLINE - 5 - (time.time() - t0)))
        if sa_size > best_size:
            best_size = sa_size
            best_mask = sa_mask

    # More SA restarts from different greedy solutions
    for seed_val in range(1, 200):
        if time.time() > DEADLINE - 3:
            break
        remaining_time = DEADLINE - 3 - time.time()
        if remaining_time < 2:
            break
        # Quick greedy
        g_mask = greedy_construct(seed_val)
        g_size = popcount(g_mask)
        # Short SA
        sa_mask, sa_size = anneal(g_mask, seed_val, max_time=min(remaining_time - 1, 15))
        if sa_size > best_size:
            best_size = sa_size
            best_mask = sa_mask

    # Final polish: try to add any remaining elements
    if time.time() < DEADLINE - 1:
        for x in range(1, N + 1):
            bit = 1 << x
            if best_mask & bit:
                continue
            new_mask = best_mask | bit
            if count_violations(new_mask) == 0:
                best_mask = new_mask
                best_size += 1

    # Convert bitmask to list
    result = []
    for x in range(1, N + 1):
        if best_mask & (1 << x):
            result.append(x)

    if not result:
        result = [1]

    return result


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
