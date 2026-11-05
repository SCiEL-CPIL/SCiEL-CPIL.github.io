import time
from math import gcd
import random

def construct():
    N = 128
    start_time = time.time()
    DEADLINE = start_time + 55.0  # Safety margin

    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]

    # Precompute for each pair (x,u): bitmask of v such that {x,u,v} is a bad triple
    # bad_third[x][u] = set of v where gcd(x,u)==gcd(x,v)==gcd(u,v)
    bad_third = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for u in range(x + 1, N + 1):
            mask = 0
            g = gcd_mat[x][u]
            for v in range(1, N + 1):
                if v != x and v != u and gcd_mat[x][v] == g and gcd_mat[u][v] == g:
                    mask |= (1 << v)
            bad_third[x][u] = mask
            bad_third[u][x] = mask

    def count_conflicts(x, S_mask):
        """Count how many pairs (u,v) in S_mask form a bad triple with x."""
        cnt = 0
        temp = S_mask
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            if u != x:
                cnt += bin(bad_third[x][u] & S_mask).count('1')
            temp -= lsb
        return cnt

    def can_add(x, S_mask):
        """Check if adding x to S_mask creates no bad triple."""
        temp = S_mask
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            if bad_third[x][u] & S_mask:
                return False
            temp -= lsb
        return True

    def is_valid(S_mask):
        """Check if S_mask is a valid independent set (no bad triples)."""
        temp = S_mask
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            rest = S_mask & ~(1 << x)
            if count_conflicts(x, rest) > 0:
                return False
            temp -= lsb
        return True

    def greedy_build(seed):
        """Build initial solution using degree-ordered greedy."""
        random.seed(seed)
        S_mask = 0

        # Compute conflict degrees
        deg = [0] * (N + 1)
        for x in range(1, N + 1):
            for u in range(x + 1, N + 1):
                d = bin(bad_third[x][u]).count('1')
                deg[x] += d
                deg[u] += d

        # Sort by degree, then random
        order = list(range(1, N + 1))
        order.sort(key=lambda x: (deg[x], random.random()))

        for x in order:
            if time.time() > DEADLINE:
                break
            if can_add(x, S_mask):
                S_mask |= (1 << x)

        return S_mask

    def simulated_annealing(initial_mask, seed):
        """Run simulated annealing starting from initial_mask."""
        random.seed(seed)
        current = initial_mask
        current_size = bin(current).count('1')
        best = initial_mask
        best_size = current_size

        all_elems = (1 << (N + 1)) - 2  # bits 1..N

        T_start = 2.0
        T_end = 0.01
        cooling = 0.995

        # Build list of elements in and out of current
        def get_lists(mask):
            in_list = []
            out_list = []
            for x in range(1, N + 1):
                if mask & (1 << x):
                    in_list.append(x)
                else:
                    out_list.append(x)
            return in_list, out_list

        T = T_start
        steps_per_temp = 50

        while time.time() < DEADLINE - 3:
            in_list, out_list = get_lists(current)

            for _ in range(steps_per_temp):
                if time.time() > DEADLINE - 3:
                    break

                move_type = random.random()

                if move_type < 0.4:
                    # 1-for-1 swap: remove u from S, add v not in S
                    if not in_list or not out_list:
                        continue
                    u = random.choice(in_list)
                    v = random.choice(out_list)
                    if u == v:
                        continue

                    new_mask = (current & ~(1 << u)) | (1 << v)
                    # Check validity of new_mask
                    # Only need to check triples involving v (since removing u can only help)
                    valid = True
                    temp = new_mask & ~(1 << v)
                    while temp:
                        lsb = temp & -temp
                        x = lsb.bit_length() - 1
                        if bad_third[v][x] & (new_mask & ~(1 << v)):
                            valid = False
                            break
                        temp -= lsb

                    if valid:
                        new_size = current_size  # same size
                        delta = 0
                        # Accept with probability 1 (same size, neutral)
                        current = new_mask
                        if current_size > best_size:
                            best = current
                            best_size = current_size
                        # Try to add more after swap
                        for w in out_list:
                            if w == v:
                                continue
                            if can_add(w, current):
                                current |= (1 << w)
                                current_size += 1
                                if current_size > best_size:
                                    best = current
                                    best_size = current_size
                                break

                elif move_type < 0.7:
                    # 1-for-2 exchange: remove u from S, add v and w not in S
                    if not in_list or len(out_list) < 2:
                        continue
                    u = random.choice(in_list)
                    v, w = random.sample(out_list, 2)
                    if u == v or u == w:
                        continue

                    new_mask = (current & ~(1 << u)) | (1 << v) | (1 << w)
                    # Check validity
                    valid = True
                    for x in [v, w]:
                        temp = new_mask & ~(1 << x)
                        while temp:
                            lsb = temp & -temp
                            y = lsb.bit_length() - 1
                            if y != x and bad_third[x][y] & (new_mask & ~(1 << x)):
                                valid = False
                                break
                            temp -= lsb
                        if not valid:
                            break

                    if valid:
                        new_size = current_size + 1  # removed 1, added 2
                        delta = 1
                        # Always accept improvement
                        current = new_mask
                        current_size = new_size
                        if current_size > best_size:
                            best = current
                            best_size = current_size
                        # Try to add more
                        for z in out_list:
                            if z == v or z == w:
                                continue
                            if can_add(z, current):
                                current |= (1 << z)
                                current_size += 1
                                if current_size > best_size:
                                    best = current
                                    best_size = current_size
                                break

                else:
                    # Try to add an element not in S
                    if not out_list:
                        continue
                    v = random.choice(out_list)
                    if can_add(v, current):
                        current |= (1 << v)
                        current_size += 1
                        if current_size > best_size:
                            best = current
                            best_size = current_size
                    else:
                        # Accept with probability exp(-delta/T) where delta is "cost"
                        # Here cost is the number of conflicts we'd create
                        conflicts = count_conflicts(v, current)
                        if conflicts > 0 and random.random() < (T / (T + conflicts)):
                            # Tentatively add (will be invalid, but helps escape)
                            # Actually, let's not add invalid states.
                            # Instead, try a swap that enables this addition.
                            if in_list:
                                u = random.choice(in_list)
                                new_mask = (current & ~(1 << u)) | (1 << v)
                                # Check if new_mask is valid
                                valid = True
                                temp = new_mask & ~(1 << v)
                                while temp:
                                    lsb = temp & -temp
                                    x = lsb.bit_length() - 1
                                    if bad_third[v][x] & (new_mask & ~(1 << v)):
                                        valid = False
                                        break
                                    temp -= lsb
                                if valid:
                                    current = new_mask
                                    # Size unchanged, but landscape changed

            T *= cooling
            if T < T_end:
                T = T_end

        # Final cleanup: try to add any remaining elements
        in_list, out_list = get_lists(best)
        for v in out_list:
            if can_add(v, best):
                best |= (1 << v)

        return best

    # Run multiple trials
    best_global_mask = 0
    best_global_size = 0

    # First: greedy to get a good starting point
    greedy_mask = greedy_build(42)
    greedy_size = bin(greedy_mask).count('1')
    best_global_mask = greedy_mask
    best_global_size = greedy_size

    # Simulated annealing from greedy and random starts
    seeds = [42, 123, 777, 2024, 5555, 8888, 1111, 3333, 9999, 4242,
             1357, 2468, 6789, 1470, 2583, 3691, 4725, 5819, 6942, 7035]

    for i, seed in enumerate(seeds):
        if time.time() > DEADLINE - 5:
            break

        if i < 5:
            # Start from greedy
            start_mask = greedy_mask
        else:
            # Start from random greedy with different seed
            start_mask = greedy_build(seed)

        sa_result = simulated_annealing(start_mask, seed)
        sa_size = bin(sa_result).count('1')

        if sa_size > best_global_size:
            best_global_size = sa_size
            best_global_mask = sa_result

    # Final improvement pass on best
    in_list, out_list = get_lists(best_global_mask) if False else ([x for x in range(1, N+1) if best_global_mask & (1 << x)], [x for x in range(1, N+1) if not (best_global_mask & (1 << x))])
    for v in out_list:
        if can_add(v, best_global_mask):
            best_global_mask |= (1 << v)

    # Convert to list
    result = [x for x in range(1, N + 1) if best_global_mask & (1 << x)]

    if not result:
        result = [1]

    return sorted(result)
