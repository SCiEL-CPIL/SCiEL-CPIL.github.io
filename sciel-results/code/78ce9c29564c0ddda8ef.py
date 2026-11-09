import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def get_paley_67():
        p = 67
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                if ((j - i) % p) in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def count_bad_fast(orient, n):
        in_m = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    in_m[j] |= (1 << i)
                else:
                    in_m[i] |= (1 << j)
                idx += 1
        
        bad = 0
        for a in range(n):
            ia = in_m[a]
            for b in range(a+1, n):
                iab = ia & in_m[b]
                if iab == 0:
                    rem = n - b - 1
                    if rem >= 2:
                        bad += rem * (rem - 1) // 2
                    continue
                for c in range(b+1, n):
                    iabc = iab & in_m[c]
                    if iabc == 0:
                        bad += n - c - 1
                        continue
                    for d in range(c+1, n):
                        if (iabc & in_m[d]) == 0:
                            bad += 1
        return bad

    def idx_to_pair(idx, n):
        """Convert flat edge index to (i, j) pair."""
        i = 0
        count = 0
        while count + (n - 1 - i) > idx:
            i += 1
            count += (n - 1 - i)
        j = i + 1 + (idx - count)
        return i, j

    def pair_to_idx(i, j, n):
        """Convert (i, j) pair to flat edge index."""
        idx = 0
        for a in range(i):
            idx += (n - 1 - a)
        idx += (j - i - 1)
        return idx

    # Try to find a valid tournament with n=66 using enhanced local search
    if time.time() < deadline - 15:
        random.seed(12345)
        n = 66
        n_pairs = n * (n - 1) // 2

        # Strategy: Start from Paley-67, remove one vertex, then local search
        p = 67
        squares = {x*x % p for x in range(1, p)}

        # Build Paley-67 orientation
        paley_orient = []
        for i in range(p):
            for j in range(i+1, p):
                if ((j - i) % p) in squares:
                    paley_orient.append(0)
                else:
                    paley_orient.append(1)

        # Try removing each vertex and see which gives the fewest bad 4-sets
        best_init = None
        best_init_bad = float('inf')
        best_removed = -1

        for remove_v in range(p):
            if time.time() > deadline - 10:
                break
            # Build orientation for n=66 after removing vertex remove_v
            orientation = []
            idx_paley = 0
            for i in range(p):
                if i == remove_v:
                    continue
                for j in range(i+1, p):
                    if j == remove_v:
                        continue
                    orientation.append(paley_orient[idx_paley])
                    idx_paley += 1
            # Wait, this indexing is wrong. Let me fix it.
            # Actually, let's just build it properly.
            orientation = []
            for i in range(p):
                if i == remove_v:
                    continue
                for j in range(i+1, p):
                    if j == remove_v:
                        continue
                    if ((j - i) % p) in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)

            bad = count_bad_fast(orientation, n)
            if bad < best_init_bad:
                best_init_bad = bad
                best_init = orientation[:]
                best_removed = remove_v

        if best_init is None:
            # Fallback to random
            best_init = [random.choice([0, 1]) for _ in range(n_pairs)]
            best_init_bad = count_bad_fast(best_init, n)

        cur_bad = best_init_bad
        cur_orient = best_init[:]

        if cur_bad == 0:
            return {'n': n, 'orientation': cur_orient}

        # Simulated annealing + hill climbing
        # Use time-based budget
        sa_start = time.time()
        sa_budget = 90  # seconds for SA
        cur_temp = 1.0
        cool_rate = 0.9997
        min_temp = 0.01

        step = 0
        while time.time() - sa_start < sa_budget and time.time() < deadline - 5:
            step += 1
            # Pick random edge to flip
            idx = random.randint(0, n_pairs - 1)
            old_val = cur_orient[idx]
            cur_orient[idx] = 1 - old_val
            new_bad = count_bad_fast(cur_orient, n)
            delta = new_bad - cur_bad

            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(cur_temp, 1e-10)):
                cur_bad = new_bad
                if cur_bad == 0:
                    return {'n': n, 'orientation': cur_orient}
            else:
                cur_orient[idx] = old_val

            # Cool down
            cur_temp *= cool_rate
            if cur_temp < min_temp:
                cur_temp = min_temp

        # If SA didn't find it, try multi-start with random restarts
        if time.time() < deadline - 5:
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                # Start from current best or random
                if restart == 0:
                    cur_orient = best_init[:]
                    cur_bad = best_init_bad
                else:
                    cur_orient = [random.choice([0, 1]) for _ in range(n_pairs)]
                    cur_bad = count_bad_fast(cur_orient, n)

                if cur_bad == 0:
                    return {'n': n, 'orientation': cur_orient}

                # Short hill climbing
                hc_start = time.time()
                hc_budget = 10
                while time.time() - hc_start < hc_budget and time.time() < deadline - 5:
                    idx = random.randint(0, n_pairs - 1)
                    old_val = cur_orient[idx]
                    cur_orient[idx] = 1 - old_val
                    new_bad = count_bad_fast(cur_orient, n)
                    if new_bad < cur_bad:
                        cur_bad = new_bad
                        if cur_bad == 0:
                            return {'n': n, 'orientation': cur_orient}
                    else:
                        cur_orient[idx] = old_val

    return get_paley_67()
