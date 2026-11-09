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

    # Try to find a valid tournament with n=66 using local search
    if time.time() < deadline - 10:
        random.seed(42)
        n = 66
        n_pairs = n * (n - 1) // 2
        orientation = [random.choice([0, 1]) for _ in range(n_pairs)]
        cur_bad = count_bad_fast(orientation, n)
        
        if cur_bad == 0:
            return {'n': n, 'orientation': orientation}
            
        # Limited local search
        for step in range(20):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, n_pairs - 1)
            i = 0
            count = 0
            while count + (n - 1 - i) > idx:
                i += 1
                count += (n - 1 - i)
            j = i + 1 + (idx - count)
            
            old_val = orientation[idx]
            orientation[idx] = 1 - old_val
            new_bad = count_bad_fast(orientation, n)
            
            if new_bad < cur_bad:
                cur_bad = new_bad
                if cur_bad == 0:
                    return {'n': n, 'orientation': orientation}
            else:
                orientation[idx] = old_val

    return get_paley_67()
