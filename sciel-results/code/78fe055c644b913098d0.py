import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill

    MAX_SUM = 7680  # 30 * 256

    # Precompute conflict bits
    conflict_bits = [0] * (MAX_SUM + 1)
    for u in range(1, MAX_SUM + 1):
        bits = 0
        for d in range(1, int(u**0.5) + 1):
            if u % d == 0:
                if d != u:
                    bits |= (1 << d)
                other = u // d
                if other != d and other != u:
                    bits |= (1 << other)
        for m in range(u * 2, MAX_SUM + 1, u):
            bits |= (1 << m)
        conflict_bits[u] = bits

    def try_add(x, sums_bits):
        if sums_bits & (sums_bits << x):
            return (False, 0)
        new_sums_bits = sums_bits << x
        all_bits = sums_bits | new_sums_bits
        temp = new_sums_bits
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            if conflict_bits[u] & all_bits:
                return (False, 0)
            temp ^= lsb
        return (True, all_bits)

    best_S = [31, 32, 34]
    best_size = len(best_S)

    def backtrack(current_S, current_bits, candidates):
        nonlocal best_S, best_size
        
        if time.time() - start_time > time_limit - 5:
            return
        
        if len(current_S) > best_size:
            best_S = current_S[:]
            best_size = len(current_S)
            
        if not candidates:
            return
        
        # Pruning: If we can't beat best_size, stop.
        if len(current_S) + len(candidates) <= best_size:
            return

        for i in range(len(candidates)):
            if time.time() - start_time > time_limit - 5:
                return
            
            x = candidates[i]
            # Skip 1 as it is always invalid in sets of size > 1
            if x == 1:
                continue
                
            ok, new_bits = try_add(x, current_bits)
            if ok:
                backtrack(current_S + [x], new_bits, candidates[i+1:])

    # Run 1: Descending order
    candidates_desc = list(range(256, 0, -1))
    backtrack([], 1, candidates_desc)

    # Run 2: Randomized restarts
    # We limit the number of restarts to ensure we finish in time.
    # Each backtracking run might be fast due to pruning.
    for _ in range(100):
        if time.time() - start_time > time_limit - 10:
            break
        candidates_rand = list(range(256, 0, -1))
        random.shuffle(candidates_rand)
        backtrack([], 1, candidates_rand)

    # Final verification
    if len(best_S) > 0:
        bits = 1
        valid = True
        for x in best_S:
            if bits & (bits << x):
                valid = False
                break
            new_bits = bits << x
            all_bits = bits | new_bits
            temp = new_bits
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                if conflict_bits[u] & all_bits:
                    valid = False
                    break
                temp ^= lsb
            if not valid:
                break
            bits = all_bits
        if not valid:
            best_S = [31, 32, 34]

    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(res)
