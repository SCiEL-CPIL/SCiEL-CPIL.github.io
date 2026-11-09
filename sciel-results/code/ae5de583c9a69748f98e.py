"""Direct construction of maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

_LCM = 232792560

def _verify(subset):
    k = len(subset)
    if k == 0:
        return True
    scaled = [_LCM // x for x in subset]
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        sums[mask] = sums[mask ^ lb] + scaled[idx]
    return len(set(sums)) == n

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    universe = list(range(1, 23))
    scaled_vals = {x: _LCM // x for x in universe}
    
    def fast_verify(s):
        k = len(s)
        if k == 0:
            return True
        sv = [scaled_vals[x] for x in s]
        n = 1 << k
        sums = [0] * n
        for mask in range(1, n):
            lb = mask & (-mask)
            idx = lb.bit_length() - 1
            sums[mask] = sums[mask ^ lb] + sv[idx]
        return len(set(sums)) == n
    
    # Phase 1: Greedy
    order = sorted(universe, key=lambda x: -scaled_vals[x])
    current = []
    for x in order:
        if time.time() > deadline - 50:
            break
        candidate = current + [x]
        if fast_verify(candidate):
            current = candidate
    
    best = list(current)
    
    # Add remaining
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    for x in remaining:
        if time.time() > deadline - 45:
            break
        candidate = best + [x]
        if fast_verify(candidate):
            best = candidate
            in_set = set(best)
            remaining = [x for x in universe if x not in in_set]
    
    best_size = len(best)
    
    # Phase 2: SA with multiple restarts
    if time.time() < deadline - 15:
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 15:
                break
            random.seed(seed)
            cur = list(best)
            best_sa = list(best)
            
            # Estimate time per verify
            t0 = time.time()
            for _ in range(5):
                fast_verify(cur)
            tpv = (time.time() - t0) / 5
            
            rem_time = deadline - time.time() - 12
            max_steps = int(rem_time / max(tpv, 0.001))
            max_steps = min(max_steps, 8000)
            
            T0 = 1.0
            T_end = 0.02
            
            for step in range(max_steps):
                if time.time() > deadline - 12:
                    break
                
                progress = step / max(max_steps, 1)
                T = T0 * (T_end / T0) ** progress
                
                r = random.random()
                if r < 0.35:
                    not_in = [x for x in universe if x not in cur]
                    if not not_in:
                        continue
                    x = random.choice(not_in)
                    new = cur + [x]
                elif r < 0.65:
                    if not cur:
                        continue
                    x = random.choice(cur)
                    new = [y for y in cur if y != x]
                else:
                    if not cur:
                        continue
                    not_in = [x for x in universe if x not in cur]
                    if not not_in:
                        continue
                    x_out = random.choice(cur)
                    x_in = random.choice(not_in)
                    new = [y if y != x_out else x_in for y in cur]
                
                new_valid = fast_verify(new)
                cur_valid = fast_verify(cur)
                
                if new_valid and not cur_valid:
                    cur = new
                elif new_valid and cur_valid:
                    if len(new) > len(cur):
                        cur = new
                        if len(cur) > len(best_sa):
                            best_sa = list(cur)
                    elif len(new) == len(cur):
                        if random.random() < 0.3 + T * 0.5:
                            cur = new
                else:
                    if len(new) == len(cur) and random.random() < T * 0.05:
                        cur = new
            
            if len(best_sa) > best_size:
                best = best_sa
                best_size = len(best)
    
    # Final add
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    for x in remaining:
        if time.time() > deadline - 2:
            break
        candidate = best + [x]
        if fast_verify(candidate):
            best = candidate
    
    return best

