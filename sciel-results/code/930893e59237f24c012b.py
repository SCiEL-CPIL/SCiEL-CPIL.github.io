import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 130
    max_val = 10**18

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    nprimes = len(primes)

    def feasible(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
                if prod > max_val:
                    return False
        return True

    def score_from_exps(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
                if prod > max_val:
                    return -1
        
        divs = [(1, 0)]
        for i, e in enumerate(exps):
            if e == 0: continue
            p = primes[i]
            mask_bit = 1 << i
            cur_len = len(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for j in range(cur_len):
                    divs.append((divs[j][0] * factor, divs[j][1] | mask_bit))
        
        divs.sort(key=lambda x: x[0])
        score = 0
        m = len(divs)
        for i in range(m - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    def get_n(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
        return prod

    def local_search(exps, iters=5000):
        cur_exps = list(exps)
        cur_score = score_from_exps(cur_exps)
        if cur_score < 0:
            return -1, 2, cur_exps
            
        for _ in range(iters):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            improved = False
            # Try increasing/decreasing each exponent
            for i in range(nprimes):
                # Increase
                if feasible(cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]):
                    new = cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]
                    s = score_from_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score = new, s
                        improved = True
                # Decrease
                if cur_exps[i] > 0:
                    new = cur_exps[:i] + [cur_exps[i] - 1] + cur_exps[i + 1:]
                    s = score_from_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score = new, s
                        improved = True
            # Try swap moves: replace prime i with prime j (i < j, exps[i]>0, exps[j]==0)
            for i in range(nprimes - 1):
                if cur_exps[i] == 0:
                    continue
                for j in range(i + 1, nprimes):
                    if cur_exps[j] != 0:
                        continue
                    # Check if swapping i->j is feasible
                    if cur_exps[i] == 1:
                        # Remove prime i, add prime j
                        new = cur_exps[:]
                        new[i] = 0
                        new[j] = 1
                        if feasible(new):
                            s = score_from_exps(new)
                            if s > cur_score:
                                cur_exps, cur_score = new, s
                                improved = True
                    else:
                        # Can't easily swap with exponent > 1, skip for now
                        pass
            if not improved:
                break
        return cur_score, get_n(cur_exps), cur_exps

    best_score = -1
    best_n = 2
    best_exps = [0] * nprimes

    # Phase 1: Generate strong seeds
    seeds = []
    
    # 1. Product of first k primes
    for k in range(1, 16):
        exps = [0] * nprimes
        for i in range(k):
            exps[i] = 1
        if feasible(exps):
            seeds.append(exps)
            
    # 2. Variations of N_15 with extra powers of 2
    base_15 = [1]*15 + [0]
    for extra_2 in range(1, 5):
        exps = base_15[:]
        exps[0] += extra_2
        if feasible(exps):
            seeds.append(exps)
            
    # 3. N_14 with powers of 2 and 3
    base_14 = [1]*14 + [0, 0]
    for e2 in range(1, 10):
        for e3 in range(0, 10):
            exps = base_14[:]
            exps[0] += e2
            exps[1] += e3
            if feasible(exps):
                seeds.append(exps)
                
    # 4. N_13 with powers of 2, 3, 5
    base_13 = [1]*13 + [0, 0, 0]
    for e2 in range(1, 8):
        for e3 in range(0, 8):
            for e5 in range(0, 6):
                exps = base_13[:]
                exps[0] += e2
                exps[1] += e3
                exps[2] += e5
                if feasible(exps):
                    seeds.append(exps)
    
    # 5. Randomized square-free subsets of size 12-15
    rng = random.Random(42)
    for _ in range(100):
        exps = [0] * nprimes
        indices = list(range(15))
        rng.shuffle(indices)
        size = rng.randint(12, 15)
        for i in range(size):
            exps[indices[i]] = 1
        if feasible(exps):
            seeds.append(exps)
    
    # 6. Randomized with small exponents
    for _ in range(100):
        exps = [0] * nprimes
        for i in range(15):
            if rng.random() < 0.8:
                exps[i] = rng.randint(1, 3)
        if feasible(exps):
            seeds.append(exps)

    # Evaluate seeds and run local search on the best ones
    seed_scores = []
    for s in seeds:
        sc = score_from_exps(s)
        if sc > 0:
            seed_scores.append((sc, s))
            
    # Sort by score descending
    seed_scores.sort(reverse=True)
    
    # Run local search on top 30 seeds
    for sc, exps in seed_scores[:30]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        final_score, final_n, final_exps = local_search(exps, iters=3000)
        if final_score > best_score:
            best_score = final_score
            best_n = final_n
            best_exps = final_exps

    # Final check on the best found
    if best_n > 10**18 or best_n < 2:
        return 2
        
    return best_n
