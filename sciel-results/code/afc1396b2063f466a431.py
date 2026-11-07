import time
import random

def _validate_full(elements):
    if not elements:
        return True
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    if len(set(elements)) != len(elements):
        return False
    sums = {0}
    for x in elements:
        new_sums = set(s + x for s in sums)
        if sums & new_sums:
            return False
        sums |= new_sums
    non_empty_sums = [s for s in sums if s > 0]
    non_empty_sums.sort()
    max_sum = non_empty_sums[-1] if non_empty_sums else 0
    for i in range(len(non_empty_sums)):
        a = non_empty_sums[i]
        if a > max_sum // 2:
            break
        for j in range(i + 1, len(non_empty_sums)):
            if non_empty_sums[j] % a == 0:
                return False
    return True

def _try_add(sums_set, candidate):
    """Returns new sums_set if valid, else None."""
    for s in sums_set:
        if s + candidate in sums_set:
            return None
    new_sums = set(sums_set)
    new_sums.update(s + candidate for s in sums_set)
    non_zero = [s for s in new_sums if s > 0]
    non_zero.sort()
    max_s = non_zero[-1]
    for a in non_zero:
        if a > max_s // 2:
            break
        for m in range(2 * a, max_s + 1, a):
            if m in new_sums:
                return None
    return new_sums

def construct():
    deadline = time.time() + 45.0
    best = []
    
    def greedy(order):
        elems = []
        sums = {0}
        for c in order:
            if time.time() > deadline:
                break
            ns = _try_add(sums, c)
            if ns is not None:
                elems.append(c)
                sums = ns
        return elems
    
    # Focus on large numbers [129..256] which have better divisibility properties
    e = greedy(range(256, 128, -1))
    if len(e) > len(best):
        best = e
    
    e = greedy(range(256, 0, -1))
    if len(e) > len(best):
        best = e
    
    random.seed(777)
    
    for t in range(400):
        if time.time() > deadline - 5:
            break
        strat = t % 7
        if strat == 0:
            pool = list(range(129, 257))
            random.shuffle(pool)
        elif strat == 1:
            pool = list(range(1, 257))
            random.shuffle(pool)
        elif strat == 2:
            pool = list(range(256, 128, -1))
        elif strat == 3:
            pool = random.sample(range(129, 257), random.randint(50, 128))
            pool.sort(reverse=True)
        elif strat == 4:
            large = random.sample(range(192, 257), random.randint(30, 64))
            med = random.sample(range(96, 192), random.randint(30, 64))
            pool = large + med
            pool.sort(reverse=True)
        elif strat == 5:
            pool = [x for x in range(129, 257) if x % 2 == 1]
            random.shuffle(pool)
        else:
            def prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5)+1):
                    if n % i == 0: return False
                return True
            pool = [x for x in range(129, 257) if prime(x)]
            random.shuffle(pool)
        
        e = greedy(pool)
        if len(e) > len(best):
            best = e
    
    # Local search: swap elements to grow the set
    if best and time.time() < deadline - 3:
        cur = list(best)
        for _ in range(80):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(len(cur)):
                if time.time() > deadline - 2:
                    break
                rem = [x for j, x in enumerate(cur) if j != i]
                cands = [x for x in range(1, 257) if x not in rem]
                random.shuffle(cands)
                for cand in cands[:25]:
                    test = rem + [cand]
                    sums = {0}
                    valid = True
                    for x in test:
                        ns = set(s + x for s in sums)
                        if sums & ns:
                            valid = False
                            break
                        sums |= ns
                    if valid:
                        nz = [s for s in sums if s > 0]
                        nz.sort()
                        ms = nz[-1]
                        for a in nz:
                            if a > ms // 2:
                                break
                            for m in range(2*a, ms+1, a):
                                if m in sums:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid and len(test) > len(best):
                        best = test
                        cur = test
                        improved = True
                        break
                if improved:
                    break
    
    if best and not _validate_full(best):
        best = [20, 21, 23]
    
    return list(best)

if __name__ == "__main__":
    print(construct())
