import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n

    # Precompute: for each pair (a,b), is a a strict subset of b?
    # Store as a list of bitmasks: subset_mask[a] has bit b set if a is strict subset of b
    subset_mask = [0] * N
    for a in range(N):
        # a is strict subset of b if (a & b) == a and a != b
        # i.e., b must contain all bits of a, and have at least one more
        # b = a | extra where extra != 0 and (extra & a) == 0... wait no
        # b must satisfy: (a & b) == a, which means b has all bits of a
        # and b != a
        # So b = a | x where x is a subset of ~a and x != 0
        # But we can just iterate all b
        for b in range(N):
            if b != a and (a & b) == a:
                subset_mask[a] |= (1 << b)
    
    def is_strict_subset(a, b):
        return (subset_mask[a] >> b) & 1

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_strict_subset(family[i], family[j]) or is_strict_subset(family[j], family[i]):
                    count += 1
        return count

    def local_search(family, max_iters, deadline):
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        temp = 2.0
        cooling = 0.9999
        min_temp = 0.01
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            i = random.randint(0, m-1)
            
            r = random.random()
            if r < 0.4:
                cur = family[i]
                if cur == 0:
                    new_mask = random.randint(1, N-1)
                else:
                    bits = [b for b in range(n) if (cur >> b) & 1]
                    if bits:
                        b = random.choice(bits)
                        new_mask = cur & ~(1 << b)
                        if random.random() < 0.3:
                            remaining_bits = [bb for bb in range(n) if (cur >> bb) & 1 and bb != b]
                            if remaining_bits:
                                bb = random.choice(remaining_bits)
                                new_mask &= ~(1 << bb)
                    else:
                        new_mask = random.randint(0, N-1)
            elif r < 0.8:
                cur = family[i]
                zero_bits = [b for b in range(n) if not ((cur >> b) & 1)]
                if zero_bits:
                    b = random.choice(zero_bits)
                    new_mask = cur | (1 << b)
                    if random.random() < 0.3:
                        remaining_zeros = [bb for bb in range(n) if not ((cur >> bb) & 1) and bb != b]
                        if remaining_zeros:
                            bb = random.choice(remaining_zeros)
                            new_mask |= (1 << bb)
                else:
                    new_mask = random.randint(0, N-1)
            else:
                new_mask = random.randint(0, N-1)
            
            if new_mask == family[i]:
                continue
            if new_mask in family[:i] + family[i+1:]:
                continue
            
            old_mask = family[i]
            
            # Compute delta using precomputed table
            lost = 0
            gained = 0
            for j in range(m):
                if j == i:
                    continue
                other = family[j]
                if is_strict_subset(old_mask, other) or is_strict_subset(other, old_mask):
                    lost += 1
                if is_strict_subset(new_mask, other) or is_strict_subset(other, new_mask):
                    gained += 1
            
            delta = gained - lost
            
            if delta > 0:
                family[i] = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            else:
                if temp > min_temp:
                    if random.random() < math.exp(delta / temp):
                        family[i] = new_mask
                        current_score += delta
            temp *= cooling
        
        return best_family, best_score

    best_overall = None
    best_score_overall = 0
    
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        
        if restart == 0:
            chain = [0]
            cur = 0
            for b in range(n):
                cur |= (1 << b)
                chain.append(cur)
            used = set(chain)
            remaining = [x for x in range(N) if x not in used]
            extra = random.sample(remaining, m - len(chain))
            family = chain + extra
        elif restart == 1:
            chain = [(1<<n)-1]
            cur = (1<<n)-1
            for b in range(n):
                cur &= ~(1 << b)
                chain.append(cur)
            used = set(chain)
            remaining = [x for x in range(N) if x not in used]
            extra = random.sample(remaining, m - len(chain))
            family = chain + extra
        else:
            # Random with size bias
            family = []
            used = set()
            for _ in range(m):
                for attempt in range(200):
                    size = random.randint(0, n)
                    if size == 0:
                        mask = 0
                    else:
                        bits = random.sample(range(n), size)
                        mask = 0
                        for b in bits:
                            mask |= (1 << b)
                    if mask not in used:
                        family.append(mask)
                        used.add(mask)
                        break
                else:
                    while True:
                        mask = random.randint(0, N-1)
                        if mask not in used:
                            family.append(mask)
                            used.add(mask)
                            break
        
        fam, score = local_search(family, 120000, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
