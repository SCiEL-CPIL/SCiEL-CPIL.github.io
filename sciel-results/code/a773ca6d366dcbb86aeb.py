# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for the 60s wall-clock limit

    def search(remaining_num, remaining_den, min_den, terms_left, current, deadline):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if remaining_num == 0:
                return list(current)
            return None
        
        if remaining_num == 0:
            return None
        
        if terms_left == 1:
            if remaining_den % remaining_num == 0:
                d = remaining_den // remaining_num
                if d >= min_den and d >= 5 and d <= 10000000:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
            return None
        
        # Pruning: max achievable with terms_left terms starting from min_den
        # Sum of 1/(min_den + i) for i in range(terms_left)
        max_sum_num, max_sum_den = 0, 1
        for i in range(terms_left):
            d = min_den + i
            new_num = max_sum_num * d + max_sum_den
            new_den = max_sum_den * d
            g = gcd(new_num, new_den)
            max_sum_num, max_sum_den = new_num // g, new_den // g
        
        if remaining_num * max_sum_den > max_sum_num * remaining_den:
            return None
        
        # Lower bound on d
        d_lower = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound on d:
        # remaining - 1/d <= max_achievable(terms_left-1, d+1)
        # max_achievable(terms_left-1, d+1) <= (terms_left-1) / (d+1)
        # remaining - 1/d <= (terms_left-1) / (d+1)
        # remaining <= 1/d + (terms_left-1)/(d+1)
        # remaining * d * (d+1) <= (d+1) + d * (terms_left-1)
        # This is a quadratic inequality in d.
        # Simpler bound: remaining <= terms_left / d  => d <= terms_left / remaining
        # d <= floor(terms_left * remaining_den / remaining_num)
        d_upper = (terms_left * remaining_den) // remaining_num
        
        if d_upper < d_lower:
            return None
            
        # Cap d_upper to a reasonable limit to avoid infinite loops if bound is loose
        # But the bound d <= k/remaining is tight.
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            # 1/d <= remaining?
            if d * remaining_num < remaining_den:
                continue
            
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num < 0:
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            if terms_left - 1 == 0:
                if new_num == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            current.append(d)
            result = search(new_num, new_den, d + 1, terms_left - 1, current, deadline)
            if result is not None:
                return result
            current.pop()
        
        return None

    def verify(result):
        if not result:
            return False
        total_num, total_den = 0, 1
        for d in result:
            total_num = total_num * d + total_den
            total_den = total_den * d
            g = gcd(total_num, total_den)
            total_num //= g
            total_den //= g
        if total_num != total_den:
            return False
        if len(set(result)) != len(result):
            return False
        if not all(5 <= d <= 10000000 for d in result):
            return False
        return True

    # Try smaller term counts first (minimize count)
    for n_terms in range(4, 11):
        if time.time() > deadline - 5:
            break
        result = search(1, 1, 5, n_terms, [], deadline)
        if result and verify(result):
            return result

    # Fallback: use a known valid solution
    # 1/5 + 1/5 + ... won't work (need distinct)
    # Let's use: 1/6 + 1/7 + 1/42 + 1/13 + 1/14 + 1/182 = 1? Let me verify with a simpler approach
    # Actually, let's just use the greedy method as fallback
    remaining_num, remaining_den = 1, 1
    result = []
    min_d = 5
    for _ in range(256):
        if remaining_num == 0:
            break
        if time.time() > deadline:
            break
        d = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        while d * remaining_num < remaining_den:
            d += 1
        if d > 10000000:
            break
        result.append(d)
        remaining_num = remaining_num * d - remaining_den
        remaining_den = remaining_den * d
        g = gcd(remaining_num, remaining_den)
        if g > 1:
            remaining_num //= g
            remaining_den //= g
        min_d = d + 1
    
    if remaining_num == 0 and verify(result):
        return result
    
    # Last resort: return a hardcoded valid solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 = 1? No.
    # Let me use a verified solution: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100 = 1? No, this is way more than 1.
    
    # I need to find a valid hardcoded solution. Let me use the result from a quick computation.
    # For now, raise an error if we get here
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
