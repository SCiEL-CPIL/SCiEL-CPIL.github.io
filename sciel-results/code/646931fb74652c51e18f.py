"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80  # 80s wall-clock guard (90s limit, 10s margin)
    
    def search(k, min_d, remaining_num, remaining_den):
        """
        Find k distinct denominators >= min_d such that sum(1/di) = remaining_num/remaining_den.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
        
        if k == 0:
            return [] if remaining_num == 0 else None
        
        if remaining_num == 0:
            return None  # Can't add more terms
        
        # Pruning: max possible sum with k terms starting from min_d
        # is sum_{i=0}^{k-1} 1/(min_d + i)
        # Compute this upper bound
        max_sum_num = 0
        max_sum_den = 1
        for i in range(k):
            d = min_d + i
            # max_sum += 1/d
            max_sum_num = max_sum_num * d + max_sum_den
            max_sum_den = max_sum_den * d
        # Reduce
        g = gcd(max_sum_num, max_sum_den)
        max_sum_num //= g
        max_sum_den //= g
        
        # Check if remaining > max_sum
        # remaining_num/remaining_den > max_sum_num/max_sum_den
        # remaining_num * max_sum_den > max_sum_num * remaining_den
        if remaining_num * max_sum_den > max_sum_num * remaining_den:
            return None
        
        # Pruning: min possible sum with k terms starting from min_d
        # is sum_{i=0}^{k-1} 1/(min_d + i) but that's the same as max... 
        # Actually the min is when we pick the largest denominators, which is unbounded.
        # But we need sum = remaining, and each term <= 1/min_d.
        # So we need k * (1/min_d) >= remaining_num/remaining_den
        # i.e., k * remaining_den >= remaining_num * min_d
        if k * remaining_den < remaining_num * min_d:
            return None
        
        # Determine range for first denominator d
        # 1/d <= remaining_num/remaining_den => d >= remaining_den/remaining_num
        # Also d >= min_d
        # And we need remaining - 1/d to be representable with k-1 terms starting from d+1
        # 1/d < remaining_num/remaining_den (strict, since we need at least one more term)
        # Actually 1/d <= remaining_num/remaining_den, and remaining - 1/d > 0
        
        # Lower bound for d: d >= ceil(remaining_den / remaining_num) if remaining_num > 0
        if remaining_num > 0:
            d_min = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        else:
            d_min = min_d
        
        # Upper bound: 1/d >= remaining_num/remaining_den / k (since remaining sum of k terms = remaining)
        # Actually, the largest term 1/d must satisfy: 1/d <= remaining_num/remaining_den
        # And 1/d >= remaining_num/remaining_den - (k-1)/(d+1) ... this is complex
        # Simpler: 1/d must be at least remaining_num/remaining_den / k (average)
        # But also, after picking d, the remaining (k-1) terms must sum to remaining - 1/d
        # and each is at most 1/(d+1), so (k-1)/(d+1) >= remaining - 1/d
        # => (k-1)/(d+1) >= remaining_num/remaining_den - 1/d
        # => (k-1)/(d+1) + 1/d >= remaining_num/remaining_den
        # This gives an upper bound on d.
        
        # For now, use a reasonable upper bound
        # 1/d >= remaining_num/remaining_den / k => d <= k * remaining_den / remaining_num
        if remaining_num > 0:
            d_max = (k * remaining_den) // remaining_num
            d_max = max(d_max, d_min)  # ensure at least one value
        else:
            d_max = d_min
        
        # Also cap d_max to keep search manageable
        # The denominator can be up to 10^7, but for small k, d won't be that large
        d_max = min(d_max, 10000000)
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            
            # Check 1/d <= remaining_num/remaining_den
            # 1/d <= remaining_num/remaining_den => remaining_den <= d * remaining_num
            if d * remaining_num < remaining_den:
                continue
            
            # Compute new remaining: remaining_num/remaining_den - 1/d
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num < 0:
                continue
            
            if new_num == 0:
                if k == 1:
                    return [d]
                else:
                    continue  # Need more terms but sum is already 0
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Recurse
            result = search(k - 1, d + 1, new_num, new_den)
            if result is not None:
                return [d] + result
        
        return None
    
    # Try increasing k from lower bound
    # Lower bound: k=8 (as computed above, k<8 impossible)
    for k in range(8, 16):
        if time.time() > deadline:
            break
        
        result = search(k, 5, 1, 1)
        if result is not None:
            return result
    
    # Fallback: known solution with more terms
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... 
    # Let's use a known Egyptian fraction for 1
    # 1 = 1/2 + 1/3 + 1/6, but 2 and 3 are < 5
    # 1 = 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/18 + 1/20 + 1/24 + 1/30 + 1/36 + 1/40 + 1/45 + 1/50 + 1/60 + 1/72 + 1/90 + 1/100 + 1/120 + 1/180 + 1/200 + 1/240 + 1/300 + 1/360 + 1/400 + 1/450 + 1/500 + 1/540 + 1/600 + 1/720 + 1/900 + 1/1000 + 1/1200 + 1/1800 + 1/2000 + 1/2400 + 1/3000 + 1/3600 + 1/4000 + 1/4500 + 1/5000 + 1/5400 + 1/6000 + 1/7200 + 1/9000 + 1/10000
    # That's too many. Let me use a simpler approach.
    
    # Use the greedy algorithm with denominators >= 5
    def greedy_egyptian():
        result = []
        num, den = 1, 1
        d = 5
        while num > 0 and d <= 10000000:
            # Find smallest d >= current_d such that 1/d <= num/den
            # d >= den/num
            d_min = max(d, (den + num - 1) // num)
            # Pick d_min
            result.append(d_min)
            # num/den -= 1/d_min
            num = num * d_min - den
            den = den * d_min
            g = gcd(num, den)
            num //= g
            den //= g
            d = d_min + 1
            if len(result) > 256:
                return None
        if num == 0 and 1 <= len(result) <= 256:
            return result
        return None
    
    result = greedy_egyptian()
    if result is not None:
        return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
