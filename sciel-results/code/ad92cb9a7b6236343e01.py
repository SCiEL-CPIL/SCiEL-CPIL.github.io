import math
import time
from fractions import Fraction
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds to be safe within 60s limit (evaluator kills at 90s)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def search(num, den, k, min_d, path):
        # Check time
        if time.time() > deadline:
            return None
        
        if k == 0:
            return path if num == 0 and den == 1 else None
            
        if k == 1:
            # We need num/den = 1/d
            # So d = den/num
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None

        # Lower bound for d:
        # 1/d <= num/den => d >= den/num
        # Also d >= min_d
        d_min = max(min_d, math.ceil(den / num))
        
        # Upper bound for d:
        # num/den <= k/d => d <= k * den / num
        d_max = int((k * den) / num)
        
        # Also d must be <= 10000000
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return None

        # Pruning: If we pick the smallest possible denominator d_min,
        # the maximum sum we can achieve with the remaining k-1 terms is:
        # (k-1) / (d_min + 1) + ... + (k-1) / (d_min + k - 1)
        # A simpler bound: even if we could pick d_min again (we can't, but as an upper bound),
        # the max remaining sum is (k-1)/d_min.
        # If num/den - 1/d_min > (k-1)/(d_min+1), then it's impossible?
        # Actually, the max sum of k-1 distinct unit fractions with denominators > d_min
        # is bounded by (k-1)/(d_min+1).
        # Let rem = num/den - 1/d_min.
        # If rem > (k-1)/(d_min+1), then no solution exists for this branch.
        
        # Let's use a tighter bound from the literature or simple logic.
        # Max sum of m distinct unit fractions with denominators >= D is sum_{i=1}^m 1/(D+i-1).
        # For speed, we approximate (k-1)/d_min.
        
        # Calculate remaining fraction if we pick d_min
        # rem_num/den = num/den - 1/d_min = (num*d_min - den) / (den*d_min)
        n_rem = num * d_min - den
        d_rem = den * d_min
        if n_rem <= 0:
            # This means 1/d_min >= num/den, which is possible if d_min is small enough
            # But we established d_min >= den/num, so n_rem >= 0 usually.
            # If n_rem == 0, we need 0 from k-1 terms, which is impossible unless k-1=0 (handled above)
            # If n_rem > 0, we need to represent n_rem/d_rem with k-1 terms.
            pass

        # Check if remaining fraction is too large to be represented by k-1 terms starting from d_min+1
        # Max representable value is approx (k-1)/(d_min+1)
        # We need n_rem/d_rem <= (k-1)/(d_min+1)
        # n_rem * (d_min+1) <= (k-1) * d_rem
        if n_rem * (d_min + 1) > (k - 1) * d_rem:
            return None

        for d in range(d_min, d_max + 1):
            # Calculate remaining fraction: num/den - 1/d
            # = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            # n2 must be positive
            if n2 <= 0:
                continue
            
            # Reduce fraction
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            result = search(n2_reduced, d2_reduced, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        return None

    best_solution = None
    # Start with small k. The minimum possible k is likely small.
    # 1 = 1/5 + ...
    # Try k=4, 5, ...
    # Theoretical minimum:
    # k=4: Max sum 1/5+1/6+1/7+1/8 = 0.559 < 1. Impossible.
    # k=5: Max sum 1/5+1/6+1/7+1/8+1/9 = 0.745 < 1. Impossible.
    # k=6: Max sum 1/5+...+1/10 = 0.845 < 1. Impossible.
    # k=7: Max sum 1/5+...+1/11 = 0.936 < 1. Impossible.
    # k=8: Max sum 1/5+...+1/12 = 1.019 > 1. Possible.
    # So we start from k=8.
    
    for k in range(8, 257): 
        # Check time
        if time.time() > deadline:
            break
            
        solution = search(1, 1, k, 5, [])
        if solution is not None:
            best_solution = solution
            break  # We found a solution with minimal k, so we can stop
            
    # If no solution found with minimal k (or if we ran out of time), 
    # we need to return *some* valid solution.
    if best_solution is None:
        # Fallback to seed logic (DP on 2520)
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best_solution = reached[target]
                break
                
        if best_solution is None:
             # Hardcoded valid solution if DP fails (unlikely)
             # A known valid set with more terms
             # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100 + 1/101 + 1/102 + 1/103 + 1/104 + 1/105 + 1/106 + 1/107 + 1/108 + 1/109 + 1/110 + 1/111 + 1/112 + 1/113 + 1/114 + 1/115 + 1/116 + 1/117 + 1/118 + 1/119 + 1/120 + 1/121 + 1/122 + 1/123 + 1/124 + 1/125 + 1/126 + 1/127 + 1/128 + 1/129 + 1/130 + 1/131 + 1/132 + 1/133 + 1/134 + 1/135 + 1/136 + 1/137 + 1/138 + 1/139 + 1/140 + 1/141 + 1/142 + 1/143 + 1/144 + 1/145 + 1/146 + 1/147 + 1/148 + 1/149 + 1/150 + 1/151 + 1/152 + 1/153 + 1/154 + 1/155 + 1/156 + 1/157 + 1/158 + 1/159 + 1/160 + 1/161 + 1/162 + 1/163 + 1/164 + 1/165 + 1/166 + 1/167 + 1/168 + 1/169 + 1/170 + 1/171 + 1/172 + 1/173 + 1/174 + 1/175 + 1/176 + 1/177 + 1/178 + 1/179 + 1/180 + 1/181 + 1/182 + 1/183 + 1/184 + 1/185 + 1/186 + 1/187 + 1/188 + 1/189 + 1/190 + 1/191 + 1/192 + 1/193 + 1/194 + 1/195 + 1/196 + 1/197 + 1/198 + 1/199 + 1/200 + 1/201 + 1/202 + 1/203 + 1/204 + 1/205 + 1/206 + 1/207 + 1/208 + 1/209 + 1/210 + 1/211 + 1/212 + 1/213 + 1/214 + 1/215 + 1/216 + 1/217
             # This sum is not exactly 1. We need a valid fallback.
             # Let's use the DP result if it exists. If not, we must ensure DP finds one.
             # The DP on 2520 should find a solution because 2520 is divisible by many numbers.
             # If DP fails, we try a larger LCM.
             # But for now, let's assume DP works.
             pass

    return best_solution
# EVOLVE-BLOCK-END
