<<<<<<< SEARCH
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try to find a 6-term solution
    # ... (search logic) ...
    
    # Fallback to DP for 7+ terms
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
=======
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try to find solutions with increasing number of terms
    # Minimum possible is 7 terms (proven 6 or fewer impossible)
    
    def search_terms(num_terms, min_denom=5, remaining=Fraction(1,1), used=set(), time_budget=10.0):
        """Find num_terms distinct denominators >= min_denom summing to remaining."""
        if time.time() > start_time + time_budget:
            return None
        if num_terms == 1:
            if remaining.denominator == 1:
                d = remaining.numerator
                if d >= min_denom and d not in used and d <= 10000000:
                    return [d]
            return None
        
        # The smallest denominator must satisfy: 1/d >= remaining/num_terms
        # So d <= num_terms / remaining
        max_d = int(num_terms / remaining)
        # Also d >= min_denom and d not in used
        # And we need at least (num_terms-1) more terms each with denom > d
        # So 1/d < remaining (obvious) and remaining - 1/d must be achievable with num_terms-1 terms
        
        start = max(min_denom, 5)
        # d must be such that remaining - 1/d > 0
        # and remaining - 1/d can be written as sum of (num_terms-1) unit fractions with denom > d
        
        for d in range(start, max_d + 1):
            if d in used:
                continue
            rem2 = remaining - Fraction(1, d)
            if rem2 <= 0:
                continue
            # Check if rem2 can be represented with num_terms-1 terms, all > d
            # Quick check: (num_terms-1)/rem2 >= d+1, i.e., rem2 <= (num_terms-1)/(d+1)
            # Actually we need 1/(d+1) + ... to be >= rem2, so rem2 <= (num_terms-1)/(d+1)
            if rem2 > Fraction(num_terms - 1, d + 1):
                continue
            result = search_terms(num_terms - 1, d + 1, rem2, used | {d}, time_budget)
            if result is not None:
                return [d] + result
        return None
    
    # Try 7 terms first
    for n in range(7, 257):
        if time.time() > deadline - 2:
            break
        result = search_terms(n, 5, Fraction(1, 1), set(), min(15.0, deadline - time.time() - 1))
        if result is not None:
            return result
    
    # Fallback: use a known decomposition
    # 1/5 + 1/7 + 1/11 + 1/13 + 1/22 + 1/30 + 1/35 + 1/42 + 1/46 + 1/55 + 1/66 + 1/77 + 1/110 + 1/143 + 1/165 + 1/198 + 1/231 + 1/286 + 1/330 + 1/429 + 1/462 + 1/506 + 1/660 + 1/715 + 1/924 + 1/1001 + 1/1155 + 1/1386 + 1/1540 + 1/2310 + 1/2706 + 1/3003 + 1/4620 + 1/5005 + 1/5511 + 1/6630 + 1/7735 + 1/8282 + 1/9240 + 1/10010 + 1/11022 + 1/13860 + 1/15470 + 1/16515 + 1/18480 + 1/20020 + 1/22044 + 1/27720 + 1/30940 + 1/33030 + 1/36960 + 1/40040 + 1/44088 + 1/55440 + 1/61880 + 1/66060 + 1/73920 + 1/80080 + 1/88176 + 1/110880 + 1/123760 + 1/132120 + 1/147840 + 1/160160 + 1/176352 + 1/221760 + 1/247520 + 1/264240 + 1/295680 + 1/320320 + 1/352704 + 1/443520 + 1/495040 + 1/528480 + 1/591360 + 1/640640 + 1/705408 + 1/887040 + 1/990080 + 1/1056960 + 1/1182720 + 1/1281280 + 1/1410816 + 1/1774080 + 1/1980160 + 1/2113920 + 1/2365440 + 1/2562560 + 1/2821632 + 1/3548160 + 1/3960320 + 1/4227840 + 1/4730880 + 1/5125120 + 1/5643264 + 1/7096320 + 1/7920640 + 1/8455680 + 1/9461760 + 1/10250240 + 1/11286528 + 1/14192640 + 1/15841280 + 1/16911360 + 1/18923520 + 1/20500480 + 1/22573056 + 1/28385280 + 1/31682560 + 1/33822720 + 1/37847040 + 1/41000960 + 1/45146112 + 1/56770560 + 1/63365120 + 1/67645440 + 1/75694080 + 1/82001920 + 1/90292224 + 1/113541120 + 1/126730240 + 1/135290880 + 1/151388160 + 1/164003840 + 1/180584448 + 1/227082240 + 1/253460480 + 1/270581760 + 1/302776320 + 1/328007680 + 1/361168896 + 1/454164480 + 1/506920960 + 1/541163520 + 1/605552640 + 1/656015360 + 1/722337792 + 1/908328960 + 1/1013841920 + 1/1082327040 + 1/1211105280 + 1/1312030720 + 1/1444675584 + 1/1816657920 + 1/2027683840 + 1/2164654080 + 1/2422210560 + 1/2624061440 + 1/2889351168 + 1/3633315840 + 1/4055367680 + 1/4329308160 + 1/4844421120 + 1/5248122880 + 1/5778702336 + 1/7266631680 + 1/8110735360 + 1/8658616320 + 1/9688842240 + 1/10496245760 + 1/11557404672
    # That's way too many. Let me use a simpler fallback.
    
    # Simple fallback: use the fact that 1 = 1/2 + 1/3 + 1/6, but we need denom >= 5
    # Use a greedy-like approach or a precomputed small solution
    # Actually, let's just return a valid (if not minimal) solution as fallback
    # 1/5 + 1/7 + 1/11 + 1/13 + 1/22 + 1/30 + 1/35 + 1/42 + 1/46 + 1/55 + 1/66 + 1/77 + 1/110 + 1/143 + 1/165 + 1/198 + 1/231 + 1/286 + 1/330 + 1/429 + 1/462 + 1/506 + 1/660 + 1/715 + 1/924 + 1/1001 + 1/1155 + 1/1386 + 1/1540 + 1/2310
    # Let me verify a simpler one:
    # Actually, let me just do a quick greedy search for fallback
    
    # Fallback: find any valid solution using a simple method
    remaining = Fraction(1, 1)
    result = []
    used = set()
    for d in range(5, 10000000):
        if time.time() > deadline - 1:
            break
        if d in used:
            continue
        term = Fraction(1, d)
        if term <= remaining:
            # Check if remaining - term can be achieved with larger denominators
            # Greedy: just take it if it doesn't overshoot
            if remaining - term >= 0:
                result.append(d)
                used.add(d)
                remaining -= term
                if remaining == 0:
                    if len(result) >= 1 and len(result) <= 256:
                        return result
                # Limit to 256 terms max
                if len(result) >= 256:
                    break
    
    raise RuntimeError("no representation found")
>>>>>>> REPLACE

Wait, that fallback is terrible. Let me think more carefully. The key insight is that we need to find the MINIMUM number of terms. Let me write a proper search.

Actually, let me reconsider. The search for 7 terms should work if I implement it properly. Let me trace through:

For 7 terms, a=5: remaining = 4/5, need 6 more terms with denom > 5.
b: 1/b >= (4/5)/6 = 2/15, so b <= 15/2 = 7.5, so b <= 7. b >= 6.
  b=6: remaining = 4/5 - 1/6 = 19/30. Need 5 terms with denom > 6.
    c: 1/c >= (19/30)/5 = 19/150, so c <= 150/19 ≈ 7.9, so c <= 7. c >= 7.
    c=7: remaining = 19/30 - 1/7 = 133/210 - 30/210 = 103/210. Need 4 terms with denom > 7.
      d: 1/d >= (103/210)/4 = 103/840, so d <= 840/103 ≈ 8.15, so d <= 8. d >= 8.
      d=8: remaining = 103/210 - 1/8 = 824/1680 - 210/1680 = 614/1680 = 307/840. Need 3 terms with denom > 8.
        e: 1/e >= (307/840)/3 = 307/2520, so e <= 2520/307 ≈ 8.2. e >= 9. Contradiction.
      
  So a=5, b=6, c=7, d=8 doesn't work for 3 remaining terms.

Let me just implement the search properly and let it find the answer.

