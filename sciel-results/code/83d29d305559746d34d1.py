import time
from fractions import Fraction

def construct():
    start_time = time.time()
    limit_time = 50.0  # Safety margin before 60s kill

    def time_up():
        return time.time() - start_time > limit_time

    def dfs(remaining, start_denom, terms_left, current):
        if time_up():
            return "TIMEOUT"
            
        if remaining == 0:
            return current[:]
        if terms_left == 0:
            return None
        if remaining < 0:
            return None
            
        if terms_left == 1:
            if remaining.denominator == 1:
                d = remaining.numerator
                if d >= start_denom and d <= 10000000:
                    return current + [d]
            return None

        # min_d: 1/d <= remaining => d >= 1/remaining
        min_d = max(start_denom, remaining.denominator // remaining.numerator + 1)
        
        # max_d: 1/d + (terms_left-1)/d >= remaining => d <= terms_left/remaining
        max_d = min(10000000, (terms_left * remaining.denominator) // remaining.numerator)
        
        for d in range(min_d, max_d + 1):
            next_remaining = remaining - Fraction(1, d)
            if next_remaining < 0:
                continue
            
            # Prune: Can we form next_remaining with terms_left-1 terms starting from d+1?
            # Max sum of next terms is approx (terms_left-1)/(d+1)
            # If next_remaining > (terms_left-1)/(d+1), prune.
            if next_remaining > Fraction(terms_left - 1, d + 1):
                continue
            
            result = dfs(next_remaining, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return result
        
        return None

    # Try to find a solution with k=8
    res = dfs(Fraction(1, 1), 5, 8, [])
    
    if res is not None and res != "TIMEOUT":
        # Verify
        if len(set(res)) == len(res) and all(5 <= x <= 10000000 for x in res):
            # Double check sum
            if sum(Fraction(1, x) for x in res) == 1:
                return res

    # Fallback: try larger k
    for k in range(9, 15):
        res = dfs(Fraction(1, 1), 5, k, [])
        if res is not None and res != "TIMEOUT":
            if len(set(res)) == len(res) and all(5 <= x <= 10000000 for x in res):
                if sum(Fraction(1, x) for x in res) == 1:
                    return res

    # Last resort: return a known valid solution if found by other means
    # Since the search is fast for k=8, this should not be reached.
    # If it is, we must return something valid.
    # A known solution with all denominators >= 5:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # This is not exact.
    
    # Let's use a simpler fallback:
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # This is not exact.
    
    # I will trust the DFS. It is very fast for k=8.
    # If it fails, I will raise an error.
    raise RuntimeError("No solution found")
