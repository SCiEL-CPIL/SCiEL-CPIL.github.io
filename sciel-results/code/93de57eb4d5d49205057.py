"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

_divs = _get_divisors(5040)
_n_divs = len(_divs)

def construct():
    deadline = time.time() + 55.0
    target = 4979
    
    # Try increasing k from 1 upward
    for k in range(1, 257):
        if time.time() > deadline:
            break
        
        if k <= 5:
            # Use combinations for small k
            for combo in combinations(_divs, k):
                if sum(combo) == target:
                    return list(combo)
        elif k == 6:
            # Optimized 6-element search
            # Pre-filter: for 6 elements summing to target, 
            # each element must be at most target - sum of 5 smallest remaining
            # But let's just do a smarter nested loop
            # Use meet-in-the-middle: split into two groups of 3
            # All 3-element subsets and their sums
            triple_sums = {}
            for i in range(_n_divs):
                for j in range(i+1, _n_divs):
                    for m in range(j+1, _n_divs):
                        s = _divs[i] + _divs[j] + _divs[m]
                        if s < target:  # only keep sums less than target
                            if s not in triple_sums:
                                triple_sums[s] = []
                            triple_sums[s].append((_divs[i], _divs[j], _divs[m]))
            
            # Now for each pair of triples that don't share elements and sum to target
            # This is still expensive. Let's try a different approach.
            
            # Actually, let's just do the 6-combination search but with pruning
            # Sort divisors descending for better pruning
            sorted_desc = sorted(_divs, reverse=True)
            
            # 6-element search with pruning
            n = len(sorted_desc)
            for i in range(n - 5):
                if time.time() > deadline:
                    break
                if sorted_desc[i] > target:
                    continue
                for j in range(i + 1, n - 4):
                    s2 = sorted_desc[i] + sorted_desc[j]
                    if s2 >= target:
                        break
                    for m in range(j + 1, n - 3):
                        s3 = s2 + sorted_desc[m]
                        if s3 >= target:
                            break
                        for p in range(m + 1, n - 2):
                            s4 = s3 + sorted_desc[p]
                            if s4 >= target:
                                break
                            for q in range(p + 1, n - 1):
                                s5 = s4 + sorted_desc[q]
                                if s5 >= target:
                                    break
                                for r in range(q + 1, n):
                                    s6 = s5 + sorted_desc[r]
                                    if s6 == target:
                                        return [sorted_desc[i], sorted_desc[j], sorted_desc[m], 
                                                sorted_desc[p], sorted_desc[q], sorted_desc[r]]
                                    elif s6 > target:
                                        break
        else:
            # For k >= 7, use greedy with local search
            # First try greedy from largest
            remainder = target
            chosen = []
            for d in sorted(_divs, reverse=True):
                if d <= remainder:
                    chosen.append(d)
                    remainder -= d
                if remainder == 0:
                    return chosen
            if remainder != 0:
                # Try a different greedy: start from a different point
                # Or use a simple local search
                # For now, just return the greedy result if it works, else continue
                pass
            break
    
    # Fallback: greedy (should not reach here for this instance)
    remainder = target
    chosen = []
    for d in sorted(_divs, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen
# EVOLVE-BLOCK-END
