import time
import math

_DEADLINE = 90.0
_START = time.time()

def _time_left():
    return _DEADLINE - 6.0 - (time.time() - _START)

def _build_smooths(limit):
    out = []
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                out.append(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(set(out))

def construct():
    global _START
    _START = time.time()
    
    smooths = _build_smooths(128)
    n_smooths = len(smooths)
    
    # best[sum] = list of terms
    best = [None] * 129
    
    # DFS to enumerate all antichains of smooths with sum <= 128
    # current_terms is a list of indices or values? Values are better for checking.
    # To check antichain: for new x, check if any t in current_terms divides x.
    # Since we pick in increasing order, x is the largest.
    # So we just check if x % t == 0 for any t in current_terms.
    
    def dfs(start_idx, current_sum, current_terms):
        if _time_left() < 0:
            return
        
        # Update best if we have a valid sum (>=1)
        if current_sum > 0 and current_sum <= 128:
            if best[current_sum] is None or len(current_terms) < len(best[current_sum]):
                best[current_sum] = list(current_terms)
            # If we already have a 1-term solution, we can't improve on it.
            # But we must continue to find solutions for other sums.
            # However, if len(current_terms) == 1, we don't need to add more terms 
            # to *this* specific sum, but we still need to explore other branches?
            # No, adding more terms increases the sum. The current sum is fixed.
            # So for this specific path, we have recorded the best for this sum.
            # We can still recurse to find larger sums.
        
        # Pruning: if current_sum > 128, we would have returned, but we check before update.
        # Actually, we should check before recursing.
        
        for i in range(start_idx, n_smooths):
            x = smooths[i]
            if current_sum + x > 128:
                break
            
            # Check antichain property: x must not be divisible by any term in current_terms
            # Since current_terms are all < x (sorted), we just check if x % t == 0
            valid = True
            for t in current_terms:
                if x % t == 0:
                    valid = False
                    break
            
            if valid:
                dfs(i + 1, current_sum + x, current_terms + [x])
    
    dfs(0, 0, [])
    
    # Verify all targets 1..128 are found
    # If any are None, it means no antichain representation exists? 
    # The problem states "All targets require valid representations".
    # So we assume they exist.
    
    # Construct final answer
    answers = [None] * 128
    for n in range(1, 129):
        answers[n-1] = best[n]
        
    return answers
