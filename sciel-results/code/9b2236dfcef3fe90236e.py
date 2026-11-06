import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 180
    
    def is_valid(S):
        """Check if set S has no three distinct a,b,c with 1/a = 1/b + 1/c."""
        S_set = set(S)
        for a in S:
            for b in S:
                if b == a:
                    continue
                # c = a*b / (b-a) must be integer in S, distinct from a and b
                if b > a:
                    num = a * b
                    den = b - a
                    if num % den == 0:
                        c = num // den
                        if c in S_set and c != a and c != b:
                            return False
                else:
                    # a > b, so 1/a < 1/b, can't have 1/a = 1/b + 1/c with c>0
                    pass
        return True
    
    # Try progressively lower starting points
    for start in range(85, 91):
        if time.time() > deadline - 5.0:
            break
        candidate = list(range(start, N + 1))
        if is_valid(candidate):
            return candidate
    
    # Fallback
    return list(range(91, N + 1))
# EVOLVE-BLOCK-END
