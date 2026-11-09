import numpy as np

def construct():
    N = 256
    S = set()
    candidates = list(range(1, N + 1))
    
    # Helper to check if adding x to S creates a 4-AP
    def creates_4ap(x, S):
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # x is 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x, (N - x) // 2) + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # x is 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # x is 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    # Helper to calculate gain in 3-APs if x is added to S
    def calc_gain(x, S):
        gain = 0
        # x is 1st: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                gain += 1
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x, N - x) + 1):
            if (x - d) in S and (x + d) in S:
                gain += 1
        # x is 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if (x - 2 * d) in S and (x - d) in S:
                gain += 1
        return gain

    while candidates:
        best_x = None
        best_gain = -1
        
        for x in candidates:
            if creates_4ap(x, S):
                continue
            
            gain = calc_gain(x, S)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
            elif gain == best_gain:
                # Tie-breaker: prefer smaller x? Or maybe larger x to spread out?
                # Smaller x is fine.
                if best_x is None or x < best_x:
                    best_x = x
        
        if best_x is None:
            break
            
        S.add(best_x)
        candidates.remove(best_x)
        
    return sorted(S)
