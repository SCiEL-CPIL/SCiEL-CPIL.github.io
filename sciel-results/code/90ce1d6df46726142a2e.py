import math

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 2, 4]
    best_score = calc_score(best)
    
    # Search over sets of the form {0, 1, B, B^2, ..., B^(n-1)}
    # This corresponds to the "base-expansion hierarchy" with sparse selection.
    # Analysis shows ratio |U-U|/|U+U| approaches 2 for large n,
    # but denominator log(2*max(U)+1) grows with n and B.
    # Optimal is small B and small n.
    
    for B in range(2, 10):
        for n in range(2, 10):
            # Generate U
            U = [0]
            val = 1
            for j in range(n):
                U.append(val)
                val *= B
            
            # Calculate score
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
    
    return best
