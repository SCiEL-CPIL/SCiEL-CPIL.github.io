"""Program for small nonzero signed harmonic subsums."""
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit

    n = 30
    # The seed gives [0]*28 + [1, -1], which means:
    # c_29 = 1, c_30 = -1, others 0
    # S = 1/29 - 1/30 = (30-29)/(29*30) = 1/870
    # So the seed value is 1/870 ≈ 0.001149...
    
    # To beat this, we need |S| < 1/870 with S != 0.
    
    # Strategy: Systematic search over configurations with small support,
    # then local search / hill climbing to refine.
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        return s
    
    # Start with the seed configuration
    coeffs = [0] * n
    coeffs[28] = 1  # c_29
    coeffs[29] = -1  # c_30
    
    current_sum = compute_sum(coeffs)
    best_sum = abs(current_sum)
    best_coeffs = coeffs[:]
    
    # Local search: try changing each coefficient to -1, 0, or 1
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = coeffs[i]
            for new_val in (-1, 0, 1):
                if new_val == old_val:
                    continue
                coeffs[i] = new_val
                s = compute_sum(coeffs)
                if s != 0:
                    abs_s = abs(s)
                    if abs_s < best_sum:
                        best_sum = abs_s
                        best_coeffs = coeffs[:]
                        improved = True
                coeffs[i] = old_val
    
    # If we found something better than seed, return it
    if best_sum < abs(compute_sum([0]*28 + [1,-1])):
        return best_coeffs
    
    # Otherwise try a more systematic search
    # Try pairs of terms first
    base_sum = Fraction(1, 29) - Fraction(1, 30)  # = 1/870
    
    # Try single non-zero coefficients - they all have |value| >= 1/30 > 1/870
    # So we need at least 2 non-zero coefficients.
    
    # Let's try all pairs (i, j) with i < j, c_i = a, c_j = b where a,b in {-1,1}
    # S = a/i + b/j
    # If a = b, S = a(1/i + 1/j) which has |S| >= 1/30 + 1/30 = 1/15 >> 1/870
    # If a = -b, S = a(1/i - 1/j) = a(j-i)/(ij)
    # |S| = |j-i|/(ij)
    # To minimize this, we want i and j close together (small j-i) and ij large.
    # Best case: j-i=1, i=29, j=30: |S| = 1/(29*30) = 1/870 (this is the seed)
    # Next: j-i=1, i=28, j=29: |S| = 1/(28*29) = 1/812 > 1/870
    # So among pairs with opposite signs and adjacent indices, the seed is optimal.
    
    # What about three or more terms?
    # Let's try: c_i = 1, c_j = -1, c_k = something
    # This is getting complex. Let's do a bounded exhaustive search over 
    # configurations with small support (number of non-zero coefficients).
    
    # Support size 2: already analyzed, best is 1/870
    # Support size 3: try all triples (i,j,k) with values in {-1,0,1} not all zero
    # Number of triples: C(30,3) * (3^3 - 1) for which positions are nonzero... 
    # Actually let's just iterate over all subsets of size 3 and all assignments.
    
    # Support size 3
    found_better = False
    for triple in combinations(range(n), 3):
        if time.time() > deadline:
            break
        i, j, k = triple
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    s = Fraction(a, i+1) + Fraction(b, j+1) + Fraction(c, k+1)
                    if s != 0:
                        if abs(s) < best_sum:
                            best_sum = abs(s)
                            best_coeffs = [0]*n
                            best_coeffs[i] = a
                            best_coeffs[j] = b
                            best_coeffs[k] = c
                            found_better = True
    
    if found_better:
        return best_coeffs
    
    # Support size 4 - this is C(30,4)*3^4 = 27405 * 81 ≈ 2.2M, might be feasible
    for quad in combinations(range(n), 4):
        if time.time() > deadline:
            break
        i, j, k, l = quad
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    for d in (-1, 1):
                        s = (Fraction(a, i+1) + Fraction(b, j+1) + 
                             Fraction(c, k+1) + Fraction(d, l+1))
                        if s != 0:
                            if abs(s) < best_sum:
                                best_sum = abs(s)
                                best_coeffs = [0]*n
                                best_coeffs[i] = a
                                best_coeffs[j] = b
                                best_coeffs[k] = c
                                best_coeffs[l] = d
                                found_better = True
    
    if found_better:
        return best_coeffs
    
    # If nothing better found, return seed
    return [0]*28 + [1, -1]

# EVOLVE-BLOCK-END
