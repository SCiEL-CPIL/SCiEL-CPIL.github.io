from fractions import Fraction
import itertools
import random
import sys
from functools import lru_cache

def construct():
    # We want 30 coefficients from {-1,0,1} such that the signed harmonic sum
    # S = sum_{i=0}^{29} c[i] / (i+1) is nonzero, and |S| is minimized.
    #
    # Strategy: Use exact arithmetic with Fractions for precision.
    # 1) Meet-in-the-middle on the first 15 terms to get all possible sums.
    # 2) For the second 15 terms, enumerate and for each sum, find the closest
    #    sum from the first half that makes the total nonzero and minimal.
    #
    # 3^15 = 14348907 which is feasible if we do it carefully.
    # Actually, let's split 30 into two halves of 15 each.
    
    half1 = list(range(1, 16))   # denominators 1..15
    half2 = list(range(16, 31))  # denominators 16..30
    
    # Enumerate all 3^15 sign assignments for half1
    # Store as (sum, tuple_of_coeffs)
    # Use a dictionary to keep only the best (closest to 0 for each... no, we need all)
    # Actually, we need all sums because we want to pair with half2 sums.
    
    # Let's generate all sums for half1
    sums1 = []
    n1 = len(half1)
    for tup in itertools.product([-1, 0, 1], repeat=n1):
        s = Fraction(0, 1)
        for i, c in zip(half1, tup):
            if c != 0:
                s += Fraction(c, i)
        sums1.append((s, tup))
    
    # Sort by absolute value for potential pruning, but we need all for pairing
    # Actually, let's just keep all. 3^15 ≈ 14M might be too much memory in Python.
    # Let's try a smaller split or a different approach.
    
    # Alternative: Greedy + Local Search with exact arithmetic
    # The current program uses floating point which can cause precision issues.
    # Let's use Fractions for exactness.
    
    def eval_c(c):
        s = Fraction(0, 1)
        for i, coeff in enumerate(c):
            if coeff != 0:
                s += Fraction(coeff, i + 1)
        return s
    
    # Start with a random or heuristic initialization
    random.seed(42)
    
    best_val = None
    best_c = None
    
    # Try a few random starts and local search
    for restart in range(50):
        # Random start
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Local search: try flipping each coefficient
        improved = True
        while improved:
            improved = False
            for i in range(30):
                for new_val in [-1, 0, 1]:
                    if new_val == c[i]:
                        continue
                    old_val = c[i]
                    c[i] = new_val
                    s = eval_c(c)
                    if s != 0:
                        if best_val is None or abs(s) < abs(best_val):
                            best_val = s
                            best_c = c[:]
                            improved = True
                    c[i] = old_val
        if not improved:
            break
    
    # Additional: try to refine by considering pairs
    if best_c is None:
        # Fallback: simple heuristic
        best_c = [0] * 30
        best_c[0] = 1
        best_c[1] = -1
        best_val = eval_c(best_c)
    
    return best_c
