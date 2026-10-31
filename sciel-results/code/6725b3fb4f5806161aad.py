"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Generate small Sidon sets explicitly
    - Evaluate score for each
    - Return the best one
    - Optionally refine with local search
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def generate_sidon_set(n):
        """Greedy construction of a Sidon set of size n with minimal max value."""
        U = [0]
        U_set = {0}
        diffs = set()
        
        for _ in range(1, n):
            candidate = 1
            while True:
                # Check if candidate can be added
                valid = True
                for u in U_set:
                    d1 = candidate - u
                    d2 = u - candidate
                    if d1 in diffs or d2 in diffs:
                        valid = False
                        break
                if valid:
                    U.append(candidate)
                    U_set.add(candidate)
                    for u in U_set:
                        if u != candidate:
                            diffs.add(candidate - u)
                            diffs.add(u - candidate)
                    break
                candidate += 1
        
        return U

    # Generate Sidon sets for small n
    best_U = [0, 1]
    best_score = compute_score(best_U)
    
    for n in range(2, 10):
        U = generate_sidon_set(n)
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Also check some non-Sidon sets that might have better ratio/size tradeoff
    # e.g. arithmetic progressions
    for n in range(2, 10):
        for step in range(1, 5):
            U = [i * step for i in range(n)]
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_U = U

    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
