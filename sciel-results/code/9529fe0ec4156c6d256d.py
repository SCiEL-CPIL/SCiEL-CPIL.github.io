"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 78  # 78s budget, 12s safety margin
    n = 30
    
    # Precompute unit fractions
    inv = [Fraction(1, i) for i in range(1, n + 1)]
    
    def sum_from_coeff(coeff):
        s = Fraction(0, 1)
        for i, c in enumerate(coeff):
            if c != 0:
                s += c * inv[i]
        return s
    
    # Start with seed
    best_coeff = [0] * 28 + [1, -1]
    best_val = sum_from_coeff(best_coeff)
    best_abs = abs(best_val)
    
    # Local search with incremental updates
    def local_search(coeff, current_sum, max_iters=1000):
        """Local search starting from coeff with current_sum. Returns (coeff, sum, abs_sum)"""
        best_c = coeff[:]
        best_s = current_sum
        best_a = abs(current_sum)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            improved = False
            for idx in range(n):
                old = best_c[idx]
                for new_val in (-1, 0, 1):
                    if new_val == old:
                        continue
                    # Delta = (new_val - old) * inv[idx]
                    delta = (new_val - old) * inv[idx]
                    new_sum = best_s + delta
                    if new_sum != 0 and abs(new_sum) < best_a:
                        best_c[idx] = new_val
                        best_s = new_sum
                        best_a = abs(new_sum)
                        improved = True
            if not improved:
                break
        
        return best_c, best_s, best_a
    
    # Phase 1: Local search from seed
    best_coeff, best_val, best_abs = local_search(best_coeff, best_val, max_iters=5000)
    
    # Phase 2: Try systematic pair modifications
    if time.time() < deadline - 5:
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                oi = best_coeff[i]
                oj = best_coeff[j]
                for vi in (-1, 0, 1):
                    for vj in (-1, 0, 1):
                        if vi == oi and vj == oj:
                            continue
                        delta = (vi - oi) * inv[i] + (vj - oj) * inv[j]
                        new_sum = best_val + delta
                        if new_sum != 0 and abs(new_sum) < best_abs:
                            best_coeff[i] = vi
                            best_coeff[j] = vj
                            best_val = new_sum
                            best_abs = abs(new_sum)
                            # Do a quick local search from here
                            if time.time() < deadline - 5:
                                best_coeff, best_val, best_abs = local_search(best_coeff, best_val, max_iters=200)
    
    # Phase 3: Random restarts
    import random
    random.seed(12345)
    if time.time() < deadline - 2:
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            coeff = [random.choice([-1, 0, 1]) for _ in range(n)]
            s = sum_from_coeff(coeff)
            if s == 0:
                continue
            if abs(s) < best_abs:
                best_coeff, best_val, best_abs = local_search(coeff, s, max_iters=300)
    
    return best_coeff
# EVOLVE-BLOCK-END
