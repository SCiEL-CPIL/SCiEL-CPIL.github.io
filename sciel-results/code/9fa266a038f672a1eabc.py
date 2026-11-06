"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def _evaluate(coeffs):
    """Return (numerator, denominator) of the signed sum."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1  # term is c / (i+1)
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def construct():
    deadline = time.time() + 55
    n = 30
    L = _lcm_range(n)  # lcm(1..30) = 2329089562800
    weights = [L // (i + 1) for i in range(n)]

    # We want to minimize |sum c_i * weights[i]| where c_i in {-1,0,1} and sum != 0
    # This is a subset-sum-like problem.
    
    # Start with a trivial solution: c_28=1, c_29=-1 gives 1/29 - 1/30 = 1/870
    # N = L/29 - L/30 = L*(30-29)/(29*30) = L/870
    best_coeffs = [0] * 28 + [1, -1]
    
    def compute_N(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    best_N = compute_N(best_coeffs)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    # Start with all zeros, add terms greedily to minimize |N|
    coeffs = [0] * n
    current_N = 0
    
    # Try to build a solution by adding terms one by one
    # Sort indices by weight descending (largest terms first)
    indices_sorted = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    for idx in indices_sorted:
        # Try adding +1 or -1 at this position
        best_val = 0
        best_abs = abs(current_N)
        for val in [1, -1]:
            new_N = current_N + val * weights[idx]
            if abs(new_N) < best_abs:
                best_abs = abs(new_N)
                best_val = val
        if best_val != 0:
            coeffs[idx] = best_val
            current_N += best_val * weights[idx]
    
    if current_N != 0 and abs(current_N) < abs(best_N):
        best_N = current_N
        best_coeffs = list(coeffs)
    
    # Phase 2: Local search with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            coeffs = list(best_coeffs)
            current_N = best_N
        else:
            # Random initialization with some structure
            coeffs = [rng.choice([-1, 0, 1]) for _ in range(n)]
            current_N = compute_N(coeffs)
        
        # Hill climbing with pair moves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try single moves
            for i in range(n):
                for nv in [-1, 0, 1]:
                    if nv == coeffs[i]:
                        continue
                    delta = (nv - coeffs[i]) * weights[i]
                    new_N = current_N + delta
                    if new_N != 0 and abs(new_N) < abs(current_N):
                        coeffs[i] = nv
                        current_N = new_N
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try pair moves if no single move improves
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i != j:
                vi = rng.choice([-1, 0, 1])
                vj = rng.choice([-1, 0, 1])
                if vi != coeffs[i] or vj != coeffs[j]:
                    delta = (vi - coeffs[i]) * weights[i] + (vj - coeffs[j]) * weights[j]
                    new_N = current_N + delta
                    if new_N != 0 and abs(new_N) < abs(current_N):
                        coeffs[i] = vi
                        coeffs[j] = vj
                        current_N = new_N
                        improved = True
        
        if current_N != 0 and abs(current_N) < abs(best_N):
            best_N = current_N
            best_coeffs = list(coeffs)
        
        if abs(best_N) == 1:
            break
    
    # Phase 3: Simulated annealing for global exploration
    if time.time() < deadline - 10:
        T = 1000.0
        cooling = 0.995
        coeffs = list(best_coeffs)
        current_N = best_N
        
        for step in range(100000):
            if time.time() > deadline - 5:
                break
            
            # Random move
            if rng.random() < 0.7:
                # Single move
                i = rng.randrange(n)
                nv = rng.choice([-1, 0, 1])
                if nv != coeffs[i]:
                    delta = (nv - coeffs[i]) * weights[i]
                    new_N = current_N + delta
                    if new_N != 0:
                        d = abs(new_N) - abs(current_N)
                        if d < 0 or rng.random() < min(1.0, 1.0 / (1.0 + d * 1e-10)):
                            coeffs[i] = nv
                            current_N = new_N
            else:
                # Pair move
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i != j:
                    vi = rng.choice([-1, 0, 1])
                    vj = rng.choice([-1, 0, 1])
                    if vi != coeffs[i] or vj != coeffs[j]:
                        delta = (vi - coeffs[i]) * weights[i] + (vj - coeffs[j]) * weights[j]
                        new_N = current_N + delta
                        if new_N != 0:
                            d = abs(new_N) - abs(current_N)
                            if d < 0 or rng.random() < min(1.0, 1.0 / (1.0 + d * 1e-10)):
                                coeffs[i] = vi
                                coeffs[j] = vj
                                current_N = new_N
            
            if current_N != 0 and abs(current_N) < abs(best_N):
                best_N = current_N
                best_coeffs = list(coeffs)
            
            T *= cooling
        
        if abs(best_N) == 1:
            break
    
    # Fallback
    if best_N is None or best_N == 0:
        return [0] * 28 + [1, -1]
    
    return best_coeffs

# EVOLVE-BLOCK-END
