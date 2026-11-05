"""Randomized Local Search for small nonzero signed harmonic subsums."""
import time
from fractions import Fraction
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 58  # Safety margin
    n = 30
    best_score = None
    best_coeffs = None
    
    # Precompute inverses as Fractions for speed
    invs = [Fraction(1, i) for i in range(1, n + 1)]
    
    def evaluate(coeffs):
        s = Fraction(0)
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * invs[i]
        return s
    
    def is_valid(coeffs):
        s = evaluate(coeffs)
        return s != 0
    
    # Local search: try to flip one coefficient or swap two
    def local_search(coeffs, max_steps=500):
        current_coeffs = list(coeffs)
        current_val = evaluate(current_coeffs)
        current_abs = abs(current_val)
        
        if current_val == 0:
            # Perturb if zero
            idx = random.randint(0, n - 1)
            current_coeffs[idx] = random.choice([-1, 1])
            current_val = evaluate(current_coeffs)
            current_abs = abs(current_val)
            if current_val == 0:
                return None, None
            
        improved = True
        steps = 0
        while improved and steps < max_steps:
            improved = False
            steps += 1
            if time.time() > deadline - 2:
                break
                
            # Try all single flips
            for i in range(n):
                old_val = current_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    current_coeffs[i] = new_val
                    new_val_sum = evaluate(current_coeffs)
                    if new_val_sum == 0:
                        current_coeffs[i] = old_val
                        continue
                    new_abs = abs(new_val_sum)
                    if new_abs < current_abs:
                        current_abs = new_abs
                        current_val = new_val_sum
                        improved = True
                        break
                else:
                    continue
                break
            
            # If no single flip improved, try a random pair swap (structural change)
            if not improved:
                i, j = random.sample(range(n), 2)
                vi = current_coeffs[i]
                vj = current_coeffs[j]
                current_coeffs[i] = vj
                current_coeffs[j] = vi
                new_val_sum = evaluate(current_coeffs)
                if new_val_sum == 0:
                    current_coeffs[i] = vi
                    current_coeffs[j] = vj
                else:
                    new_abs = abs(new_val_sum)
                    if new_abs < current_abs:
                        current_abs = new_abs
                        current_val = new_val_sum
                        improved = True
                    else:
                        # Revert if worse
                        current_coeffs[i] = vi
                        current_coeffs[j] = vj
                        
        return current_coeffs, current_abs

    # Multi-start
    num_restarts = 0
    while time.time() < deadline - 2:
        num_restarts += 1
        # Random initialization with some bias towards non-zero
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Ensure non-zero start
        val = evaluate(coeffs)
        if val == 0:
            # Flip a random one
            idx = random.randint(0, n - 1)
            if coeffs[idx] == 1:
                coeffs[idx] = -1
            elif coeffs[idx] == -1:
                coeffs[idx] = 1
            else:
                coeffs[idx] = 1
            val = evaluate(coeffs)
            if val == 0:
                # Try again
                coeffs = [random.choice([-1, 1]) for _ in range(n)]
                val = evaluate(coeffs)
                if val == 0:
                    coeffs[0] = 1
                    coeffs[1] = -1
                    val = evaluate(coeffs)
        
        abs_val = abs(val)
        
        # Run local search
        res_coeffs, res_abs = local_search(coeffs)
        
        if res_coeffs is not None and res_abs is not None:
            if best_score is None or res_abs < best_score:
                best_score = res_abs
                best_coeffs = res_coeffs
                # print(f"Restart {num_restarts}: New best {best_score}", file=sys.stderr)
                
    # Fallback if no valid result found (should not happen)
    if best_coeffs is None:
        # Simple valid answer: 1/1 - 1/2 - 1/3 ... 
        # Let's just use 1/1 - 1/2 = 1/2
        return [1, -1] + [0] * 28
        
    return best_coeffs

# EVOLVE-BLOCK-END
