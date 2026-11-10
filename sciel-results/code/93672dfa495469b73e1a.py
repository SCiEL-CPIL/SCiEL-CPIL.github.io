"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _is_valid_answer(ans, divisors):
    """Check if answer is valid: 10 distinct positive ints <= 1e6, each divisible by corresponding divisor."""
    if not isinstance(ans, (list, tuple)):
        return False
    if len(ans) != len(divisors):
        return False
    seen = set()
    for i, (val, div) in enumerate(zip(ans, divisors)):
        if not isinstance(val, int) or isinstance(val, bool):
            return False
        if val < 1 or val > P['maximum']:
            return False
        if val % div != 0:
            return False
        if val in seen:
            return False
        seen.add(val)
    return True

def _span(ans):
    return max(ans) - min(ans)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, aligned with divisors [6,10,14,15,21,22,26,33,35,39].
    Each representative must be divisible by its corresponding input.
    Minimize max(representatives) - min(representatives), the length of their closed enclosing interval.
    
    Strategy: Use Integer Linear Programming (ILP) via scipy.optimize.milp.
    Variables:
    - x_i (for i in 0..9): The multiple count for divisor i. Value is d_i * x_i.
    - L: The lower bound of the interval.
    - U: The upper bound of the interval.
    
    Objective: Minimize U - L.
    Constraints:
    1. L <= d_i * x_i <= U for all i.
    2. d_i * x_i != d_j * x_j for i != j (Distinctness).
       This is non-linear/complicated for standard ILP. 
       
    Wait, scipy.milp handles linear constraints. Distinctness `d_i x_i != d_j x_j` is non-linear.
    However, we can handle distinctness by observing that if we minimize the span, 
    and the values are integers, collisions are specific events.
    
    Actually, standard ILP solvers don't handle `!=` directly.
    We can use a "big-M" formulation or simply solve the relaxed problem where distinctness is not enforced, 
    and then repair if collisions occur. 
    Given the small number of variables (10), collisions are rare in the optimal tight packing 
    unless the divisors share factors (which they do: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39).
    
    Let's try a different structural approach that is still "ILP-like" but handles the discrete nature better:
    Since the span is small, we can enumerate the span S.
    But the directive asks for a DIFFERENT strategy.
    
    Let's use **Branch and Bound with Linear Relaxation** or simply a **Smart Exhaustive Search** 
    that is structurally different from the previous "window scanning".
    
    Actually, let's stick to the ILP idea but handle distinctness via **Binary Variables**.
    Variables:
    x_i: integer multiple count for divisor i.
    y_{ij}: binary variable indicating if x_i collides with x_j? No, that's too many vars.
    
    Alternative Structural Change:
    Use **Simulated Annealing** with a strong penalty for span and collisions.
    This is a metaheuristic, structurally different from the deterministic backtracking/window scan.
    
    Let's implement Simulated Annealing.
    State: List of 10 integers (multiples).
    Energy: Span + Penalty * (Number of Collisions + Number of Constraint Violations).
    
    This explores the global landscape rather than local windows.
    """
    divisors = P['divisors']
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial best: Greedy
    def greedy_init():
        # Sort divisors
        indices = sorted(range(n), key=lambda i: divisors[i])
        used = set()
        res = [0] * n
        min_val = 1
        # Try to pack tightly
        # Start with a reasonable guess
        for i in indices:
            d = divisors[i]
            k = max(1, (min_val + d - 1) // d)
            val = k * d
            while val in used:
                k += 1
                val = k * d
            res[i] = val
            used.add(val)
            if val > min_val:
                min_val = val
        return res

    best_ans = greedy_init()
    best_span = _span(best_ans) if _is_valid_answer(best_ans, divisors) else float('inf')
    
    if not _is_valid_answer(best_ans, divisors):
        best_ans = divisors[:]
        best_span = _span(best_ans)

    # Simulated Annealing
    rng = random.Random(42)
    current_ans = best_ans[:]
    current_energy = float('inf')
    
    # Calculate Energy
    def calc_energy(ans):
        if not _is_valid_answer(ans, divisors):
            # Heavy penalty for invalid
            penalty = 0
            seen = set()
            for i, (val, div) in enumerate(zip(ans, divisors)):
                if not isinstance(val, int) or isinstance(val, bool):
                    return 1e9
                if val < 1 or val > P['maximum']:
                    penalty += 10000
                if val % div != 0:
                    penalty += 10000
                if val in seen:
                    penalty += 5000
                else:
                    seen.add(val)
            return 100000 + penalty + _span(ans)
        
        return _span(ans)

    current_energy = calc_energy(current_ans)
    
    T = 1000.0
    cooling_rate = 0.999
    
    for step in range(1000000):
        if time.time() > deadline:
            break
            
        # Generate neighbor: pick one index and change its multiple
        idx = rng.randint(0, n - 1)
        d = divisors[idx]
        current_val = current_ans[idx]
        
        # Random walk in the multiple space
        # Bias towards staying close to current min/max
        min_v = min(current_ans)
        max_v = max(current_ans)
        
        # Propose a new value
        # 10% chance to jump far
        if rng.random() < 0.1:
            new_k = rng.randint(1, P['maximum'] // d)
        else:
            # Small step
            step_size = rng.randint(-5, 5)
            new_k = (current_val // d) + step_size
            if new_k < 1:
                new_k = 1
            if new_k * d > P['maximum']:
                new_k = P['maximum'] // d
                
        new_val = new_k * d
        
        if new_val == current_val:
            continue
            
        # Create new state
        old_val = current_ans[idx]
        current_ans[idx] = new_val
        
        new_energy = calc_energy(current_ans)
        
        # Acceptance
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_span and _is_valid_answer(current_ans, divisors):
                best_span = new_energy
                best_ans = current_ans[:]
        else:
            # Reject
            current_ans[idx] = old_val
            
        T *= cooling_rate
        if T < 1.0:
            T = 1.0

    # Final Validation
    if not _is_valid_answer(best_ans, divisors):
        best_ans = divisors[:]
        
    result = [int(v) for v in best_ans]
    return result

# EVOLVE-BLOCK-END
