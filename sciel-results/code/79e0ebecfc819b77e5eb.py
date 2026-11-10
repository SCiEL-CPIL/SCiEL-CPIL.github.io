"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache

# Try to import ortools for a powerful constraint solver
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

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
    Minimize max(representatives) - min(representatives).
    """
    divisors = P['divisors']
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Fallback solution: the divisors themselves
    best_ans = divisors[:]
    best_span = _span(best_ans)
    
    # If OR-Tools is available, use it for a rigorous optimization
    if HAS_ORTOOLS:
        model = cp_model.CpModel()
        
        # Variables
        # x[i] is the value for divisor i
        # k[i] is the multiplier such that x[i] = k[i] * divisors[i]
        # L is the minimum value (lower bound of the window)
        # U is the maximum value (upper bound of the window)
        # S is the span U - L
        
        x_vars = []
        k_vars = []
        
        for i in range(n):
            d = divisors[i]
            # k[i] must be at least 1
            k_i = model.NewIntVar(1, P['maximum'] // d, f'k_{i}')
            x_i = model.NewIntVar(1, P['maximum'], f'x_{i}')
            
            # Constraint: x_i = k_i * d
            model.Add(x_i == k_i * d)
            
            x_vars.append(x_i)
            k_vars.append(k_i)
        
        L = model.NewIntVar(1, P['maximum'], 'L')
        U = model.NewIntVar(1, P['maximum'], 'U')
        S = model.NewIntVar(0, P['maximum'], 'S')
        
        # Constraints: L <= x_i <= U for all i
        for i in range(n):
            model.Add(L <= x_vars[i])
            model.Add(x_vars[i] <= U)
            
        # Constraint: S = U - L
        model.Add(S == U - L)
        
        # Constraints: All x_i must be distinct
        # We can use AllDifferent
        model.AddAllDifferent(x_vars)
        
        # Objective: Minimize S
        model.Minimize(S)
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 75.0  # Leave 5s for overhead
        solver.parameters.num_workers = 16
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            candidate_ans = [int(solver.Value(x_vars[i])) for i in range(n)]
            if _is_valid_answer(candidate_ans, divisors):
                candidate_span = _span(candidate_ans)
                if candidate_span < best_span:
                    best_span = candidate_span
                    best_ans = candidate_ans
    
    # If OR-Tools is not available or didn't find a better solution,
    # use a fallback heuristic: Greedy search with local improvement
    if time.time() < deadline:
        # Greedy approach: Try to pack into a small window
        # Start with a small span estimate and increase
        # Based on the divisors, a span of ~40-50 is likely feasible
        
        # Heuristic: Try windows of increasing size
        # For each window size S, try a few positions L
        # This is a simplified version of the previous strategy but faster
        
        # Let's try a specific known good packing if we can find one quickly
        # Or just run a quick random local search
        
        rng = random.Random(42)
        
        # Local search: Start with random valid solution, try to reduce span
        # 1. Generate random valid solution
        # 2. For each step, try to move one value closer to the mean
        # 3. Repeat until no improvement
        
        def get_random_solution():
            used = set()
            res = []
            # Shuffle divisors to break symmetry
            indices = list(range(n))
            rng.shuffle(indices)
            
            for i in indices:
                d = divisors[i]
                # Pick a random multiple
                max_k = P['maximum'] // d
                # To keep them close, pick from a limited range of k
                # Let's pick k from [100, 200] initially to keep values in [1000, 8000]
                # But we need to ensure distinctness.
                
                # Better: Pick a target range, say [1000, 2000]
                # And find a multiple in that range not used.
                
                found = False
                # Try a few random ranges
                for _ in range(100):
                    lo = rng.randint(1, P['maximum'] - 1000)
                    hi = lo + 1000
                    k_min = (lo + d - 1) // d
                    k_max = hi // d
                    if k_min > k_max:
                        continue
                    candidates = [k * d for k in range(k_min, k_max + 1) if k * d not in used]
                    if candidates:
                        val = rng.choice(candidates)
                        used.add(val)
                        res.append((i, val))
                        found = True
                        break
                if not found:
                    # Fallback: just pick any unused multiple
                    k = 1
                    while k * d in used:
                        k += 1
                    val = k * d
                    used.add(val)
                    res.append((i, val))
            
            # Sort back to original order
            final_res = [0] * n
            for i, val in res:
                final_res[i] = val
            return final_res
        
        current_ans = get_random_solution()
        current_span = _span(current_ans)
        
        if current_span < best_span:
            best_span = current_span
            best_ans = current_ans[:]
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                d = divisors[i]
                current_val = current_ans[i]
                # Try to move this value closer to the center of the current span
                center = (max(current_ans) + min(current_ans)) / 2.0
                
                # Look for a better value in a neighborhood
                # We want to reduce the span, so we want to shrink the interval
                # Try values closer to the center, ensuring distinctness and divisibility
                
                # Let's try moving towards the center
                # The new value must be a multiple of d
                # And distinct from others
                
                # We can try a few steps in each direction
                step = 1
                max_steps = 50 # Don't search too far
                
                # Try moving left
                k_current = current_val // d
                for k in range(k_current - 1, k_current - max_steps - 1, -1):
                    if k < 1: break
                    new_val = k * d
                    if new_val in current_ans: continue
                    # Check if this reduces span
                    # We need to check if new_val is within [min, max] or if it shrinks the range
                    # Actually, just check if the new span is smaller
                    old_min = min(current_ans)
                    old_max = max(current_ans)
                    
                    # If new_val is outside [old_min, old_max], it might increase span
                    # Unless it replaces the min or max and is closer to the other extreme
                    # Simple check: calculate new span
                    new_ans = current_ans[:]
                    new_ans[i] = new_val
                    new_span = _span(new_ans)
                    if new_span < current_span:
                        current_ans = new_ans
                        current_span = new_span
                        if current_span < best_span:
                            best_span = current_span
                            best_ans = current_ans[:]
                        improved = True
                        break # Restart loop for this i? Or just continue?
                
                # Try moving right
                for k in range(k_current + 1, k_current + max_steps + 1):
                    new_val = k * d
                    if new_val > P['maximum']: break
                    if new_val in current_ans: continue
                    new_ans = current_ans[:]
                    new_ans[i] = new_val
                    new_span = _span(new_ans)
                    if new_span < current_span:
                        current_ans = new_ans
                        current_span = new_span
                        if current_span < best_span:
                            best_span = current_span
                            best_ans = current_ans[:]
                        improved = True
                        break
        
    # Final validation
    if not _is_valid_answer(best_ans, divisors):
        # Fallback to the trivial answer
        best_ans = divisors[:]
    
    # Ensure it's a list of Python ints
    result = [int(v) for v in best_ans]
    return result

# EVOLVE-BLOCK-END
