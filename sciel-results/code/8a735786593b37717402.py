import time
import random
from fractions import Fraction
from math import floor

def construct():
    start_time = time.time()
    # 90s wall clock limit, use 85s to be safe
    deadline = start_time + 85.0
    
    n = 30
    # Precompute weights as Fractions for exact arithmetic
    # w[i] corresponds to coefficient of 1/(i+1)
    # Sum S = sum(c[i] * (1/(i+1)))
    # We want to minimize |S| subject to S != 0
    
    weights = [Fraction(1, i + 1) for i in range(n)]
    
    # State: c (list of ints -1,0,1), S (Fraction)
    # Initialize randomly
    c = [random.choice([-1, 0, 1]) for _ in range(n)]
    S = Fraction(0, 1)
    for i in range(n):
        if c[i] != 0:
            S += c[i] * weights[i]
    
    best_S = S
    best_c = c[:]
    
    # If initial is 0, force non-zero
    if S == 0:
        # Pick a random index to flip to 1
        idx = random.randint(0, n - 1)
        c[idx] = 1
        S += weights[idx]
        if S == 0:
            c[(idx + 1) % n] = 1
            S += weights[(idx + 1) % n]
        best_S = S
        best_c = c[:]
        
    # Simulated Annealing Parameters
    # We want to minimize |S|. 
    # Cost function: f(S) = |S|
    # Since S is rational, |S| = |num|/den.
    # Comparing |S1| < |S2| is exact with Fractions.
    
    # Temperature schedule
    # Start high, decay exponentially
    T_start = 1.0
    T_end = 1e-10
    # We don't know the number of steps in advance, so we use time-based decay or fixed steps with time check.
    # Let's use a fixed number of steps that fits in the time budget, but break early if time runs out.
    # Or better: iterate until deadline, adjusting T based on elapsed time.
    
    current_time = time.time()
    
    # To make SA effective, we need to know the "scale" of the cost.
    # The minimum non-zero sum is likely very small (1/LCM or similar).
    # However, comparing Fractions directly is robust.
    
    # Strategy: 
    # 1. Try to move to a state with smaller |S| (always accept).
    # 2. Try to move to a state with larger |S| (accept with probability exp(-delta/T)).
    # But since S can be 0, and we must avoid 0, we treat 0 as a forbidden state or high cost.
    # Actually, the problem says "signed sum ... must be nonzero".
    # So if we move to 0, we reject it.
    
    # Optimization: 
    # Instead of recalculating S from scratch, update it incrementally.
    # Delta for changing c[i] from old to new is (new - old) * weights[i]
    
    while time.time() < deadline:
        # Decay temperature based on elapsed time
        elapsed = time.time() - start_time
        # Exponential decay: T = T_start * (T_end/T_start)^(elapsed / total_time)
        # Let's assume we want to reach T_end at deadline.
        # T = T_start * exp(-k * elapsed)
        # T_end = T_start * exp(-k * deadline) -> k = -ln(T_end/T_start) / deadline
        # k = ln(1e10) / 85 ≈ 23/85 ≈ 0.27
        k = 0.27
        T = T_start * (T_end / T_start) ** (elapsed / 85.0)
        
        # Pick a random index
        i = random.randint(0, n - 1)
        old_val = c[i]
        
        # Pick a new value different from old
        # Options: -1, 0, 1
        # Randomly select one of the other two
        if old_val == 0:
            new_val = random.choice([-1, 1])
        elif old_val == 1:
            new_val = random.choice([-1, 0])
        else: # old_val == -1
            new_val = random.choice([0, 1])
            
        delta_coeff = new_val - old_val
        if delta_coeff == 0:
            continue
            
        delta_S = delta_coeff * weights[i]
        new_S = S + delta_S
        
        # Check if new_S is 0
        if new_S == 0:
            # Reject moves to 0
            continue
            
        # Compare absolute values
        # We want to minimize |S|
        # cost = |S|
        old_cost = abs(S)
        new_cost = abs(new_S)
        
        if new_cost < old_cost:
            # Always accept improvement
            c[i] = new_val
            S = new_S
            
            # Update best
            if new_cost < abs(best_S):
                best_S = S
                best_c = c[:]
        else:
            # Accept worse move with probability exp(-delta/T)
            # delta = new_cost - old_cost
            # Since costs are Fractions, delta is Fraction.
            # exp(-delta/T) requires converting to float.
            # For small numbers, float precision might be an issue, but 
            # if |S| is very small, delta is small.
            # If |S| is large, delta is large, prob is ~0.
            try:
                delta_val = float(new_cost - old_cost)
                # If T is very small, exp(-large) is 0
                if T > 1e-15:
                    prob = 2.718281828 ** (-delta_val / T)
                    if random.random() < prob:
                        c[i] = new_val
                        S = new_S
            except:
                pass
                
        # Early termination if we find a "very small" sum?
        # The theoretical minimum is 1/LCM(1..30).
        # LCM(1..30) is huge.
        # If we get |S| < 1e-100, we are likely at the minimum.
        # But comparing Fractions is exact.
        # If best_S == Fraction(1, LCM), we can stop.
        # LCM(1..30) = 2329089562800
        # Let's check if best_S is the minimum possible.
        # The set of all possible sums is { k / LCM(1..30) | k is integer }.
        # So the minimum non-zero absolute value is 1 / LCM(1..30).
        LCM_30 = 2329089562800
        min_possible = Fraction(1, LCM_30)
        
        if abs(best_S) == min_possible:
            break

    # Return best coefficients found
    return best_c
