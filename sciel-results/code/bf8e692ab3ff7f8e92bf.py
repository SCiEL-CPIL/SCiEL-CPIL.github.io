import numpy as np
from math import gcd
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # 85s limit to be safe (evaluator kills at 90s)
    
    # Target: Minimize |sum c_i / i| for c_i in {-1, 0, 1}, sum != 0.
    # Strategy: Random restarts + Local Search (Greedy + Simulated Annealing style)
    
    best_val = None
    best_coeff = None
    best_abs_val = float('inf')
    
    def evaluate(coeffs):
        # Returns (value, abs_value)
        # Using Fraction for exactness
        total = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                total += Fraction(c, i)
        return total, abs(total)
        
    def local_search(coeffs, max_steps=2000):
        # Greedy local search with random restarts inside? No, just greedy moves.
        # We try to flip coefficients to reduce |sum|.
        # Moves:
        # 1. c -> c+1 (if c < 1)
        # 2. c -> c-1 (if c > -1)
        # 3. c -> 0 (if c != 0)
        # 4. 0 -> 1 or 0 -> -1
        
        current_coeffs = list(coeffs)
        current_val, current_abs = evaluate(current_coeffs)
        
        if current_abs == 0:
             # If we hit 0, we need to move away.
             # Just pick a random move to get non-zero.
             pass
             
        improved = True
        steps = 0
        while improved and steps < max_steps:
            improved = False
            steps += 1
            if time.time() > deadline:
                break
                
            # Generate possible moves
            moves = []
            for i in range(30):
                c = current_coeffs[i]
                # Possible next states for c
                if c == -1:
                    moves.append((i, 0))
                    moves.append((i, 1))
                elif c == 0:
                    moves.append((i, -1))
                    moves.append((i, 1))
                elif c == 1:
                    moves.append((i, 0))
                    moves.append((i, -1))
                    
            # Randomize order to escape symmetry/local traps? 
            # Or just try all? 30*2 = 60 moves. 
            # Evaluating 60 fractions each step is fast.
            random.shuffle(moves)
            
            best_move = None
            best_new_abs = current_abs
            
            for i, new_c in moves:
                # Try the move
                old_c = current_coeffs[i]
                current_coeffs[i] = new_c
                
                new_val, new_abs = evaluate(current_coeffs)
                
                # We want non-zero. If new_val is 0, we penalize it heavily or ignore it unless current is also 0?
                # The task requires non-zero.
                if new_val == 0:
                    # If we are currently non-zero and move to 0, bad.
                    # If we are currently 0 and move to 0, bad.
                    # We should avoid 0.
                    # Let's just say 0 is "infinity" bad for the objective "minimize non-zero abs".
                    # But wait, if we are stuck at a local min that is non-zero, we shouldn't go to 0.
                    # If we are at 0, we must leave.
                    # Let's treat 0 as very large cost.
                    new_abs = float('inf')
                    
                if new_abs < best_new_abs:
                    best_new_abs = new_abs
                    best_move = (i, new_c, old_c)
                    
                # Revert
                current_coeffs[i] = old_c
                
            if best_move is not None:
                i, new_c, old_c = best_move
                current_coeffs[i] = new_c
                current_val, current_abs = evaluate(current_coeffs)
                improved = True
            else:
                # No improvement found in this full pass
                # Try a random kick to escape local minimum
                if random.random() < 0.1: # 10% chance to kick
                    i = random.randint(0, 29)
                    old_c = current_coeffs[i]
                    # Pick a random different value
                    options = [x for x in [-1, 0, 1] if x != old_c]
                    new_c = random.choice(options)
                    current_coeffs[i] = new_c
                    current_val, current_abs = evaluate(current_coeffs)
                    # If this makes it 0, that's bad, but maybe it leads somewhere?
                    # Actually, let's just accept it and see if next steps fix it.
                    # But if it's 0, we are in a "trap" for the non-zero constraint.
                    # Let's force a non-zero kick if we are at 0.
                    if current_val == 0:
                         # Try to find a single move that makes it non-zero
                         for k in range(30):
                             oc = current_coeffs[k]
                             for nc in [x for x in [-1, 0, 1] if x != oc]:
                                 current_coeffs[k] = nc
                                 v, av = evaluate(current_coeffs)
                                 if v != 0:
                                     current_val = v
                                     current_abs = av
                                     break
                                 current_coeffs[k] = oc
                             if current_val != 0:
                                 break
                                 
        return current_coeffs, current_val, current_abs

    # Main Loop: Random Restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        
        # Initialize random coefficients
        # Bias towards non-zero? Or uniform?
        # Uniform over {-1, 0, 1}
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Ensure initial sum is not 0? If it is, local search will handle it.
        
        # Run local search
        new_coeffs, new_val, new_abs = local_search(coeffs, max_steps=1000)
        
        if new_val != 0:
            if new_abs < best_abs_val:
                best_abs_val = new_abs
                best_val = new_val
                best_coeff = new_coeffs
                # Print progress occasionally? No, avoid stdout overhead if possible.
                
    # Fallback if no non-zero found (shouldn't happen)
    if best_coeff is None:
        # Return a simple non-zero solution
        best_coeff = [0]*29 + [1]
        best_val = Fraction(1, 30)
        best_abs_val = abs(best_val)
        
    # Ensure native Python ints
    return [int(c) for c in best_coeff]

if __name__ == "__main__":
    # Test
    res = construct()
    print(res)
    s = sum(c / i for i, c in enumerate(res, 1))
    print(s)
