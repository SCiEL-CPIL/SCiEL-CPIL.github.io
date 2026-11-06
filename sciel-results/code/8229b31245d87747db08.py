"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def compute_square_nonzero(poly):
    square = np.convolve(poly, poly)
    return count_nonzero(square)

def construct():
    start_time = time.time()
    max_time = 55  # Leave 5s buffer
    best_score = float('inf')
    best_poly = None
    
    # Parameters for Simulated Annealing
    # State: list of 12 (exp, coeff) tuples
    # We maintain a dense array for fast convolution, but the state is sparse.
    
    def get_dense(state):
        max_exp = max(e for e, c in state)
        arr = [0] * (max_exp + 1)
        for e, c in state:
            arr[e] = c
        return arr

    def eval_state(state):
        # Check validity: distinct exponents, coeffs in [-64,64] and non-zero
        exponents = [e for e, c in state]
        if len(set(exponents)) != 12:
            return float('inf')
        for e, c in state:
            if c == 0 or c < -64 or c > 64 or e < 0 or e > 40:
                return float('inf')
        dense = get_dense(state)
        return compute_square_nonzero(dense)

    # Initialize state: 12 random distinct exponents in [0, 40], coeffs in [-64, 64]
    def init_state():
        exponents = random.sample(range(0, 41), 12)
        random.shuffle(exponents)
        state = []
        for e in exponents:
            # Bias towards small coefficients? No, let SA find it.
            # But +/-1 are good starts for cancellation.
            if random.random() < 0.5:
                c = random.choice([-1, 1])
            else:
                c = random.randint(-64, 64)
                if c == 0: c = 1
            state.append((e, c))
        return state

    # Move generators
    def mutate_exponent(state):
        """Change one exponent to a new random value."""
        new_state = [s[:] for s in state]
        i = random.randint(0, 11)
        old_e, c = new_state[i]
        new_e = random.randint(0, 40)
        # Avoid collision with other exponents
        while any(e == new_e for e, _ in new_state if e != old_e):
            new_e = random.randint(0, 40)
        new_state[i] = (new_e, c)
        return new_state

    def mutate_coefficient(state):
        """Change one coefficient to a new value."""
        new_state = [s[:] for s in state]
        i = random.randint(0, 11)
        e, old_c = new_state[i]
        # Strategy: try small changes or random jumps
        if random.random() < 0.7:
            # Local change
            delta = random.choice([-2, -1, 1, 2])
            new_c = old_c + delta
        else:
            # Random jump
            new_c = random.randint(-64, 64)
        
        if new_c == 0:
            new_c = 1
        if new_c < -64: new_c = -64
        if new_c > 64: new_c = 64
        
        new_state[i] = (e, new_c)
        return new_state

    # Simulated Annealing Loop
    # We run multiple restarts
    num_restarts = 50
    steps_per_restart = 500
    T_start = 10.0
    T_end = 0.1
    cooling_rate = (T_end / T_start) ** (1.0 / steps_per_restart)

    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        state = init_state()
        current_score = eval_state(state)
        T = T_start
        
        if current_score < best_score:
            best_score = current_score
            best_poly = get_dense(state)
            # Trim trailing zeros
            while len(best_poly) > 1 and best_poly[-1] == 0:
                best_poly.pop()
            if best_poly[-1] == 0:
                 best_poly[-1] = 1
            if len(best_poly) > 41:
                 best_poly = best_poly[:41]
                 while len(best_poly) > 1 and best_poly[-1] == 0:
                     best_poly.pop()
                 if best_poly[-1] == 0: best_poly[-1] = 1

        for step in range(steps_per_restart):
            if time.time() - start_time > max_time:
                break
                
            # Choose move
            if random.random() < 0.4:
                new_state = mutate_exponent(state)
            else:
                new_state = mutate_coefficient(state)
                
            new_score = eval_state(new_state)
            
            if new_score < current_score:
                # Accept
                state = new_state
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = get_dense(state)
                    while len(best_poly) > 1 and best_poly[-1] == 0:
                        best_poly.pop()
                    if best_poly[-1] == 0:
                        best_poly[-1] = 1
                    if len(best_poly) > 41:
                        best_poly = best_poly[:41]
                        while len(best_poly) > 1 and best_poly[-1] == 0:
                            best_poly.pop()
                        if best_poly[-1] == 0: best_poly[-1] = 1
            else:
                # Accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0:
                    # If score is inf, reject
                    if new_score == float('inf'):
                        continue
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        state = new_state
                        current_score = new_score
            
            T *= cooling_rate

    # Fallback
    if best_poly is None:
        best_poly = [0]*41
        for i in range(12):
            best_poly[i] = 1
            
    # Final cleanup
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    if best_poly[-1] == 0:
        best_poly[-1] = 1
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        if best_poly[-1] == 0: best_poly[-1] = 1
        
    return best_poly
# EVOLVE-BLOCK-END
