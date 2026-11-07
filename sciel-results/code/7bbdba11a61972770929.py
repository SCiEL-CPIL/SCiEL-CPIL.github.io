"""Optimized program: minimize interval length for distinct multiples of fixed divisors."""
import time
import math
import random

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = 10

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit

    # Strategy:
    # We are looking for n_1, ..., n_10 (integers >= 1) such that a_i = n_i * d_i are distinct and minimize max(a_i) - min(a_i).
    # Use Simulated Annealing with heuristic initialization and directed moves.

    random.seed(42)
    
    best_obj = float('inf')
    best_state = None

    def get_a(state):
        return [k * d for k, d in zip(state, DIVISORS)]
    
    def evaluate(a):
        # Returns objective value. High value for invalid states.
        if any(x < 1 or x > MAX_VAL for x in a):
            return 10**10
        if len(set(a)) != N:
            return 10**10
        return max(a) - min(a)

    def init_state():
        # Heuristic initialization: pick a random center C, set k_i ~ C/d_i
        # C should be large enough to allow distinctness but small enough to be interesting.
        # Optimal width is likely small. Let's guess width ~ 100-1000.
        # So C can be anywhere from 100 to 1000000.
        # Actually, small C is better for small width.
        # Let's pick C in [100, 10000]
        C = random.randint(100, 10000)
        state = []
        for d in DIVISORS:
            k = max(1, int(C / d))
            # Add some noise
            k += random.randint(-2, 2)
            if k < 1: k = 1
            state.append(k)
        return state

    state = init_state()
    a = get_a(state)
    obj = evaluate(a)
    best_obj = obj
    best_state = state[:]

    T = 50.0
    cooling = 0.995
    
    iters = 0
    no_improve_count = 0
    max_no_improve = 2000

    while time.time() < deadline:
        iters += 1
        
        # Determine move
        # 70% chance: directed move (shrink range)
        # 30% chance: random move
        if random.random() < 0.7:
            # Find min and max
            min_val = min(a)
            max_val = max(a)
            min_idx = a.index(min_val)
            max_idx = a.index(max_val)
            
            # Try to shrink max or grow min
            if random.random() < 0.5:
                # Shrink max
                idx = max_idx
                # Try decreasing k
                step = -random.randint(1, 5)
            else:
                # Grow min
                idx = min_idx
                # Try increasing k
                step = random.randint(1, 5)
        else:
            # Random move
            idx = random.randint(0, N - 1)
            step = random.randint(-5, 5)
            if step == 0:
                continue

        new_state = state[:]
        new_state[idx] += step
        if new_state[idx] < 1:
            new_state[idx] = 1
        if new_state[idx] * DIVISORS[idx] > MAX_VAL:
            continue
            
        new_a = get_a(new_state)
        new_obj = evaluate(new_a)
        
        # Accept if better, or with probability based on temperature
        if new_obj < obj or random.random() < math.exp((obj - new_obj) / max(T, 1e-9)):
            state = new_state
            a = new_a
            obj = new_obj
            if obj < best_obj:
                best_obj = obj
                best_state = state[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        T *= cooling
        if T < 0.01 or no_improve_count > max_no_improve:
            T = 50.0
            no_improve_count = 0
            # Restart with new heuristic init
            state = init_state()
            a = get_a(state)
            obj = evaluate(a)
            if obj < best_obj:
                best_obj = obj
                best_state = state[:]

    # Final polish: local search on best_state
    # Try all single-step changes for each element
    final_state = best_state[:]
    final_a = get_a(final_state)
    final_obj = evaluate(final_a)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            # Try k_i - 1
            if final_state[i] > 1:
                new_state = final_state[:]
                new_state[i] -= 1
                new_a = get_a(new_state)
                new_obj = evaluate(new_a)
                if new_obj < final_obj:
                    final_state = new_state
                    final_a = new_a
                    final_obj = new_obj
                    improved = True
                    break # Restart loop as indices changed
            # Try k_i + 1
            if final_state[i] * DIVISORS[i] < MAX_VAL:
                new_state = final_state[:]
                new_state[i] += 1
                new_a = get_a(new_state)
                new_obj = evaluate(new_a)
                if new_obj < final_obj:
                    final_state = new_state
                    final_a = new_a
                    final_obj = new_obj
                    improved = True
                    break # Restart loop as indices changed

    # Convert to Python ints
    result = [int(x) for x in get_a(final_state)]
    return result

# EVOLVE-BLOCK-END
