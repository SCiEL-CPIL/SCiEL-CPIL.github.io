"""Rewritten program using Simulated Annealing for sparse square minimization."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Constants
    MAX_DEG = 40
    NUM_TERMS = 12
    MAX_VAL = 64

    def evaluate(positions, values):
        """
        Evaluate the number of non-zero terms in the square of the polynomial.
        positions: list of 12 ints
        values: list of 12 ints
        Returns: count of non-zero coeffs in P^2
        """
        # Create dense representation
        dense = np.zeros(MAX_DEG + 1, dtype=np.int64)
        for p, v in zip(positions, values):
            if 0 <= p <= MAX_DEG:
                dense[p] = v
            else:
                # Invalid position, return large penalty
                return 1000
        
        # Convolve
        sq = np.convolve(dense, dense, mode='full')
        # Count non-zeros
        return int(np.count_nonzero(sq))

    def random_state():
        """Generate a random valid state."""
        positions = random.sample(range(MAX_DEG + 1), NUM_TERMS)
        # Values: bias towards small integers for better cancellation chances
        values = []
        for _ in range(NUM_TERMS):
            if random.random() < 0.5:
                values.append(random.choice([-1, 1]))
            else:
                values.append(random.randint(-MAX_VAL, MAX_VAL))
                if values[-1] == 0:
                    values[-1] = 1
        return positions, values

    def initial_structures():
        """Generate a few structural starting points."""
        states = []
        
        # 1. Alternating signs on AP
        for step in [1, 2, 3]:
            if (NUM_TERMS - 1) * step <= MAX_DEG:
                pos = list(range(0, NUM_TERMS * step, step))
                val = [1 if i % 2 == 0 else -1 for i in range(NUM_TERMS)]
                states.append((pos, val))
                
        # 2. Random
        for _ in range(20):
            states.append(random_state())
            
        return states

    def anneal(positions, values, time_limit):
        """
        Run Simulated Annealing on the given state.
        Returns best (positions, values, score) found during this run.
        """
        current_pos = positions[:]
        current_val = values[:]
        current_score = evaluate(current_pos, current_val)
        
        best_pos = current_pos[:]
        best_val = current_val[:]
        best_score = current_score
        
        start_t = time.time()
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995 # Adjust based on time budget
        
        while time.time() - start_t < time_limit:
            # Generate neighbor
            move_type = random.random()
            new_pos = current_pos[:]
            new_val = current_val[:]
            
            if move_type < 0.4:
                # Change one position
                idx = random.randint(0, NUM_TERMS - 1)
                new_pos[idx] = random.randint(0, MAX_DEG)
                # Ensure uniqueness
                if len(set(new_pos)) != NUM_TERMS:
                    continue
            elif move_type < 0.7:
                # Change one value
                idx = random.randint(0, NUM_TERMS - 1)
                # Try to pick a value that might cause cancellation
                # Often small values are better
                if random.random() < 0.7:
                    new_val[idx] = random.choice([-1, 1, -2, 2])
                else:
                    new_val[idx] = random.randint(-MAX_VAL, MAX_VAL)
                    if new_val[idx] == 0:
                        new_val[idx] = 1
            elif move_type < 0.8:
                # Swap two positions
                i, j = random.sample(range(NUM_TERMS), 2)
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
                if len(set(new_pos)) != NUM_TERMS:
                    continue
            elif move_type < 0.9:
                # Swap two values
                i, j = random.sample(range(NUM_TERMS), 2)
                new_val[i], new_val[j] = new_val[j], new_val[i]
            else:
                # Jitter position
                idx = random.randint(0, NUM_TERMS - 1)
                delta = random.choice([-1, 1])
                new_pos[idx] = current_pos[idx] + delta
                if 0 <= new_pos[idx] <= MAX_DEG and len(set(new_pos)) == NUM_TERMS:
                    pass
                else:
                    continue
            
            new_score = evaluate(new_pos, new_val)
            
            # Acceptance
            delta_e = new_score - current_score
            if delta_e <= 0:
                # Better or equal
                current_pos = new_pos
                current_val = new_val
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_pos = current_pos[:]
                    best_val = current_val[:]
            else:
                # Worse, accept with probability exp(-delta_e / T)
                # Since delta_e is integer >= 1, and T starts at 1.0
                prob = np.exp(-delta_e / T)
                if random.random() < prob:
                    current_pos = new_pos
                    current_val = new_val
                    current_score = new_score
            
            # Cool down
            T = max(T * cooling_rate, T_min)
            
            # If we hit the theoretical lower bound (unlikely to be 1, but maybe 12?), stop early
            if best_score <= 12:
                break
                
        return best_pos, best_val, best_score

    # Main loop: Multiple restarts
    global_best_score = 1000
    global_best_pos = None
    global_best_val = None
    
    # Initial structural candidates
    initial_states = initial_structures()
    
    # Time per restart
    # We have ~50s. Let's do restarts of ~2s each, allowing for ~25 restarts.
    # Or dynamic: run one long anneal, then restart if stuck?
    # Better: Run several independent anneals with different temps/seeds.
    
    num_restarts = 15
    time_per_restart = 50.0 / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        # Get initial state
        if i < len(initial_states):
            pos, val = initial_states[i]
        else:
            pos, val = random_state()
            
        # Run anneal
        b_pos, b_val, b_score = anneal(pos, val, time_per_restart)
        
        if b_score < global_best_score:
            global_best_score = b_score
            global_best_pos = b_pos
            global_best_val = b_val
            
        # If we find a very good solution, we might want to polish it further
        if global_best_score <= 14:
             # Spend remaining time polishing the best found so far
             remaining = deadline - time.time() - 1
             if remaining > 0:
                 b_pos, b_val, b_score = anneal(global_best_pos, global_best_val, remaining)
                 if b_score < global_best_score:
                     global_best_score = b_score
                     global_best_pos = b_pos
                     global_best_val = b_val
                 break

    # Fallback
    if global_best_pos is None:
        global_best_pos, global_best_val = random_state()
        global_best_score = evaluate(global_best_pos, global_best_val)

    # Construct final coefficient list
    # Ensure positions are sorted and valid
    # The problem asks for a list of coefficients in ascending degree.
    # We need to map (pos, val) to a list.
    
    # Check validity
    if len(set(global_best_pos)) != NUM_TERMS:
        # Fix if collision happened (shouldn't with our moves, but just in case)
        # This shouldn't happen as we check in moves
        pass
        
    max_pos = max(global_best_pos)
    coeff_list = [0] * (max_pos + 1)
    for p, v in zip(global_best_pos, global_best_val):
        coeff_list[p] = v
        
    # Ensure length <= 41
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
        
    # Ensure nonzero final coefficient
    # If the highest degree term in our support is not at the end of the list, 
    # we might have trailing zeros if max_pos < 40.
    # The requirement is "nonzero final coefficient".
    # If our polynomial has degree D < 40, the list length is D+1.
    # coeff_list[-1] is coeff at degree D, which is non-zero by definition of max_pos.
    # So this is automatically satisfied if we trim to max_pos.
    
    # However, if we want to be safe about "length<=41", we are fine.
    
    return coeff_list

# EVOLVE-BLOCK-END
