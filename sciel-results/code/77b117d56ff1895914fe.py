"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def compute_score(U):
        """
        Compute the score for a set U.
        score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U or 0 not in U:
            return -1.0
        
        U_list = sorted(list(set(U)))
        if U_list[0] != 0:
            return -1.0
            
        n = len(U_list)
        if n < 2:
            return 0.0
            
        max_u = U_list[-1]
        if max_u == 0:
            return 0.0
            
        # Compute |U-U|
        diffs = set()
        for i in range(n):
            for j in range(n):
                diffs.add(U_list[i] - U_list[j])
        n_diff = len(diffs)
        
        # Compute |U+U|
        sums = set()
        for i in range(n):
            for j in range(n):
                sums.add(U_list[i] + U_list[j])
        n_sum = len(sums)
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def is_valid(U):
        """Check if U is a valid set of non-negative integers containing 0."""
        if not U:
            return False
        if 0 not in U:
            return False
        for x in U:
            if x < 0:
                return False
        return True

    # Initial seeds: Mix of small Sidon sets and random sets
    seeds = []
    
    # Known small Sidon sets (Golomb rulers)
    sidon_seeds = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 42],
        [0, 1, 4, 10, 18, 23, 31, 42, 52],
        [0, 1, 4, 10, 18, 23, 31, 42, 52, 62],
    ]
    seeds.extend(sidon_seeds)
    
    # Random sets of various sizes
    random.seed(42)
    for _ in range(10):
        size = random.randint(3, 15)
        max_val = random.randint(10, 100)
        # Ensure 0 is included
        s = {0}
        while len(s) < size:
            s.add(random.randint(1, max_val))
        seeds.append(list(s))
    
    best_set = seeds[0]
    best_score = compute_score(best_set)
    
    # Simulated Annealing
    # We will run multiple restarts with different initial states
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Pick a random seed
        current_set = set(random.choice(seeds))
        current_score = compute_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
        # Annealing parameters
        T = 1.0
        cooling_rate = 0.9995
        min_T = 1e-6
        max_iters = 5000
        
        # Pre-compute max possible span for moves
        current_max = max(current_set) if current_set else 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            if T < min_T:
                break
                
            n = len(current_set)
            if n == 0:
                break
                
            # Choose a move type
            move_type = random.random()
            
            new_set = set(current_set)
            new_score = current_score
            
            if move_type < 0.2:
                # Insert a new element
                # Range for new element: 1 to current_max * 2 or current_max + 10
                max_insert = max(current_max * 2, current_max + 10, 10)
                x = random.randint(1, max_insert)
                if x not in new_set:
                    new_set.add(x)
                    new_score = compute_score(new_set)
                    
            elif move_type < 0.35:
                # Delete an element (not 0)
                if n > 1:
                    candidates = [x for x in current_set if x != 0]
                    if candidates:
                        x = random.choice(candidates)
                        new_set.remove(x)
                        new_score = compute_score(new_set)
                        
            elif move_type < 0.75:
                # Shift an element
                if n > 1:
                    candidates = [x for x in current_set if x != 0]
                    if candidates:
                        x = random.choice(candidates)
                        # Small shift
                        delta = random.randint(-5, 5)
                        if delta == 0:
                            continue
                        new_x = x + delta
                        if new_x >= 0 and new_x not in new_set:
                            new_set.remove(x)
                            new_set.add(new_x)
                            new_score = compute_score(new_set)
                            
            else:
                # Jump an element to a new random position
                if n > 1:
                    candidates = [x for x in current_set if x != 0]
                    if candidates:
                        x = random.choice(candidates)
                        # Jump within a range relative to current max
                        # Allow expanding the span slightly
                        jump_max = max(current_max * 1.5, 50)
                        new_x = random.randint(0, jump_max)
                        if new_x != x and new_x not in new_set:
                            new_set.remove(x)
                            new_set.add(new_x)
                            new_score = compute_score(new_set)
            
            # Acceptance criterion
            delta_score = new_score - current_score
            if delta_score > 0:
                # Always accept improvement
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Accept with probability exp(delta / T)
                # Note: delta_score is negative
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
            
            T *= cooling_rate

    return best_set

# EVOLVE-BLOCK-END
