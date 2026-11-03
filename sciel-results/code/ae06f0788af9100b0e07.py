"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def score_set(U):
        U_set = set(U)
        max_U = max(U)
        if max_U == 0:
            return 0.0
        
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    # Start with a small known good set
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    current_set = best_set.copy()
    current_score = best_score

    # Simulated Annealing Parameters
    initial_temp = 0.1
    cooling_rate = 0.999
    min_temp = 0.001
    
    # Define a range for potential elements
    # The paper mentions sizes up to 54265, but we don't know the max value.
    # Let's assume elements are not too large initially, but can grow.
    # A safe upper bound for candidates can be dynamic.
    
    temp = initial_temp
    
    while temp > min_temp and time.time() < deadline:
        # Generate a neighbor
        # Strategy: 
        # 1. Add a random element
        # 2. Remove a random element (not 0)
        # 3. Replace a random element with a new one
        
        move_type = random.choice(['add', 'remove', 'replace'])
        
        if move_type == 'add':
            # Add a random element
            # Range: 1 to 2 * max(current) + 100
            max_val = max(current_set)
            candidate = random.randint(1, max_val * 2 + 100)
            if candidate not in current_set:
                new_set = current_set + [candidate]
            else:
                continue
        elif move_type == 'remove':
            # Remove a random element (not 0)
            removable = [x for x in current_set if x != 0]
            if not removable:
                continue
            elem_to_remove = random.choice(removable)
            new_set = [x for x in current_set if x != elem_to_remove]
        elif move_type == 'replace':
            # Replace a random element (not 0) with a new one
            removable = [x for x in current_set if x != 0]
            if not removable:
                continue
            elem_to_replace = random.choice(removable)
            max_val = max(current_set)
            candidate = random.randint(1, max_val * 2 + 100)
            if candidate in current_set and candidate != elem_to_replace:
                continue
            new_set = [x for x in current_set if x != elem_to_replace]
            new_set.append(candidate)
        else:
            continue
        
        # Ensure 0 is present (it should be, but just in case)
        if 0 not in new_set:
            new_set.append(0)
        
        # Sort for consistency
        new_set = sorted(new_set)
        
        # Calculate new score
        new_score = score_set(new_set)
        
        # Acceptance probability
        delta_score = new_score - current_score
        
        if delta_score >= 0:
            # Accept if better
            current_set = new_set
            current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()
        else:
            # Accept with probability exp(-delta_score / temp)
            # Note: delta_score is negative, so -delta_score is positive
            prob = math.exp(-delta_score / temp)
            if random.random() < prob:
                current_set = new_set
                current_score = new_score
        
        # Cool down
        temp *= cooling_rate
        
        # If we've been stuck for a while, maybe restart temperature?
        # Or just continue cooling.
        
    return best_set
# EVOLVE-BLOCK-END
