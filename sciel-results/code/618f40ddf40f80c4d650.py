"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score_set_bitset(s):
    """
    Fast scoring using bitset operations.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in s.
    """
    if not s:
        return 0
    s_set = set(s)
    max_val = max(s_set)
    # Create a bitset where bit i is 1 if i in s
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    count = 0
    # d can range from 1 to max_val // 2
    # a can range from 1 to max_val - 2d
    # We check if there is ANY a such that a, a+d, a+2d are in s
    # (bitset & (bitset >> d) & (bitset >> (2*d))) != 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
            
    return count

def _construct_greedy():
    """
    Construct a good initial set using a greedy heuristic.
    """
    current_set = set()
    sat_diffs = set()
    
    for step in range(32):
        best_elem = None
        best_new_diffs_count = -1
        
        for elem in range(1, 513):
            if elem in current_set:
                continue
            
            new_diffs = 0
            
            # Case 1: elem is start
            max_d_start = (512 - elem) // 2
            for d in range(1, max_d_start + 1):
                if d in sat_diffs:
                    continue
                if (elem + d) in current_set and (elem + 2 * d) in current_set:
                    new_diffs += 1
            
            # Case 2: elem is middle
            max_d_mid = min(elem - 1, 512 - elem)
            for d in range(1, max_d_mid + 1):
                if d in sat_diffs:
                    continue
                if (elem - d) in current_set and (elem + d) in current_set:
                    new_diffs += 1
            
            # Case 3: elem is end
            max_d_end = elem // 2
            for d in range(1, max_d_end + 1):
                if d in sat_diffs:
                    continue
                if (elem - 2 * d) in current_set and (elem - d) in current_set:
                    new_diffs += 1
            
            if new_diffs > best_new_diffs_count:
                best_new_diffs_count = new_diffs
                best_elem = elem
        
        if best_elem is not None:
            current_set.add(best_elem)
            # Update sat_diffs
            bitset = 0
            for x in current_set:
                bitset |= (1 << x)
            new_sat = set()
            max_val = max(current_set)
            for d in range(1, max_val // 2 + 1):
                if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                    new_sat.add(d)
            sat_diffs = new_sat
        else:
            available = [x for x in range(1, 513) if x not in current_set]
            if available:
                best_elem = random.choice(available)
                current_set.add(best_elem)
                bitset = 0
                for x in current_set:
                    bitset |= (1 << x)
                new_sat = set()
                max_val = max(current_set)
                for d in range(1, max_val // 2 + 1):
                    if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                        new_sat.add(d)
                sat_diffs = new_sat

    return list(current_set)

def _simulated_annealing(initial_set, deadline):
    """
    Optimize the set using Simulated Annealing.
    """
    current_set = set(initial_set)
    current_score = _score_set_bitset(current_set)
    best_set = list(current_set)
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    all_elements = list(range(1, 513))
    
    while time.time() < deadline and T > T_min:
        # Generate a move: swap a random element in current_set with a random element outside
        if len(current_set) < 32:
            break
            
        elem_in = random.choice(list(current_set))
        elem_out = random.choice([x for x in all_elements if x not in current_set])
        
        # Perform swap
        new_set = (current_set - {elem_in}) | {elem_out}
        new_score = _score_set_bitset(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        
        T *= cooling_rate
        
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin
    
    # 1. Construct initial set using Greedy heuristic
    initial_set = _construct_greedy()
    
    # 2. Optimize using Simulated Annealing
    optimized_set, score = _simulated_annealing(initial_set, deadline)
    
    # 3. If time remains, try a few restarts from the best found
    best_set = optimized_set
    best_score = score
    
    # Try a few random perturbations/restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        perturbed = set(best_set)
        # Remove 3 random elements
        for _ in range(3):
            if perturbed:
                to_remove = random.choice(list(perturbed))
                perturbed.remove(to_remove)
        
        # Add 3 random elements
        while len(perturbed) < 32:
            new_elem = random.randint(1, 512)
            if new_elem not in perturbed:
                perturbed.add(new_elem)
        
        # Optimize perturbed set
        opt_set, opt_score = _simulated_annealing(list(perturbed), deadline)
        
        if opt_score > best_score:
            best_score = opt_score
            best_set = opt_set
    
    return best_set

# EVOLVE-BLOCK-END
