import time
import random
import math

# Maximum time allowed for the function (evaluator kills at 90s, we want safety)
TIME_LIMIT = 45.0
N = 256

def is_3ap_free_fast(bitset):
    """
    Check if the set represented by bitset contains no non-constant 3-term arithmetic progression.
    bitset: integer where bit i is set if (i+1) is in the set.
    """
    # Get list of elements
    elements = []
    for i in range(1, N + 1):
        if bitset & (1 << (i - 1)):
            elements.append(i)
    
    if len(elements) < 3:
        return True
    
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            # Case 1: c = 2b - a, need c > b and c in set
            c = 2 * b - a
            if c > b and c <= N:
                if bitset & (1 << (c - 1)):
                    return False
            # Case 2: c = 2a - b, need c < a and c in set
            c = 2 * a - b
            if c < a and c >= 1:
                if bitset & (1 << (c - 1)):
                    return False
    return True

def can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset creates a 3-AP.
    Returns True if x can be added without creating a 3-AP.
    """
    # Check all pairs (a, b) in current set where adding x would create a 3-AP
    # 3-AP involving x: (x, m, n) or (n, m, x) or (m, x, n) where m is middle
    # Case 1: x is first term: x, m, n => m = x + d, n = x + 2d for some d >= 1
    #   Need m, n in set
    # Case 2: x is middle: m, x, n => m = x - d, n = x + d for some d >= 1
    #   Need m, n in set
    # Case 3: x is last term: m, n, x => n = x - d, m = x - 2d for some d >= 1
    #   Need m, n in set
    
    # More efficient: for each d, check the three cases
    # d can range from 1 to N-1
    
    # Case 2: x is middle. m = x-d, n = x+d
    for d in range(1, min(x, N - x + 1)):
        m = x - d
        n = x + d
        if m >= 1 and n <= N:
            if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
                return False
    
    # Case 1: x is first. m = x+d, n = x+2d
    for d in range(1, (N - x) // 2 + 1):
        m = x + d
        n = x + 2 * d
        if n <= N:
            if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
                return False
    
    # Case 3: x is last. m = x-2d, n = x-d
    for d in range(1, x // 2 + 1):
        m = x - 2 * d
        n = x - d
        if m >= 1:
            if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
                return False
    
    return True

def construct():
    start_time = time.time()
    
    # Reciprocals for numbers 1 to 256
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    # Initial solution: Greedy adding in order of decreasing reciprocal (i.e., increasing number)
    # This prioritizes small numbers which have larger reciprocals
    current_bitset = 0
    current_score = 0.0
    
    # Try adding numbers in order of decreasing reciprocal value (smallest numbers first)
    for x in range(1, N + 1):
        if can_add(current_bitset, x):
            current_bitset |= (1 << (x - 1))
            current_score += recip[x]
    
    # Track best solution found
    best_bitset = current_bitset
    best_score = current_score
    
    # Simulated Annealing
    current_temp = 0.01
    cooling_rate = 0.999
    min_temp = 1e-8
    step_count = 0
    
    # Precompute for faster swaps
    while time.time() - start_time < TIME_LIMIT:
        if current_temp < min_temp:
            # Restart with higher temperature if we're stuck
            current_temp = 0.005
            # Random restart from best
            current_bitset = best_bitset
            current_score = best_score
        
        step_count += 1
        move_type = random.random()
        
        if move_type < 0.4:
            # Try adding a random number not in set
            # Find available numbers
            available = []
            for x in range(1, N + 1):
                if not (current_bitset & (1 << (x - 1))):
                    available.append(x)
            
            if available:
                x_add = random.choice(available)
                if can_add(current_bitset, x_add):
                    delta_score = recip[x_add]
                    # Always accept improvement
                    if delta_score > 0:
                        current_bitset |= (1 << (x_add - 1))
                        current_score += delta_score
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                    else:
                        # Accept worse with prob exp(delta/T)
                        if random.random() < math.exp(delta_score / current_temp):
                            current_bitset |= (1 << (x_add - 1))
                            current_score += delta_score
        
        elif move_type < 0.7:
            # Try removing a random number from set
            in_set = []
            for x in range(1, N + 1):
                if current_bitset & (1 << (x - 1)):
                    in_set.append(x)
            
            if in_set:
                x_rem = random.choice(in_set)
                delta_score = -recip[x_rem]
                # Always accept if it helps escape local optima, or with prob exp(delta/T)
                if random.random() < math.exp(delta_score / current_temp):
                    current_bitset &= ~(1 << (x_rem - 1))
                    current_score += delta_score
        
        else:
            # Swap: remove one, add another
            in_set = []
            for x in range(1, N + 1):
                if current_bitset & (1 << (x - 1)):
                    in_set.append(x)
            
            not_in_set = []
            for x in range(1, N + 1):
                if not (current_bitset & (1 << (x - 1))):
                    not_in_set.append(x)
            
            if in_set and not_in_set:
                x_rem = random.choice(in_set)
                x_add = random.choice(not_in_set)
                
                # Try the swap
                trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                if can_add(trial_bitset, x_add):
                    trial_bitset |= (1 << (x_add - 1))
                    delta_score = recip[x_add] - recip[x_rem]
                    
                    if delta_score > 0:
                        current_bitset = trial_bitset
                        current_score += delta_score
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                    else:
                        if random.random() < math.exp(delta_score / current_temp):
                            current_bitset = trial_bitset
                            current_score += delta_score
        
        current_temp *= cooling_rate
        
        # Periodic local optimization: try to add any number that can be added
        if step_count % 1000 == 0:
            improved = True
            while improved:
                improved = False
                for x in range(1, N + 1):
                    if not (current_bitset & (1 << (x - 1))):
                        if can_add(current_bitset, x):
                            current_bitset |= (1 << (x - 1))
                            current_score += recip[x]
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_bitset = current_bitset
    
    # Final polish: try to add any remaining numbers that can be added
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    improved = True
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
