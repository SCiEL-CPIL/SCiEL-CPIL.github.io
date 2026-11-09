import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 45.0  # Safety margin before 90s kill

# Precompute conflict bitsets: conflict[x] has bits set for all y such that x,y,z form a 3-AP for some z
conflict = [0] * (N + 1)

def precompute_conflicts():
    """Precompute conflict bitsets for all pairs that form 3-APs."""
    for d in range(1, (N + 1) // 2):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # a, b, c form a 3-AP
            conflict[a] |= (1 << (b - 1)) | (1 << (c - 1))
            conflict[b] |= (1 << (a - 1)) | (1 << (c - 1))
            conflict[c] |= (1 << (a - 1)) | (1 << (b - 1))

precompute_conflicts()

def can_add_fast(bitset, x):
    """
    Check if adding x to the set creates a 3-AP using precomputed conflicts.
    Returns True if x can be added without creating a 3-AP.
    """
    return (bitset & conflict[x]) == 0

def greedy_construct(order):
    """
    Greedy construction: add elements in the given order if they don't create a 3-AP.
    Returns (bitset, score) where score is the exact sum of reciprocals as a Fraction.
    """
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add_fast(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

def get_elements_fast(bitset):
    """Get list of elements in bitset efficiently."""
    return [i for i in range(1, N + 1) if bitset & (1 << (i - 1))]

def local_search(bitset, score, deadline):
    """
    Local search to improve the solution using bitset-based conflict checking.
    Returns (best_bitset, best_score) as (int, Fraction).
    """
    best_bitset = bitset
    best_score = score
    current_bitset = bitset
    current_score = score
    
    recip = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = Fraction(1, i)
    
    float_recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        float_recip[i] = 1.0 / i
    
    # Phase 1: Add all elements that can be added
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (current_bitset & (1 << (x - 1))):
                if can_add_fast(current_bitset, x):
                    current_bitset |= (1 << (x - 1))
                    current_score += recip[x]
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset
    
    # Phase 2: Simulated annealing
    current_temp = 0.05
    cooling_rate = 0.9998
    min_temp = 1e-12
    no_improve_count = 0
    max_no_improve = 3000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            current_bitset = best_bitset
            current_score = best_score
            current_temp = max(current_temp * 3, 0.1)
            no_improve_count = 0
        
        move_type = random.random()
        
        if move_type < 0.4:
            not_in_set = [x for x in range(1, N + 1) if not (current_bitset & (1 << (x - 1)))]
            if not_in_set:
                x_add = random.choice(not_in_set)
                if can_add_fast(current_bitset, x_add):
                    new_bitset = current_bitset | (1 << (x_add - 1))
                    delta = float_recip[x_add]
                    if delta > 0:
                        current_bitset = new_bitset
                        current_score += recip[x_add]
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        if random.random() < math.exp(delta / current_temp):
                            current_bitset = new_bitset
                            current_score += recip[x_add]
                            no_improve_count += 1
                        else:
                            no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        elif move_type < 0.6:
            in_set = get_elements_fast(current_bitset)
            if in_set:
                x_rem = random.choice(in_set)
                new_bitset = current_bitset & ~(1 << (x_rem - 1))
                delta = -float_recip[x_rem]
                if random.random() < math.exp(delta / current_temp):
                    current_bitset = new_bitset
                    current_score -= recip[x_rem]
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        else:
            in_set = get_elements_fast(current_bitset)
            not_in_set = [x for x in range(1, N + 1) if not (current_bitset & (1 << (x - 1)))]
            if in_set and not_in_set:
                x_rem = random.choice(in_set)
                x_add = random.choice(not_in_set)
                
                trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                if can_add_fast(trial_bitset, x_add):
                    trial_bitset |= (1 << (x_add - 1))
                    delta = float_recip[x_add] - float_recip[x_rem]
                    
                    if delta > 0:
                        current_bitset = trial_bitset
                        current_score += recip[x_add] - recip[x_rem]
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        if random.random() < math.exp(delta / current_temp):
                            current_bitset = trial_bitset
                            current_score += recip[x_add] - recip[x_rem]
                            no_improve_count += 1
                        else:
                            no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp
    
    return best_bitset, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    strategies = []
    
    strategies.append(list(range(1, N + 1)))
    
    for trial in range(20):
        order = list(range(1, N + 1))
        for i in range(len(order) - 1, 0, -1):
            if random.random() < 0.3:
                j = max(0, i - random.randint(1, 5))
                order[i], order[j] = order[j], order[i]
        strategies.append(order)
    
    strategies.append(list(range(N, 0, -1)))
    
    for trial in range(20):
        order = []
        remaining = set(range(1, N + 1))
        while remaining:
            sorted_remaining = sorted(remaining)
            k = max(1, len(sorted_remaining) // 5)
            candidates = sorted_remaining[:k]
            x = random.choice(candidates)
            order.append(x)
            remaining.remove(x)
        strategies.append(order)
    
    best_bitset = 0
    best_score = Fraction(0, 1)
    
    for order in strategies:
        if time.time() > deadline - 15:
            break
        bitset, score = greedy_construct(order)
        if score > best_score:
            best_score = score
            best_bitset = bitset
    
    best_bitset, best_score = local_search(best_bitset, best_score, deadline)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add_fast(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    improved = True
    
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
