import random
import time

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def score(S):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a.
        """
        S_set = set(S)
        count = 0
        # d can be at most (512-1)/2 = 255
        for d in range(1, 256):
            found = False
            for a in S_set:
                # Check if a+d and a+2d are in S
                if a + d in S_set and a + 2*d in S_set:
                    found = True
                    break
            if found:
                count += 1
        return count

    # Initial solution: The seed program's output
    # It's a good dense start, but maybe not optimal.
    # We can also try a random start or a structured start.
    # Let's try a few initial candidates and pick the best one.
    
    candidates = []
    # 1. The seed
    candidates.append(list(range(1, 33)))
    
    # 2. A spread out set: 1, 2, ..., 16 and 257, ..., 272
    # This captures small d from both blocks and potentially large d across blocks.
    c2 = list(range(1, 17)) + list(range(257, 273))
    candidates.append(c2)
    
    # 3. Another spread: 1..16 and 129..144 (step 128)
    c3 = list(range(1, 17)) + list(range(129, 145))
    candidates.append(c3)
    
    # 4. Random
    random.seed(42)
    c4 = random.sample(range(1, 513), 32)
    candidates.append(c4)
    
    best_initial = max(candidates, key=score)
    best_score = score(best_initial)
    
    # Simulated Annealing
    current_set = best_initial[:]
    current_score = best_score
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for safety
    
    temperature = 5.0
    cooling = 0.9999
    min_temp = 0.01
    
    while time.time() - start_time < time_limit:
        # Pick a random index to modify
        idx = random.randint(0, 31)
        old_val = current_set[idx]
        
        # Generate a new value
        # Sometimes swap with another element to maintain density?
        # Let's just replace.
        new_val = random.randint(1, 512)
        
        # Ensure distinctness
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        
        new_score = score(current_set)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Accept with probability exp(delta / T)
            if temperature > min_temp:
                # Avoid underflow/overflow issues, delta is small integer
                prob = 2.71828 ** (delta / temperature)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    current_set[idx] = old_val
            else:
                # Revert
                current_set[idx] = old_val
        
        # Cool down
        temperature *= cooling
        
        # If temperature drops too low, reheat and restart from best
        if temperature < min_temp:
            current_set = best_set[:]
            current_score = best_score
            temperature = 5.0
            # Slight perturbation of best set to escape local optimum
            for _ in range(3):
                idx = random.randint(0, 31)
                current_set[idx] = random.randint(1, 512)
                # Ensure distinctness for the perturbation
                while current_set[idx] in current_set[:idx] + current_set[idx+1:]:
                    current_set[idx] = random.randint(1, 512)
            current_score = score(current_set)

    return best_set
