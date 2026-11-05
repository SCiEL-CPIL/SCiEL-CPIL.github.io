import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    num_primes = len(primes)
    
    # Cache for scores to avoid recalculation
    # Key: tuple of (idx, exp) sorted by idx
    score_cache = {}
    
    def calc_score_from_state(state):
        """
        state: tuple of (idx, exp) sorted by idx
        """
        if not state:
            return 0, 1
            
        key = state
        if key in score_cache:
            return score_cache[key]
            
        n = 1
        divs = [1]
        valid = True
        
        for idx, exp in state:
            p = primes[idx]
            power = 1
            new_divs = []
            for _ in range(exp + 1):
                for d in divs:
                    val = d * power
                    new_divs.append(val)
                power *= p
            divs = new_divs
            n *= p ** exp
            
        if n > 10**18 or n < 2:
            score_cache[key] = (-1, n)
            return -1, n
            
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
                
        score_cache[key] = (score, n)
        return score, n

    def state_to_n(state):
        n = 1
        for idx, exp in state:
            n *= primes[idx] ** exp
        return n

    # Phase 1: Find a good square-free candidate
    # Reusing the DFS logic from the current program but simplified
    def find_best_square_free():
        best_score = 0
        best_state = ()
        
        # Try primorials of various lengths
        for k in range(5, 16):
            if k >= num_primes:
                break
            state = tuple((i, 1) for i in range(k))
            score, n = calc_score_from_state(state)
            if score > best_score:
                best_score = score
                best_state = state
                
        # Simple local search on square-free sets (swaps)
        current_state = best_state
        current_score = best_score
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            current_set = set(idx for idx, _ in current_state)
            
            # Try swapping one prime for another
            for i, idx in enumerate(current_state):
                for j in range(num_primes):
                    if j in current_set:
                        continue
                    new_state_list = list(current_state)
                    # Remove idx
                    new_state_list = [(a, b) for a, b in new_state_list if a != idx]
                    # Add j
                    new_state_list.append((j, 1))
                    new_state_list.sort(key=lambda x: x[0])
                    new_state = tuple(new_state_list)
                    
                    score, n = calc_score_from_state(new_state)
                    if score > current_score:
                        current_state = new_state
                        current_score = score
                        best_score = score
                        best_state = new_state
                        improved = True
                        break
                if improved:
                    break
                    
        return best_state, best_score

    init_state, init_score = find_best_square_free()
    
    # Phase 2: Local Search with Exponents
    # State is a list of (idx, exp)
    # We will use Hill Climbing / Simulated Annealing
    
    def local_search(initial_state, max_iters=5000):
        current_state = list(initial_state)
        current_score, _ = calc_score_from_state(tuple(current_state))
        best_state = current_state[:]
        best_score = current_score
        
        # Simulated Annealing parameters
        temp = 1.0
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Generate a move
            move_type = random.random()
            new_state = current_state[:]
            
            if move_type < 0.7:
                # Adjust exponent
                if not new_state:
                    continue
                idx_pos = random.randint(0, len(new_state) - 1)
                idx, exp = new_state[idx_pos]
                
                if random.random() < 0.5:
                    # Increase
                    new_exp = exp + 1
                    # Check product limit quickly
                    # n * p <= 10^18
                    # We don't have n here directly, but we can check later
                    new_state[idx_pos] = (idx, new_exp)
                else:
                    # Decrease
                    if exp > 1:
                        new_state[idx_pos] = (idx, exp - 1)
                    else:
                        # Remove
                        new_state.pop(idx_pos)
                        
            elif move_type < 0.85:
                # Add a prime
                # Find a prime not in state
                current_indices = set(idx for idx, _ in new_state)
                available = [j for j in range(num_primes) if j not in current_indices]
                if available:
                    j = random.choice(available)
                    new_state.append((j, 1))
                    new_state.sort(key=lambda x: x[0])
                    
            else:
                # Remove a prime
                if new_state:
                    idx_pos = random.randint(0, len(new_state) - 1)
                    idx, exp = new_state[idx_pos]
                    if exp == 1:
                        new_state.pop(idx_pos)
                    else:
                        new_state[idx_pos] = (idx, exp - 1)
                        
            new_state_tuple = tuple(new_state)
            new_score, n = calc_score_from_state(new_state_tuple)
            
            if new_score < 0:
                continue
                
            delta = new_score - current_score
            
            if delta > 0:
                current_state = new_state
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_state = new_state[:]
            else:
                # Accept with probability exp(delta / temp)
                # delta is negative
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_state = new_state
                    current_score = new_score
                    
            temp *= cooling
            
        return best_state, best_score

    # Run local search from the initial state
    best_state, best_score = local_search(init_state, max_iters=3000)
    
    # Try a few other starting points
    # 1. Primorial 10
    state_10 = tuple((i, 1) for i in range(10))
    s10, _ = calc_score_from_state(state_10)
    if s10 > 0:
        res_state, res_score = local_search(list(state_10), max_iters=1000)
        if res_score > best_score:
            best_score = res_score
            best_state = res_state
            
    # 2. Primorial 12
    state_12 = tuple((i, 1) for i in range(12))
    s12, _ = calc_score_from_state(state_12)
    if s12 > 0:
        res_state, res_score = local_search(list(state_12), max_iters=1000)
        if res_score > best_score:
            best_score = res_score
            best_state = res_state

    # 3. Random restarts
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Generate a random valid state
        random_state = []
        n = 1
        for i in range(num_primes):
            if n > 10**18:
                break
            # Random exponent 1 or 2
            exp = random.choice([1, 1, 2])
            p = primes[i]
            if n * (p ** exp) <= 10**18:
                n *= p ** exp
                random_state.append((i, exp))
            else:
                if exp > 1:
                    n *= p
                    random_state.append((i, 1))
                    
        if random_state:
            random_state.sort(key=lambda x: x[0])
            res_state, res_score = local_search(random_state, max_iters=500)
            if res_score > best_score:
                best_score = res_score
                best_state = res_state

    # Final refinement on the best state
    if best_state:
        res_state, res_score = local_search(best_state, max_iters=1000)
        if res_score > best_score:
            best_score = res_score
            best_state = res_state

    # Construct final n
    final_n = 1
    for idx, exp in best_state:
        final_n *= primes[idx] ** exp
        
    if final_n < 2:
        return 2
        
    return final_n

if __name__ == "__main__":
    n = construct()
    print(f"Constructed: {n}")
