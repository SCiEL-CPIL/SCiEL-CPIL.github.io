import random
import time
import math

def calc_max_disc(signs, n=256, limit=None):
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p: min_p = curr
                if curr > max_p: max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc
    return max_disc

def greedy_repair(signs, n=256, deadline=None):
    """Greedy repair: repeatedly flip the sign that most reduces max discrepancy."""
    current_disc = calc_max_disc(signs)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        best_gain = 0
        best_idx = -1
        # Try flipping each position and see if it reduces the discrepancy
        for i in range(n):
            signs[i] *= -1
            new_disc = calc_max_disc(signs, n, limit=current_disc)
            signs[i] *= -1
            gain = current_disc - new_disc
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0 and best_gain > 0:
            signs[best_idx] *= -1
            current_disc -= best_gain
            improved = True
    return current_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Initial solution
    signs = [random.choice((-1, 1)) for _ in range(n)]
    current_disc = calc_max_disc(signs)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Iterated Local Search with Simulated Annealing
    # Each iteration: run SA, then perturb, repeat
    sa_iterations = 5000
    perturbation_strength = 5  # number of random flips for perturbation
    
    while time.time() < deadline:
        # Simulated Annealing phase
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        for _ in range(sa_iterations):
            if time.time() >= deadline:
                break
            # Pick a random index to flip
            i = random.randint(0, n - 1)
            signs[i] *= -1
            
            # Calculate new discrepancy with a limit to speed up rejection
            limit = current_disc + int(T * 5) + 1
            new_disc = calc_max_disc(signs, n, limit)
            
            delta = new_disc - current_disc
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_disc = new_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                # Reject
                signs[i] *= -1
                
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Greedy repair on current state
        current_disc = greedy_repair(signs, n, deadline)
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
        
        # Perturbation: flip a few random signs to escape local minimum
        for _ in range(perturbation_strength):
            i = random.randint(0, n - 1)
            signs[i] *= -1
        current_disc = calc_max_disc(signs)
    
    # Final polish: tight SA on the best solution
    if time.time() < deadline - 5:
        signs = best_signs[:]
        current_disc = best_disc
        T = 1.0
        T_min = 0.001
        cooling = 0.9999
        remaining = int((deadline - time.time()) / 0.001)
        for _ in range(min(remaining, 10000)):
            if time.time() >= deadline:
                break
            i = random.randint(0, n - 1)
            signs[i] *= -1
            new_disc = calc_max_disc(signs, n, limit=current_disc)
            delta = new_disc - current_disc
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_disc = new_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                signs[i] *= -1
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_signs
