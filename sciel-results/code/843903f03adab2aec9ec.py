import time
import random
from math import gcd
from functools import reduce
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Precompute LCM of 1..30 for exact rational arithmetic
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
    
    # Initial solution: [0]*28 + [1, -1] gives sum 1/29 - 1/30 = 1/(29*30)
    best_val = A[29] - A[30]
    best_abs = abs(best_val)
    best_c = [0] * 30
    best_c[28] = 1
    best_c[29] = -1
    
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s
    
    # Fast local search with single and double flips
    def local_search(c, current_sum, max_iters=50):
        nonlocal best_val, best_abs, best_c
        improved = True
        iteration = 0
        
        while improved and iteration < max_iters:
            improved = False
            iteration += 1
            
            # Single-flip moves
            for i in range(30):
                old = c[i]
                for new_val in (-1, 0, 1):
                    if new_val == old:
                        continue
                    c[i] = new_val
                    new_sum = current_sum + (new_val - old) * A[i + 1]
                    if new_sum != 0:
                        new_abs = abs(new_sum)
                        if new_abs < best_abs:
                            best_abs = new_abs
                            best_val = new_sum
                            best_c = c[:]
                            current_sum = new_sum
                            improved = True
                    c[i] = old
            
            # Two-flip moves if no single flip improved
            if not improved:
                for i in range(30):
                    if time.time() - start_time > deadline - 2:
                        break
                    old_i = c[i]
                    for new_i in (-1, 0, 1):
                        if new_i == old_i:
                            continue
                        c[i] = new_i
                        for j in range(i + 1, 30):
                            old_j = c[j]
                            for new_j in (-1, 0, 1):
                                if new_j == old_j:
                                    continue
                                c[j] = new_j
                                new_sum = (current_sum + (new_i - old_i) * A[i + 1] + (new_j - old_j) * A[j + 1])
                                if new_sum != 0:
                                    new_abs = abs(new_sum)
                                    if new_abs < best_abs:
                                        best_abs = new_abs
                                        best_val = new_sum
                                        best_c = c[:]
                                        current_sum = new_sum
                                        improved = True
                                c[j] = old_j
                        c[i] = old_i
        
        return current_sum
    
    # Simulated annealing for global exploration
    def simulated_annealing(c, current_sum, t_start, t_end, steps):
        nonlocal best_val, best_abs, best_c
        
        if steps <= 0 or t_start <= t_end:
            return current_sum
        
        alpha = (t_end / t_start) ** (1.0 / steps)
        T = t_start
        
        rand_int = random.randint
        rand_choice = random.choice
        rand_float = random.random
        exp_func = math.exp
        
        for _ in range(steps):
            if time.time() - start_time > deadline:
                break
            
            i = rand_int(0, 29)
            old_val = c[i]
            new_val = rand_choice((-1, 0, 1))
            if new_val == old_val:
                T *= alpha
                continue
            
            c[i] = new_val
            new_sum = current_sum + (new_val - old_val) * A[i + 1]
            
            if new_sum == 0:
                c[i] = old_val
                T *= alpha
                continue
            
            new_abs = abs(new_sum)
            current_abs = abs(current_sum)
            
            delta = new_abs - current_abs
            
            if delta < 0:
                current_sum = new_sum
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_val = new_sum
                    best_c = c[:]
            else:
                if T > 0:
                    ratio = delta / T
                    if ratio < 500:
                        prob = exp_func(-ratio)
                    else:
                        prob = 0.0
                    if rand_float() < prob:
                        current_sum = new_sum
                    else:
                        c[i] = old_val
            
            T *= alpha
        
        return current_sum
    
    random.seed(42)
    
    # Phase 1: Local search from initial point
    c = best_c[:]
    local_search(c, get_sum(c), max_iters=100)
    
    # Phase 2: Structured candidate generation with SA refinement
    def generate_structured_candidate():
        strategy = random.random()
        
        if strategy < 0.25:
            # Pairs of adjacent terms with opposite signs
            c = [0] * 30
            num_pairs = random.randint(1, 6)
            used = set()
            for _ in range(num_pairs):
                i = random.randint(0, 28)
                if i not in used and (i+1) not in used:
                    c[i] = random.choice([1, -1])
                    c[i+1] = -c[i]
                    used.add(i)
                    used.add(i+1)
            if get_sum(c) == 0:
                idx = random.randint(0, 29)
                if c[idx] != 1:
                    c[idx] = 1
                else:
                    c[idx] = -1
            return c
        
        elif strategy < 0.45:
            # Sparse random
            c = [0] * 30
            num_nonzero = random.randint(2, 10)
            indices = random.sample(range(30), num_nonzero)
            for i in indices:
                c[i] = random.choice([-1, 1])
            if get_sum(c) == 0:
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 1])
            return c
        
        elif strategy < 0.65:
            # Dense random
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
            if get_sum(c) == 0:
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 1])
            return c
        
        elif strategy < 0.80:
            # Block structure
            c = [0] * 30
            start = random.randint(0, 24)
            len_pos = random.randint(1, 5)
            len_neg = random.randint(1, 5)
            if start + len_pos + len_neg <= 30:
                sign = random.choice([1, -1])
                for k in range(len_pos):
                    c[start + k] = sign
                for k in range(len_neg):
                    c[start + len_pos + k] = -sign
            if get_sum(c) == 0:
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 1])
            return c
        
        elif strategy < 0.90:
            # Alternating blocks
            c = [0] * 30
            pos = 0
            sign = random.choice([1, -1])
            while pos < 30 and random.random() < 0.7:
                block_len = random.randint(1, 4)
                for k in range(min(block_len, 30 - pos)):
                    c[pos + k] = sign
                pos += block_len
                sign = -sign
            if get_sum(c) == 0:
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 1])
            return c
        
        else:
            # Perturbation of best
            c = best_c[:]
            num_flips = random.randint(1, 4)
            indices = random.sample(range(30), num_flips)
            for i in indices:
                c[i] = random.choice([-1, 0, 1])
            if get_sum(c) == 0:
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 1])
            return c
    
    # Phase 3: Main search loop
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time <= 0:
        return best_c
    
    # Estimate how many SA steps we can afford
    # Each SA step is ~100ns, so ~10M steps in 1 second
    # We'll do SA for ~60% of remaining time, local search for rest
    
    sa_time_budget = remaining_time * 0.6
    restarts = 0
    
    while time.time() - start_time < deadline:
        restarts += 1
        
        # Generate structured candidate
        c = generate_structured_candidate()
        s = get_sum(c)
        if s == 0:
            continue
        
        # SA phase: moderate temperature for exploration
        # Scale temperature based on current best
        t_start = max(best_abs * 10, 1e6)
        t_end = max(best_abs * 0.1, 1.0)
        
        # Estimate steps: sa_time_budget / 100ns per step
        elapsed_so_far = time.time() - start_time
        time_left = deadline - elapsed_so_far
        if time_left <= 0:
            break
        
        # Do SA for a fraction of remaining time
        sa_steps = int(min(time_left * 0.3, 5e6))
        if sa_steps < 1000:
            sa_steps = 1000
        
        s = simulated_annealing(c, s, t_start, t_end, sa_steps)
        
        # Local search refinement
        if time.time() - start_time < deadline - 1:
            local_search(c, s, max_iters=30)
        
        # Periodic targeted search
        if restarts % 20 == 0:
            for i in range(1, 30):
                c = [0] * 30
                c[i-1] = 1
                c[i] = -1
                s = get_sum(c)
                if s != 0:
                    local_search(c, s, max_iters=20)
                if time.time() - start_time > deadline - 5:
                    break
    
    return best_c
