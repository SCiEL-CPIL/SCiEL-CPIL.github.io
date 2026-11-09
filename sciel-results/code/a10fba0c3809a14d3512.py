import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Precompute LCM of 1..30 for exact rational arithmetic
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i

    # Initial solution: [0]*28 + [1, -1] gives sum 1/29 - 1/30 = 1/(29*30)
    # In integer terms: A[29] - A[30]
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

    # Local search with 2-flip neighborhood for better escape from local optima
    def local_search(c, current_sum):
        nonlocal best_val, best_abs, best_c
        improved = True
        # Limit iterations to prevent infinite loops in tight cases
        max_iters = 200
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1-flip moves
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
            
            # 2-flip moves (try pairs to escape local minima)
            if not improved and iter_count % 5 == 0:
                for i in range(30):
                    if time.time() - start_time > deadline:
                        return current_sum
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
                                new_sum = current_sum + (new_i - old_i) * A[i + 1] + (new_j - old_j) * A[j + 1]
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

    # Random restarts with local search
    random.seed(42)

    # First, do local search from the initial point
    c = best_c[:]
    local_search(c, get_sum(c))

    # Check time
    if time.time() - start_time > deadline:
        return best_c

    # Random restarts with diverse initialization strategies
    while time.time() - start_time < deadline:
        strategy = random.random()
        
        if strategy < 0.3:
            # Sparse random: mostly zeros with a few nonzeros
            num_nonzero = random.randint(2, 6)
            c = [0] * 30
            indices = random.sample(range(30), num_nonzero)
            for idx in indices:
                c[idx] = random.choice([-1, 1])
        elif strategy < 0.6:
            # Pairs strategy: create pairs that partially cancel
            c = [0] * 30
            for _ in range(random.randint(2, 4)):
                i, j = random.sample(range(30), 2)
                c[i] = random.choice([-1, 1])
                c[j] = -c[i]
            # Add some noise
            for idx in random.sample(range(30), random.randint(0, 3)):
                if c[idx] == 0:
                    c[idx] = random.choice([-1, 1])
        else:
            # Dense random
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        s = get_sum(c)
        if s == 0:
            continue
        
        # Do local search
        local_search(c, s)
        
        if time.time() - start_time > deadline:
            break

    return best_c
