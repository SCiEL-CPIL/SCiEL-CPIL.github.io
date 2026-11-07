"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 256
    universe = list(range(2, N + 1))
    
    def a_divides_bc(a, b, c):
        g = math.gcd(a, b)
        return c % (a // g) == 0
    
    def is_valid_to_add(S, a):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                if a_divides_bc(a, b, c):
                    return False
                if a_divides_bc(b, a, c):
                    return False
                if a_divides_bc(c, a, b):
                    return False
        return True
    
    # Greedy from largest to smallest
    S = set()
    for a in sorted(universe, reverse=True):
        if time.time() - start_time > DEADLINE - 10:
            break
        if is_valid_to_add(S, a):
            S.add(a)
    
    best_S = S.copy()
    best_size = len(S)
    
    # Local search: try adding removed elements
    removed = set(universe) - S
    improved = True
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        for a in sorted(removed, reverse=True):
            if time.time() - start_time > DEADLINE - 5:
                break
            if is_valid_to_add(S, a):
                S.add(a)
                removed.discard(a)
                improved = True
                if len(S) > best_size:
                    best_S = S.copy()
                    best_size = len(S)
    
    # Random restarts with perturbation
    random.seed(42)
    for _ in range(100):
        if time.time() - start_time > DEADLINE - 3:
            break
        S_trial = best_S.copy()
        n_remove = random.randint(1, min(3, len(S_trial)))
        for _ in range(n_remove):
            if S_trial:
                S_trial.discard(random.choice(sorted(S_trial)))
        removed_trial = set(universe) - S_trial
        for a in sorted(removed_trial, reverse=True):
            if is_valid_to_add(S_trial, a):
                S_trial.add(a)
        if len(S_trial) > best_size:
            best_S = S_trial
            best_size = len(S_trial)
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
