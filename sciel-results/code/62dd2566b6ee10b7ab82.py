import math
import time
import random
import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes):
    """Check if points form an admissible tuple."""
    k = len(points)
    if k == 0:
        return True
    for p in primes:
        if p > k:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _greedy_initial():
    """Generate initial solution using greedy heuristic."""
    points = [1] + [p for p in _primes(N) if p * p > N]
    changed = True
    while changed:
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 285
    
    # Get initial solution
    best_points = _greedy_initial()
    best_size = len(best_points)
    
    primes_list = _primes(N)
    all_numbers = list(range(N + 1))
    
    # Simulated annealing to improve
    current_points = best_points[:]
    current_size = best_size
    
    # Temperature schedule
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.9995
    
    T = T_start
    iteration = 0
    
    while time.time() < deadline and T > T_end:
        iteration += 1
        
        # Choose a move: add or remove a random element
        if random.random() < 0.5 and current_size < N:
            # Add a random element not in current set
            current_set = set(current_points)
            candidates = [x for x in all_numbers if x not in current_set]
            if not candidates:
                break
            new_point = random.choice(candidates)
            new_points = current_points + [new_point]
            move_type = 'add'
        else:
            # Remove a random element
            if current_size <= 1:
                T *= cooling_rate
                continue
            idx = random.randint(0, current_size - 1)
            new_points = current_points[:idx] + current_points[idx+1:]
            move_type = 'remove'
        
        # Check if new solution is admissible
        if _is_admissible(new_points, primes_list):
            new_size = len(new_points)
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if new_size >= current_size:
                current_points = new_points
                current_size = new_size
                if current_size > best_size:
                    best_points = current_points[:]
                    best_size = current_size
            else:
                # Delta is negative (worse)
                delta = new_size - current_size
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_points = new_points
                    current_size = new_size
        
        T *= cooling_rate
    
    # Final greedy cleanup to ensure admissibility
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for p in _primes(len(best_points)):
            classes = {}
            for v in best_points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                best_points = [v for v in best_points if v not in smallest]
                changed = True
                break
    
    return best_points
