import time
import random

# Global variables for the backtracking solver
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0 # Leave 5s margin for 120s limit (killed at 150s, but evaluator says 120s finish)
# The prompt says "Each evaluation must finish within 120 s." and "evaluator KILLS ... after 150s".
# So 115s is a safe internal limit.

def can_add(x, current_set, current_sums):
    # Check if x can be added to current_set
    # Sums to check: x + y for y in current_set, and 2*x
    for y in current_set:
        s = x + y
        if s in current_sums:
            return False
    if 2 * x in current_sums:
        return False
    return True

def add_element(x, current_set, current_sums):
    for y in current_set:
        current_sums.add(x + y)
    current_sums.add(2 * x)
    current_set.append(x)

def remove_element(x, current_set, current_sums):
    # x is the last element added
    for y in current_set:
        if y != x:
            current_sums.discard(x + y)
    current_sums.discard(2 * x)
    current_set.pop()

def dfs(current_set, current_sums, start_val, target_size, rng):
    global best_set, best_size
    
    if time.time() - start_time > time_limit:
        return False
    
    current_size = len(current_set)
    if current_size > best_size:
        best_size = current_size
        best_set = current_set[:]
        
    if current_size == target_size:
        return True
        
    # Generate valid candidates
    valid_candidates = []
    for x in range(start_val, 285):
        if can_add(x, current_set, current_sums):
            valid_candidates.append(x)
            
    if not valid_candidates:
        return False
        
    # Heuristic: Randomize the order of valid candidates to allow different search paths
    # Or sort by some metric? Random shuffle is good for diversification in restarts.
    rng.shuffle(valid_candidates)
    
    # Optional: Try candidates that are further apart first?
    # For now, random shuffle is a good diversification strategy.
    
    for x in valid_candidates:
        if time.time() - start_time > time_limit:
            return False
            
        add_element(x, current_set, current_sums)
        if dfs(current_set, current_sums, x + 1, target_size, rng):
            return True
        remove_element(x, current_set, current_sums)
        
    return False

def construct():
    global start_time, best_set, best_size
    
    start_time = time.time()
    
    # Initialize with prescribed elements
    initial_set = [1, 2, 5, 11]
    initial_sums = set()
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            initial_sums.add(initial_set[i] + initial_set[j])
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    target_size = 20
    
    # Run multiple randomized DFS attempts
    # Each attempt uses a different random seed for shuffling candidates
    attempts = 0
    while time.time() - start_time < time_limit - 2:
        attempts += 1
        rng = random.Random(attempts)
        # We need to pass rng to dfs
        dfs(initial_set[:], initial_sums.copy(), 12, target_size, rng)
        
        if best_size == target_size:
            break
            
    return best_set
