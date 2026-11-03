import math
import random
import time

def score_set(U):
    """
    Calculates the score for a given set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2 * max(U) + 1)
    """
    U = set(U)
    if not U:
        return 1.0
    
    # Calculate sum set and difference set
    # Using set comprehensions
    sums = {a + b for a in U for b in U}
    diffs = {a - b for a in U for b in U}
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    ratio = len(diffs) / len(sums)
    
    # If ratio <= 1, log is <= 0, score <= 1.
    # We want to maximize, so we return 1.0 as baseline if not better.
    if ratio <= 1:
        return 1.0
        
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety
    deadline = start_time + 110
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Include some known good candidates to ensure we start strong
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15]
    ]
    
    for c in initial_candidates:
        # Ensure 0 is in set and sorted
        c = sorted(set(c))
        if 0 not in c:
            c.append(0)
            c.sort()
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Search for better sets
    # Try different set sizes
    for n in range(3, 9):
        if time.time() > deadline:
            break
            
        # Multiple random restarts for each size
        for trial in range(15):
            if time.time() > deadline:
                break
                
            # Generate a random initial set
            # Heuristic: max element around n * 4
            max_bound = n * 4
            while True:
                try:
                    # Pick n-1 distinct random numbers from 1..max_bound
                    rest = random.sample(range(1, max_bound + 1), n - 1)
                    U = [0] + rest
                    U.sort()
                    if len(set(U)) == n:
                        break
                except ValueError:
                    # Not enough numbers in range, increase bound
                    max_bound += 10
                    continue
            
            current_score = score_set(U)
            if current_score > best_score:
                best_score = current_score
                best_set = U
                
            # Local search (Hill Climbing)
            steps = 0
            max_steps = 2000
            # Adaptive step size range
            step_range = 5
            
            while steps < max_steps and time.time() < deadline:
                steps += 1
                
                # Pick a random index to modify (excluding 0)
                i = random.randint(1, n - 1)
                
                # Propose a change
                step = random.randint(-step_range, step_range)
                if step == 0:
                    continue
                    
                new_val = U[i] + step
                
                # Constraints: non-negative, not 0 (since 0 is at index 0), distinct
                if new_val < 0 or new_val == 0:
                    continue
                    
                # Check if new_val is already in the set
                # Since U is small, 'in' is fast
                if new_val in U:
                    continue
                    
                # Create new set
                new_U = U.copy()
                new_U[i] = new_val
                new_U.sort()
                
                # Evaluate
                new_score = score_set(new_U)
                
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = U
                    # Reset step_range or keep it?
                    # Maybe reduce step_range slightly to refine?
                    # step_range = max(1, step_range - 1)
                    
    return best_set
