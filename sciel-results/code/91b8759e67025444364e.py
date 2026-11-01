import time
import math
import random

def calculate_score(U):
    """
    Calculate the raw score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    if not U:
        return 0.0
    if max(U) == 0:
        return 1.0 # log(1/1)/log(1) is undefined, but limit is 1? 
        # Actually if U={0}, U+U={0}, U-U={0}. Ratio 1. Log 1 = 0.
        # Denom log(1) = 0. 0/0. 
        # But task says nonnegative integers containing 0.
        # If max U = 0, U={0}. 
        # Let's assume U has at least one non-zero element for the formula to make sense in context of growth.
        # Or handle 0/0 as 0 contribution.
        return 1.0
    
    U_list = list(U)
    max_u = max(U_list)
    denom = math.log(2 * max_u + 1)
    
    # Calculate U+U
    sum_set = set()
    for a in U_list:
        for b in U_list:
            sum_set.add(a + b)
    
    # Calculate U-U
    diff_set = set()
    for a in U_list:
        for b in U_list:
            diff_set.add(a - b)
    
    num = math.log(len(diff_set) / len(sum_set))
    
    return 1.0 + num / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 4, 7],
        [0, 1, 2, 5, 6],
        [0, 1, 4, 6, 7],
    ]
    
    best_set = candidates[0]
    best_score = calculate_score(set(best_set))
    
    # Local search
    for _ in range(10000): # Max iterations
        if time.time() > deadline:
            break
        
        # Pick a random candidate to explore, or the current best?
        # Let's explore the current best primarily, but also keep a pool of good ones.
        # Simple hill climbing on the best set found so far.
        current = set(best_set)
        current_score = best_score
        
        improved = False
        # Try adding an element
        max_u = max(current)
        # Search range for new elements
        # We can try up to max_u + 10 or so
        for x in range(1, max_u + 11):
            if x in current:
                continue
            new_set = current | {x}
            s = calculate_score(new_set)
            if s > current_score:
                current_score = s
                current = new_set
                improved = True
        
        # Try removing an element
        for y in list(current):
            if y == 0:
                continue
            new_set = current - {y}
            if not new_set:
                continue
            s = calculate_score(new_set)
            if s > current_score:
                current_score = s
                current = new_set
                improved = True
        
        # Try replacing an element
        for y in list(current):
            if y == 0:
                continue
            for x in range(1, max(current) + 11):
                if x in current:
                    continue
                new_set = (current - {y}) | {x}
                s = calculate_score(new_set)
                if s > current_score:
                    current_score = s
                    current = new_set
                    improved = True
                    break # Restart replacement loop for this y? Or just update and break out of x loop?
            
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
        else:
            # If no improvement, try a random perturbation to escape local minimum
            # Or pick a different seed
            # For simplicity, let's just try a random new seed from a small pool
            # Or random restart
            pass
            
    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Better initial seeds based on analysis
    # The set {0, 1, 3, 7} had score ~1.096
    # Let's try to find sets with higher ratio.
    
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 4, 6, 7],
        [0, 1, 2, 5, 7],
        [0, 1, 3, 4, 7, 8],
    ]
    
    best_set = candidates[0]
    best_score = calculate_score(set(best_set))
    
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        current = set(best_set)
        current_score = best_score
        improved = False
        
        # 1. Try adding
        max_u = max(current)
        # Limit search space for new elements to keep max_u small
        # If we add large elements, denom grows.
        # Try elements up to max_u + 5
        for x in range(1, max_u + 6):
            if x in current:
                continue
            new_set = current | {x}
            s = calculate_score(new_set)
            if s > current_score:
                current_score = s
                current = new_set
                improved = True
                break # Accept first improvement? Or find best? 
                # Finding best is better but slower. 
                # Let's find best in this neighborhood.
        
        # Re-do to find best neighbor
        current = set(best_set)
        current_score = best_score
        max_u = max(current)
        best_neighbor = None
        best_neighbor_score = current_score
        
        # Add
        for x in range(1, max_u + 6):
            if x in current:
                continue
            new_set = current | {x}
            s = calculate_score(new_set)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = new_set
        
        # Remove
        for y in list(current):
            if y == 0:
                continue
            new_set = current - {y}
            s = calculate_score(new_set)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = new_set
        
        # Replace
        for y in list(current):
            if y == 0:
                continue
            for x in range(1, max(current) + 6):
                if x in current:
                    continue
                new_set = (current - {y}) | {x}
                s = calculate_score(new_set)
                if s > best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor = new_set
                    
        if best_neighbor is not None and best_neighbor_score > best_score:
            best_score = best_neighbor_score
            best_set = list(best_neighbor)
        else:
            # Stuck. Try random restart from a random permutation of a good set?
            # Or just accept and try to add a random element far away?
            # Let's try adding a random element from a wider range occasionally
            if random.random() < 0.1:
                new_x = random.randint(1, max(current) + 20)
                if new_x not in current:
                    new_set = current | {new_x}
                    s = calculate_score(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = list(new_set)
                    else:
                        # Maybe try removing a random element
                        if len(current) > 1:
                            to_remove = random.choice([y for y in current if y != 0])
                            new_set = current - {to_remove}
                            s = calculate_score(new_set)
                            if s > best_score:
                                best_score = s
                                best_set = list(new_set)

    return sorted(best_set)
# EVOLVE-BLOCK-END
