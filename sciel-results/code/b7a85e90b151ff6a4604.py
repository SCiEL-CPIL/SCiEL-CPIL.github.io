import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit to be safe
    
    # We are looking for a subset S of [1, 22] such that all subset sums of 1/x (x in S) are distinct.
    # This is equivalent to saying that the set {1/x : x in S} is "dissociated".
    # A set A is dissociated if no non-trivial linear combination with coefficients in {-1, 0, 1} equals 0.
    # In the context of subset sums, this means that if sum_{a in T1} a = sum_{a in T2} a, then T1 = T2.
    
    # We will use a backtracking search to find the largest such set.
    # To speed up the collision check, we will use integer arithmetic by scaling by the LCM of 1..22.
    
    # LCM of 1..22 is 232792560
    LCM = 232792560
    values = {i: LCM // i for i in range(1, 23)}
    
    best_set = []
    best_size = 0
    
    # To make the search efficient, we should add elements in an order that makes collisions unlikely early on.
    # Adding larger reciprocals (smaller indices) first tends to create larger gaps, making it harder for smaller reciprocals to cause collisions.
    # Alternatively, adding smaller reciprocals (larger indices) first might be better because they are "smaller" perturbations.
    # Let's try sorting indices in descending order (largest indices first, i.e., smallest reciprocals).
    # This is a heuristic. We will also try ascending order.
    
    # For a robust search, we can use a branch and bound approach.
    # State: current_set, current_sums (set of integers), remaining_indices
    # Pruning: if len(current_set) + len(remaining_indices) <= best_size, prune.
    
    # To further optimize, we can pre-check if a new element can be added without collision.
    # Collision check: given current_sums (set of ints) and new value v,
    # collision exists if there exist s1, s2 in current_sums such that s1 + v = s2.
    # This is equivalent to checking if (s2 - s1) = v for any s1, s2 in current_sums.
    # Or simply, check if any s + v is in current_sums.
    
    def check_collision(current_sums, v):
        for s in current_sums:
            if s + v in current_sums:
                return True
        return False
    
    def search(current_set, current_sums, remaining_indices):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) + len(remaining_indices) <= best_size:
            return
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if not remaining_indices:
            return
            
        # Try to add each element in remaining_indices
        for i in range(len(remaining_indices)):
            if time.time() > deadline:
                return
                
            idx = remaining_indices[i]
            v = values[idx]
            
            if check_collision(current_sums, v):
                continue
                
            # No collision, proceed
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
                
            new_remaining = remaining_indices[:i] + remaining_indices[i+1:]
            
            search(current_set + [idx], new_sums, new_remaining)
            
            # Backtrack
    
    # To improve the search, we can order the elements.
    # Let's try a few different orderings.
    
    orderings = [
        list(range(22, 0, -1)),  # 22, 21, ..., 1
        list(range(1, 23)),      # 1, 2, ..., 22
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        search([], {0}, order)
        
    if not best_set:
        # Fallback
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
