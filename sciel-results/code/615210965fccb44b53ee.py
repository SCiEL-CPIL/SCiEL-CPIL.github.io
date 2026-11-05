import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph
    # conflicts[i] is a set of j such that i+j is a square
    # We use 0-indexed for internal representation, 1-indexed for values
    # val = idx + 1
    conflicts = [set() for _ in range(N)]
    
    for i in range(N):
        val_i = i + 1
        for j in range(N):
            val_j = j + 1
            if val_i + val_j in squares:
                conflicts[i].add(j)
                
    # Initial set: all x such that x % 3 == 1
    # This is valid because (1+1)%3 = 2, which is not a quadratic residue mod 3
    current_set = set()
    for v in vertices:
        if v % 3 == 1:
            current_set.add(v)
            
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Local Search
    # We will try to improve the set by:
    # 1. Adding any vertex that doesn't conflict with current set
    # 2. Swapping: removing a vertex u, and adding a set of vertices V_add 
    #    that don't conflict with (current_set - {u}) and don't conflict with each other.
    
    # To make it faster, we precompute "safe" additions for a given set? 
    # That's expensive. Instead, we check on the fly.
    
    # Helper: check if a candidate can be added to a set
    def can_add(candidate_val, S):
        idx = candidate_val - 1
        for v in S:
            if (v - 1) in conflicts[idx]:
                return False
        return True
        
    # Helper: check if a set of candidates is mutually compatible and compatible with S
    def can_add_set(candidates, S):
        # Check mutual compatibility
        for i, c1 in enumerate(candidates):
            idx1 = c1 - 1
            for c2 in candidates[i+1:]:
                idx2 = c2 - 1
                if idx2 in conflicts[idx1]:
                    return False
        # Check compatibility with S
        for c in candidates:
            if not can_add(c, S):
                return False
        return True
        
    # Optimization: Precompute for each vertex, the list of vertices it conflicts with
    # We already have conflicts[idx]
    
    # Search Loop
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # 1. Try simple additions
        # Iterate through candidates not in set
        candidates = [v for v in vertices if v not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add(c, current_set):
                current_set.add(c)
                improved = True
                # Don't break, try to add more if possible
                # But we need to update the list of candidates? 
                # Since we're checking against current_set which is growing, 
                # it's safe to just continue.
                
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(best_set)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # 2. Try swaps: Remove one, add one or more
        # We focus on 1-for-1 and 1-for-2 swaps initially
        
        # Identify "loose" elements in current_set? 
        # Or just random removal.
        
        if no_improvement_count > 10:
            # Try swaps
            # Pick a random element to remove
            if current_set:
                remove_candidate = random.choice(list(current_set))
                temp_set = current_set - {remove_candidate}
                
                # Find all vertices that can be added to temp_set
                addable = []
                for v in vertices:
                    if v in temp_set:
                        continue
                    if can_add(v, temp_set):
                        addable.append(v)
                        
                # Try to add as many as possible from addable
                # Greedy: sort addable by degree (fewest conflicts with temp_set? 
                # or just random)
                random.shuffle(addable)
                
                new_additions = []
                for v in addable:
                    if can_add(v, temp_set | set(new_additions)):
                        new_additions.append(v)
                        
                if len(new_additions) > 1:
                    # We removed 1, added >1. Net gain.
                    current_set = temp_set | set(new_additions)
                    improved = True
                    
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(best_set)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        
        if not improved and no_improvement_count > max_no_improvement:
            # Random restart or perturbation
            # Remove 5-10 random elements to escape local optimum
            if len(current_set) > 10:
                to_remove = random.sample(list(current_set), min(10, len(current_set)//2))
                current_set -= set(to_remove)
            no_improvement_count = 0
            
    # Final check: ensure best_set is valid
    # (It should be, but just in case)
    best_list = list(best_set)
    
    # Verify validity
    valid = True
    for i in range(len(best_list)):
        a = best_list[i]
        for j in range(i, len(best_list)):
            b = best_list[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed if something went wrong
        return list(range(1, 257, 3))
        
    return best_list
