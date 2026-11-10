import random
import math
import time
from fractions import Fraction

def construct():
    N = 256
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = Fraction(1, i)
    
    # Precompute conflict pairs: for each pair (a, b), determine if they conflict
    # Two elements a, b conflict if adding both would create a 3-AP with some c
    # Actually, we need: for each x, which other elements y would form a 3-AP with x
    # and some third element z
    
    # For backtracking: when we decide to include/exclude elements, we need to track
    # which elements are "blocked" by current choices
    
    # Precompute for each pair (i, j) with i < j, the third element k that would
    # complete a 3-AP: k = 2*j - i (if i,j,k form AP with j middle)
    # or k = 2*i - j (if j,i,k form AP with i middle) 
    # or k = (i+j)/2 (if i,k,j form AP with k middle)
    
    # Better approach: for each x, precompute all y such that x and y are in a 3-AP
    # with some z. Then when we add x to S, y becomes "conflicted" if z is also in S.
    
    # For efficient backtracking, precompute for each x the list of (y, z) pairs
    # where {x, y, z} is a 3-AP and x is one of the elements
    ap_partners = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            # Check if x, y, z form a 3-AP for some z
            # Case 1: x is middle: z = 2*x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                ap_partners[x].append((y, z))
            # Case 2: y is middle: z = 2*y - x  
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                ap_partners[x].append((y, z))
            # Case 3: z is middle: z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    ap_partners[x].append((y, z))
    
    # Deduplicate ap_partners
    for x in range(1, N + 1):
        ap_partners[x] = list(set(ap_partners[x]))
    
    # Greedy initialization: add elements in order of decreasing reciprocal
    S_bool = [False] * (N + 1)
    S_list = []
    candidates = sorted(range(1, N + 1), key=lambda x: -recip[x])
    
    def is_valid_addition(S_bool, S_list, x):
        for y in S_list:
            # Check if x, y, and some z in S form a 3-AP
            # x is middle: z = 2x - y
            z = 2 * x - y
            if z != x and z != y and 1 <= z <= N and S_bool[z]:
                return False
            # y is middle: z = 2y - x
            z = 2 * y - x
            if z != x and z != y and 1 <= z <= N and S_bool[z]:
                return False
            # z is middle: z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z != x and z != y and 1 <= z <= N and S_bool[z]:
                    return False
        return True
    
    for x in candidates:
        if is_valid_addition(S_bool, S_list, x):
            S_bool[x] = True
            S_list.append(x)
    
    current_score = sum(recip[x] for x in S_list)
    best_score = current_score
    best_S_list = list(S_list)
    best_S_bool = list(S_bool)
    
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin
    
    # Backtracking search with branch and bound
    # Process elements in order of decreasing reciprocal (small numbers first)
    # State: which elements are included, which are excluded, which are undecided
    
    # For efficiency, use bitmasks or boolean arrays
    # We'll do DFS with pruning
    
    # Upper bound: sum of reciprocals of all remaining undecided elements
    # (loose but fast)
    
    def backtrack(idx, S_bool, S_list, current_score):
        nonlocal best_score, best_S_list, best_S_bool
        
        if time.time() - start_time > time_limit:
            return
        
        # If we've processed all elements
        if idx > N:
            if current_score > best_score:
                best_score = current_score
                best_S_list = list(S_list)
                best_S_bool = list(S_bool)
            return
        
        x = candidates[idx]
        
        # Option 1: Exclude x
        # Upper bound if we exclude x: current_score + sum of reciprocals of remaining
        # (very loose bound, but we can compute tighter)
        
        # Option 2: Include x (if valid)
        if is_valid_addition(S_bool, S_list, x):
            S_bool[x] = True
            S_list.append(x)
            backtrack(idx + 1, S_bool, S_list, current_score + recip[x])
            S_list.pop()
            S_bool[x] = False
        
        # Option 1: Exclude x
        backtrack(idx + 1, S_bool, S_list, current_score)
    
    # The full backtracking is too slow for N=256. Let's use a smarter approach:
    # Start from greedy solution, then do local search with swaps and additions
    
    # Improved local search: try all possible swaps and additions systematically
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        
        # Try all possible single additions
        for x in range(1, N + 1):
            if not best_S_bool[x]:
                if is_valid_addition(best_S_bool, best_S_list, x):
                    new_score = best_score + recip[x]
                    if new_score > best_score:
                        best_score = new_score
                        best_S_bool[x] = True
                        best_S_list.append(x)
                        changed = True
        
        # Try all possible swaps
        for x in list(best_S_list):
            for y in range(1, N + 1):
                if best_S_bool[y]:
                    continue
                if recip[y] <= recip[x]:
                    continue
                temp_bool = list(best_S_bool)
                temp_bool[x] = False
                temp_list = [z for z in best_S_list if z != x]
                if is_valid_addition(temp_bool, temp_list, y):
                    new_score = best_score - recip[x] + recip[y]
                    if new_score > best_score:
                        best_score = new_score
                        best_S_bool[x] = False
                        best_S_bool[y] = True
                        best_S_list.remove(x)
                        best_S_list.append(y)
                        changed = True
                        break
            if changed:
                break
        
        # Try double swaps: remove two, add two
        if not changed and time.time() - start_time < time_limit - 2.0:
            S_list_copy = list(best_S_list)
            for i in range(len(S_list_copy)):
                for j in range(i + 1, len(S_list_copy)):
                    x1 = S_list_copy[i]
                    x2 = S_list_copy[j]
                    # Try removing x1 and x2, adding y1 and y2
                    temp_bool = list(best_S_bool)
                    temp_bool[x1] = False
                    temp_bool[x2] = False
                    temp_list = [z for z in best_S_list if z != x1 and z != x2]
                    
                    # Try adding pairs of elements
                    for y1 in range(1, N + 1):
                        if best_S_bool[y1] or y1 == x1 or y1 == x2:
                            continue
                        if not is_valid_addition(temp_bool, temp_list, y1):
                            continue
                        temp_bool2 = list(temp_bool)
                        temp_bool2[y1] = True
                        temp_list2 = temp_list + [y1]
                        
                        for y2 in range(1, N + 1):
                            if temp_bool[y2] or y2 == x1 or y2 == x2 or y2 == y1:
                                continue
                            if recip[y1] + recip[y2] <= recip[x1] + recip[x2]:
                                continue
                            if is_valid_addition(temp_bool2, temp_list2, y2):
                                new_score = best_score - recip[x1] - recip[x2] + recip[y1] + recip[y2]
                                if new_score > best_score:
                                    best_score = new_score
                                    best_S_bool = temp_bool
                                    best_S_bool[y1] = True
                                    best_S_bool[y2] = True
                                    best_S_list = temp_list2
                                    best_S_list.append(y2)
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
    
    return list(best_S_list)
