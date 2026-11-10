import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 10
    p, q = 2, 5
    
    def crt(s1, s2):
        return s1 * q + s2
    
    def decomp(i):
        return (i % p, i // p)
    
    def is_latin(sq):
        # Check rows
        for i in range(n):
            row = sq[i]
            if len(set(row)) != n:
                return False
        # Check columns
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True
    
    def are_orthogonal(sq1, sq2):
        seen = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True
    
    mols = []
    seen_keys = set()
    
    # Generate all candidate Latin squares from the CRT construction
    # s1 = (i1 + j1 + a*i2 + b*j2) % 2
    # s2 = (i2 + k*j2 + c*i1 + d*j1) % 5
    # where i = i1 + 5*i2, j = j1 + 5*j2 (i1,j1 in 0..1, i2,j2 in 0..4)
    
    candidates = []
    
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1, 2, 3, 4]:
                for d in [0, 1, 2, 3, 4]:
                    for k in [1, 2, 3, 4]:
                        if time.time() > deadline:
                            break
                        
                        sq = [[0]*n for _ in range(n)]
                        for i in range(n):
                            i1, i2 = decomp(i)
                            for j in range(n):
                                j1, j2 = decomp(j)
                                s1 = (i1 + j1 + a*i2 + b*j2) % p
                                s2 = (i2 + k*j2 + c*i1 + d*j1) % q
                                sq[i][j] = crt(s1, s2)
                        
                        if not is_latin(sq):
                            continue
                        
                        key = tuple(tuple(row) for row in sq)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        
                        candidates.append(sq)
    
    # Now greedily select mutually orthogonal squares
    # Sort candidates by some heuristic to find orthogonal sets faster
    # Try each candidate as a starting point
    
    best_set = []
    
    for start_idx, start_sq in enumerate(candidates):
        if time.time() > deadline:
            break
        
        current_set = [start_sq]
        
        for other_sq in candidates[start_idx+1:]:
            if time.time() > deadline:
                break
            
            # Check if other_sq is orthogonal to all in current_set
            is_orth_to_all = True
            for existing in current_set:
                if not are_orthogonal(existing, other_sq):
                    is_orth_to_all = False
                    break
            
            if is_orth_to_all:
                current_set.append(other_sq)
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                    if len(best_set) >= 8:
                        return best_set
    
    # If we didn't find 8, try a different approach:
    # Use the known fact that we can get up to 8 MOLS of order 10
    # Try to build from scratch with a more systematic approach
    
    if len(best_set) < 8:
        # Try all possible first squares
        for first_sq in candidates:
            if time.time() > deadline:
                break
            
            current_set = [first_sq]
            
            for second_sq in candidates:
                if second_sq is first_sq:
                    continue
                if time.time() > deadline:
                    break
                
                if not are_orthogonal(first_sq, second_sq):
                    continue
                
                current_set = [first_sq, second_sq]
                
                for third_sq in candidates:
                    if third_sq is first_sq or third_sq is second_sq:
                        continue
                    if time.time() > deadline:
                        break
                    
                    if not are_orthogonal(first_sq, third_sq) or not are_orthogonal(second_sq, third_sq):
                        continue
                    
                    current_set = [first_sq, second_sq, third_sq]
                    
                    for fourth_sq in candidates:
                        if fourth_sq in current_set:
                            continue
                        if time.time() > deadline:
                            break
                        
                        if all(are_orthogonal(existing, fourth_sq) for existing in current_set):
                            current_set.append(fourth_sq)
                            if len(current_set) > len(best_set):
                                best_set = current_set[:]
                                if len(best_set) >= 8:
                                    return best_set
    
    # Return what we have, ensuring at least 1 square
    if not best_set and candidates:
        best_set = [candidates[0]]
    
    # Ensure we return valid Python types
    result = []
    for sq in best_set:
        clean_sq = [[int(sq[i][j]) for j in range(n)] for i in range(n)]
        result.append(clean_sq)
    
    return result
