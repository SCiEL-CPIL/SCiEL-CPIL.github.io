import sys
import time
import numpy as np

def construct():
    n = 4096
    max_sum = 3 * n
    
    used_sums = bytearray(max_sum + 1)
    chosen = []
    
    best_set = []
    best_size = 0
    
    start_time = time.time()
    timeout = 55.0
    
    def can_add(x):
        """
        Check if x can be added to chosen without creating sum collisions.
        Returns a list of new sums if valid, else None.
        """
        new_sums = []
        
        # 1. (x, x, x)
        s = 3 * x
        if used_sums[s]:
            return None
        new_sums.append(s)
        
        # 2. (a, x, x) for a in chosen
        for a in chosen:
            s = a + 2 * x
            if used_sums[s]:
                return None
            new_sums.append(s)
        
        # 3. (a, b, x) for a <= b in chosen
        k = len(chosen)
        for i in range(k):
            a = chosen[i]
            for j in range(i, k):
                b = chosen[j]
                s = a + b + x
                if used_sums[s]:
                    return None
                new_sums.append(s)
        
        # Check for internal collisions among new_sums
        if len(new_sums) != len(set(new_sums)):
            return None
            
        return new_sums

    def backtrack(start_val, depth):
        nonlocal best_set, best_size, start_time
        
        if time.time() - start_time > timeout:
            return
        
        if depth > best_size:
            best_size = depth
            best_set = chosen[:]
        
        # Pruning heuristic
        if depth > 0:
            max_add = int((3 * n) / (depth * depth))
            if depth + max_add <= best_size:
                return
        
        tried = 0
        max_tries = 20
        
        for x in range(start_val, n + 1):
            if time.time() - start_time > timeout:
                return
            
            if tried >= max_tries:
                break
            
            new_sums = can_add(x)
            if new_sums is not None:
                chosen.append(x)
                for s in new_sums:
                    used_sums[s] = 1
                
                backtrack(x + 1, depth + 1)
                
                chosen.pop()
                for s in new_sums:
                    used_sums[s] = 0
                
                tried += 1

    backtrack(1, 0)
    
    return best_set
