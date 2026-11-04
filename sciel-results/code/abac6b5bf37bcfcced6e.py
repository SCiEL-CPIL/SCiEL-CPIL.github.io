import time
import random
import math

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    if len(U) < 2:
        return 1.0
        
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    # Calculate sumset
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    sum_size = len(sumset)
    
    # Calculate difference set
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diff_size = len(diffset)
    
    ratio = diff_size / sum_size
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Predefined good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 6],
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 3, 9],
        [0, 2, 3, 6],
        [0, 1, 4, 6],
        [0, 1, 2, 6],
        [0, 1, 3, 5],
        [0, 1, 4, 5],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Random search for small sets
    # N in 3 to 6
    # Max element in 1 to 30
    
    for n in range(3, 7):
        # Check time
        if time.time() > deadline:
            break
            
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Generate random set with 0
            # Ensure increasing order
            U = [0]
            current = 1
            for _ in range(n - 1):
                # Jump size
                jump = random.randint(1, 5)
                current += jump
                if current > 30:
                    break
                U.append(current)
            
            if len(U) != n:
                continue
                
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # Local search on best_set
    if time.time() < deadline - 5:
        U = best_set[:]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try modifying each element
            for i in range(len(U)):
                if i == 0:
                    continue
                # Try changing U[i]
                for delta in range(-5, 6):
                    new_val = U[i] + delta
                    if new_val < 1 or new_val > 40:
                        continue
                    if new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    s = score_set(new_U)
                    if s > best_score:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
                    
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to keep the structure 
# or just provide the function. The prompt asks to keep function names.
# I will wrap it in the EVOLVE block as requested by the format of the current program.
# Actually, the current program has the function inside the block.
# I will output the whole file.
# EVOLVE-BLOCK-END
