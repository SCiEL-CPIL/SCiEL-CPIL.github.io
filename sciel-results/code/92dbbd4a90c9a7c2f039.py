import time
from math import log, floor
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Geometric progression search
    # U = {floor(b^k) for k=0,...,m}
    # Try various bases and lengths
    
    # First, try integer bases with various lengths
    for b in range(2, 30):
        if time.time() > deadline - 5:
            break
        for m in range(2, 25):
            if time.time() > deadline - 5:
                break
            U = [floor(b**k) for k in range(m + 1)]
            # Remove duplicates
            U = sorted(set(U))
            if len(U) < 2:
                continue
            if max(U) > 10**7:  # Keep max reasonable for computation
                break
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Non-integer base geometric progression
    # Try fractional bases
    for b_int in range(10, 100):  # b = b_int / 10.0
        b = b_int / 10.0
        if b < 1.0:
            continue
        for m in range(2, 30):
            if time.time() > deadline - 5:
                break
            U = [floor(b**k) for k in range(m + 1)]
            U = sorted(set(U))
            if len(U) < 2:
                continue
            if max(U) > 10**7:
                break
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 3: Ternary search over base b for fixed m
    # For a given m, the score as a function of b is unimodal (hopefully)
    for m in range(3, 20):
        if time.time() > deadline - 10:
            break
        
        # Search over b in [1.5, 15]
        lo = 1.5
        hi = 15.0
        for _ in range(40):  # 40 iterations of ternary search
            if time.time() > deadline - 10:
                break
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            
            U1 = [floor(m1**k) for k in range(m + 1)]
            U1 = sorted(set(U1))
            s1 = score_set(U1) if len(U1) >= 2 and max(U1) <= 10**7 else -1
            
            U2 = [floor(m2**k) for k in range(m + 1)]
            U2 = sorted(set(U2))
            s2 = score_set(U2) if len(U2) >= 2 and max(U2) <= 10**7 else -1
            
            if s1 < s2:
                lo = m1
            else:
                hi = m2
        
        # Evaluate at the optimal point
        b_opt = (lo + hi) / 2
        U = [floor(b_opt**k) for k in range(m + 1)]
        U = sorted(set(U))
        if len(U) >= 2 and max(U) <= 10**7:
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 4: Local search on the best set found
    if time.time() < deadline - 15:
        current_set = set(best_set)
        
        # Try adding elements
        max_current = max(best_set)
        for new_elem in range(1, min(max_current + 50, 500)):
            if time.time() > deadline - 15:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
                max_current = max(best_set)
        
        # Try removing elements
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline - 15:
                    break
                if elem == 0:
                    continue
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
    
    # Phase 5: Simulated annealing with fast scoring
    if time.time() < deadline - 20:
        import random
        current = set(best_set)
        current_score = best_score
        T = 0.1
        cooling = 0.999
        
        while time.time() < deadline - 20 and T > 0.001:
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.4:
                # Add an element
                max_u = max(current)
                new_elem = random.randint(1, max_u + 20)
                if new_elem not in current:
                    neighbor = current | {new_elem}
            elif move_type < 0.7:
                # Remove an element
                removable = [x for x in current if x != 0 and x != max(current)]
                if removable:
                    elem_to_remove = random.choice(removable)
                    neighbor = current - {elem_to_remove}
                else:
                    continue
            elif move_type < 0.9:
                # Replace an element
                removable = [x for x in current if x != 0]
                if removable:
                    elem_to_replace = random.choice(removable)
                    max_u = max(current)
                    new_elem = random.randint(1, max_u + 20)
                    if new_elem not in current:
                        neighbor = (current - {elem_to_replace}) | {new_elem}
                    else:
                        continue
                else:
                    continue
            else:
                # Swap two elements
                elems = [x for x in current if x != 0]
                if len(elems) >= 2:
                    e1, e2 = random.sample(elems, 2)
                    max_u = max(current)
                    n1 = random.randint(1, max_u + 20)
                    n2 = random.randint(1, max_u + 20)
                    if n1 not in current and n2 not in current and n1 != n2:
                        neighbor = (current - {e1, e2}) | {n1, n2}
                    else:
                        continue
                else:
                    continue
            
            if len(neighbor) < 2:
                continue
            
            neighbor_score = score_set(list(neighbor))
            if neighbor_score < 0:
                continue
            
            delta = neighbor_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            
            T *= cooling
    
    return best_set
