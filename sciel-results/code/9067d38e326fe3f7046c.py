import time
from math import log
from itertools import combinations

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = -1
    best_set = [0, 1]
    
    # Exhaustive search for small n and max_val
    # n=3 to 7, max_val up to 30
    # This is fast enough.
    
    for n in range(3, 8):
        # Max val limit depends on n to keep runtime reasonable
        # For n=7, C(30,6) is ~600k. 
        # For n=8, C(30,7) is ~2M. 
        # Let's do n=3..7 with max_val=30.
        max_val = 30
        if n > 6:
            max_val = 25
            
        for comb in combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline - 5:
                break
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
                
        if time.time() > deadline - 5:
            break
            
    # If time remains, do a local search on the best set found
    if time.time() < deadline - 2:
        # Simple local search refinement
        current = list(best_set)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            n = len(current)
            # Try replacing elements
            for i in range(1, n):
                old_val = current[i]
                # Try nearby values
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 10:
                max_val = max(current)
                for new_val in range(1, max_val + 11):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
