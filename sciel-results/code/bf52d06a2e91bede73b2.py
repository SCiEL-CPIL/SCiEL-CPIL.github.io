import time
import random
from math import log

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

    # Initial candidates based on manual analysis
    # Small sets with high ratio
    candidates = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 4, 6],
        [0, 1, 4, 7],
        [0, 1, 3, 4],
        [0, 1, 2, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 1, 3, 4, 6],
        [0, 1, 4, 5, 7],
        [0, 1, 2, 5, 6],
    ]
    
    best_set = None
    best_score = -1
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Local Search
    if best_set is None:
        return [0, 1]
        
    current_set = best_set
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 100
    step_size = 10
    
    while no_improve_count < max_no_improve and time.time() < deadline - 5:
        n = len(current_set)
        improved = False
        
        # 1. Try modifying existing elements
        for i in range(1, n):
            if time.time() > deadline - 5:
                break
            old_val = current_set[i]
            # Try a range of new values
            # Keep values non-negative and distinct
            # Try nearby values and some jumps
            candidates_val = set()
            for delta in range(-step_size, step_size + 1):
                nv = old_val + delta
                if nv > 0:
                    candidates_val.add(nv)
            # Try some larger jumps
            for _ in range(5):
                candidates_val.add(random.randint(1, max(current_set) + 50))
                
            for nv in candidates_val:
                if nv == old_val:
                    continue
                # Check if nv is already in set
                if nv in current_set:
                    continue
                new_set = current_set[:]
                new_set[i] = nv
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                ns = score_set(new_set)
                if ns > current_score:
                    current_set = new_set
                    current_score = ns
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
                
        # 2. Try adding an element
        if not improved and n < 20:
            candidates_val = set()
            max_val = max(current_set)
            # Try values in gaps
            for i in range(1, n):
                gap_start = current_set[i-1]
                gap_end = current_set[i]
                if gap_end - gap_start > 1:
                    candidates_val.add(gap_start + 1)
                    candidates_val.add(gap_end - 1)
                    candidates_val.add((gap_start + gap_end) // 2)
            # Try extending
            candidates_val.add(max_val + 1)
            candidates_val.add(max_val + 2)
            candidates_val.add(max_val * 2)
            # Random
            for _ in range(10):
                candidates_val.add(random.randint(1, max_val * 2))
                
            for nv in candidates_val:
                if nv in current_set:
                    continue
                new_set = sorted(current_set + [nv])
                ns = score_set(new_set)
                if ns > current_score:
                    current_set = new_set
                    current_score = ns
                    improved = True
                    no_improve_count = 0
                    break
                    
        # 3. Try removing an element
        if not improved and n > 3:
            for i in range(1, n):
                new_set = [current_set[j] for j in range(n) if j != i]
                ns = score_set(new_set)
                if ns > current_score:
                    current_set = new_set
                    current_score = ns
                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1
            # Increase step size occasionally to escape local minima
            if no_improve_count % 10 == 0:
                step_size = int(step_size * 1.5)
                if step_size > 1000:
                    step_size = 1000
        else:
            step_size = max(10, step_size // 2)

    return current_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
