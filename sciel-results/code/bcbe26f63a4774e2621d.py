import time
import math
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        if len(U) < 2:
            return 0.0
        
        max_u = U[-1]
        if max_u == 0:
            return 0.0
            
        sum_set = set()
        diff_set = set()
        
        n = len(U)
        for i in range(n):
            ui = U[i]
            for j in range(n):
                uj = U[j]
                sum_set.add(ui + uj)
                diff_set.add(ui - uj)
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return 0.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0, 1, 3]
    best_score = calc_score(best_set)

    def update_best(U, s):
        nonlocal best_score, best_set
        if s > best_score:
            best_score = s
            best_set = U[:]

    # --- Phase 1: Exhaustive search for small sets (n=3, n=4) ---
    # This is crucial for finding the true local optima in small dimensions.
    
    # n=3: [0, a, b] with 1 <= a < b
    # Max b: score decreases as max_u increases if ratio doesn't grow fast enough.
    # For n=3, max ratio is 3/2 (if sums are {0, a, 2a, a+b, 2a+b, 2a+2b? no}
    # Sums: 0, a, 2a, a+b, 2a+b, 2a+2b? No, sums of {0,a,b}: 0, a, 2a, a+b, 2a+b, 2b? No.
    # Sums: 0+0=0, 0+a=a, 0+b=b, a+0=a, a+a=2a, a+b=a+b, b+0=b, b+a=a+b, b+b=2b.
    # Unique sums: {0, a, b, 2a, a+b, 2b}. Size 6 unless collisions.
    # Diffs: 0, a, b, -a, -b, a-b, b-a. Size 7 unless a=b (impossible) or a-b=0 etc.
    # If a=1, b=2: sums {0,1,2,3,4}, diffs {-2,-1,0,1,2}. Ratio 5/5=1. Score 1.
    # If a=1, b=3: sums {0,1,2,3,4,6}, diffs {-3,-2,-1,0,1,2,3}. Ratio 7/6.
    # If a=1, b=4: sums {0,1,2,4,5,8}, diffs {-4,-3,-2,-1,0,1,2,3,4}. Ratio 9/6=1.5.
    # log(1.5)/log(9) ~ 0.405/2.2 ~ 0.18. Score 1.18?
    # Let's just search.
    
    # Search n=3 up to b=30
    for a in range(1, 15):
        for b in range(a + 1, 35):
            if time.time() > deadline - 10:
                return best_set
            U = [0, a, b]
            s = calc_score(U)
            update_best(U, s)

    # Search n=4 up to c=35
    # This is 15 * 20 * 30 ~ 9000 iterations, very fast.
    for a in range(1, 12):
        for b in range(a + 1, 25):
            for c in range(b + 1, 40):
                if time.time() > deadline - 10:
                    return best_set
                U = [0, a, b, c]
                s = calc_score(U)
                update_best(U, s)

    # Search n=5 up to d=30 (limited)
    # 10 * 10 * 15 * 20 = 30000. Might be tight but worth it if fast enough.
    # Let's restrict a bit more.
    for a in range(1, 8):
        for b in range(a + 1, 18):
            for c in range(b + 1, 30):
                for d in range(c + 1, 40):
                    if time.time() > deadline - 5:
                        return best_set
                    U = [0, a, b, c, d]
                    s = calc_score(U)
                    update_best(U, s)

    # --- Phase 2: Local Search from Best Found and Promising Starts ---
    
    def local_search(U, max_iters=2000):
        nonlocal best_score, best_set
        current = sorted(set(U))
        current_score = calc_score(current)
        update_best(current, current_score)
        
        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            max_u = current[-1]
            
            # 1. Try adding an element
            # Range: 1 to max_u + 15
            # We can optimize by checking only a few candidates or random ones if range is large
            # But for small sets, range is small.
            for new_elem in range(1, max_u + 15):
                if new_elem in current:
                    continue
                new_U = sorted(current + [new_elem])
                s = calc_score(new_U)
                if s > current_score + 1e-12:
                    current = new_U
                    current_score = s
                    improved = True
                    update_best(current, current_score)
                    break
            
            if improved:
                continue
            
            # 2. Try removing an element
            for i in range(1, len(current)):
                new_U = current[:i] + current[i+1:]
                if len(new_U) < 2:
                    continue
                s = calc_score(new_U)
                if s > current_score + 1e-12:
                    current = new_U
                    current_score = s
                    improved = True
                    update_best(current, current_score)
                    break
            
            if improved:
                continue
            
            # 3. Try swapping an element
            # Replace current[i] with new_elem
            for i in range(1, len(current)):
                old_elem = current[i]
                # Search range for new element
                # We can search a wider range for swaps
                for new_elem in range(1, max_u + 20):
                    if new_elem == old_elem or new_elem in current:
                        continue
                    new_U = current[:]
                    new_U[i] = new_elem
                    new_U.sort()
                    s = calc_score(new_U)
                    if s > current_score + 1e-12:
                        current = new_U
                        current_score = s
                        improved = True
                        update_best(current, current_score)
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0
        
        return current, current_score

    # Run local search from the best set found in exhaustive search
    if time.time() < deadline - 20:
        local_search(best_set[:], max_iters=3000)

    # Run local search from other promising starting points
    # These are often good seeds for local search
    starting_points = [
        [0, 1, 3, 7],
        [0, 1, 3, 9],
        [0, 1, 4, 7],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 2, 3, 7],
        [0, 1, 3, 4, 10],
        [0, 1, 3, 7, 15],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 10, 22],
    ]
    
    # Add the best set found so far
    starting_points.append(best_set[:])
    
    # Remove duplicates
    seen = set()
    unique_starts = []
    for sp in starting_points:
        key = tuple(sp)
        if key not in seen:
            seen.add(key)
            unique_starts.append(sp)
            
    for start in unique_starts:
        if time.time() > deadline - 10:
            break
        # Only run if it's potentially competitive
        s_start = calc_score(start)
        if s_start > best_score - 0.05:
            local_search(start, max_iters=2000)

    # --- Phase 3: Random Restarts ---
    if time.time() < deadline - 5:
        random.seed(42)
        restarts = 10
        for r in range(restarts):
            if time.time() > deadline - 2:
                break
            
            k = random.randint(3, 8)
            m = random.randint(10, 50)
            others = random.sample(range(1, m + 1), k - 1)
            U_start = [0] + sorted(others)
            
            local_search(U_start, max_iters=1000)

    return best_set

# EVOLVE-BLOCK-END
