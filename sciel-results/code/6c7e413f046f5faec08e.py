import time
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def calc_raw(U):
        U = sorted(set(U))
        if not U or U[0] != 0:
            return 0.0
        n = len(U)
        if n == 1:
            return 1.0
        M = max(U)
        if M == 0:
            return 1.0
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        if len(sumset) == 0:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 1:
            return 1.0
        denom = math.log(2 * M + 1)
        if denom <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / denom

    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 9],
        [0, 1, 4],
        [0, 2, 5]
    ]
    
    best_set = candidates[0]
    best_score = calc_raw(best_set)

    # Local search
    max_val = 30
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(candidates)):
            U = set(candidates[i])
            # Try adding
            for k in range(1, max_val + 1):
                if k not in U:
                    new_U = list(U | {k})
                    s = calc_raw(new_U)
                    if s > best_score:
                        best_score = s
                        best_set = new_U
                        improved = True
            # Try removing
            for k in U:
                if k != 0:
                    new_U = list(U - {k})
                    if len(new_U) >= 2:
                        s = calc_raw(new_U)
                        if s > best_score:
                            best_score = s
                            best_set = new_U
                            improved = True
            # Try swapping
            for k_out in U:
                if k_out == 0: continue
                for k_in in range(1, max_val + 1):
                    if k_in not in U:
                        new_U = (U - {k_out}) | {k_in}
                        s = calc_raw(list(new_U))
                        if s > best_score:
                            best_score = s
                            best_set = list(new_U)
                            improved = True
        
        # If improved, reset candidates to just the best found to focus search
        if improved:
            candidates = [best_set]
        else:
            # If no improvement, try expanding max_val slightly or break
            # For this task, small sets are likely optimal
            break

    # Brute force small M to be sure
    # Check all subsets of {1..10}
    if time.time() < deadline:
        for n in range(2, 6):
            for combo in itertools.combinations(range(1, 11), n-1):
                U = [0] + list(combo)
                s = calc_raw(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    return best_set
