import time
import random
from math import log, floor

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

    best_set = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    best_score = score_set(best_set)

    def generate_sumset(A, B):
        """Generate U = {a + b for a in A for b in B}"""
        U = set()
        for a in A:
            for b in B:
                U.add(a + b)
        return sorted(list(U))

    def local_search_sumset(A, B, base_score):
        """
        Local search on A and B.
        A and B are lists of non-negative integers containing 0.
        """
        current_A = list(A)
        current_B = list(B)
        current_U = generate_sumset(current_A, current_B)
        current_score = base_score
        
        no_improve = 0
        max_no_improve = 50
        
        while no_improve < max_no_improve and time.time() < deadline - 5:
            improved = False
            
            # Try modifying A
            if len(current_A) > 1:
                idx = random.randint(1, len(current_A) - 1)
                old_val = current_A[idx]
                # Try nearby values
                for delta in range(-10, 11):
                    if delta == 0: continue
                    new_val = old_val + delta
                    if new_val < 0: continue
                    if new_val in current_A: continue
                    
                    new_A = list(current_A)
                    new_A[idx] = new_val
                    new_A = sorted(set(new_A))
                    
                    new_U = generate_sumset(new_A, current_B)
                    new_score = score_set(new_U)
                    
                    if new_score > current_score:
                        current_A = new_A
                        current_U = new_U
                        current_score = new_score
                        improved = True
                        no_improve = 0
                        break
            
            # Try modifying B
            if not improved and len(current_B) > 1:
                idx = random.randint(1, len(current_B) - 1)
                old_val = current_B[idx]
                for delta in range(-10, 11):
                    if delta == 0: continue
                    new_val = old_val + delta
                    if new_val < 0: continue
                    if new_val in current_B: continue
                    
                    new_B = list(current_B)
                    new_B[idx] = new_val
                    new_B = sorted(set(new_B))
                    
                    new_U = generate_sumset(current_A, new_B)
                    new_score = score_set(new_U)
                    
                    if new_score > current_score:
                        current_B = new_B
                        current_U = new_U
                        current_score = new_score
                        improved = True
                        no_improve = 0
                        break
            
            # Try adding to A
            if not improved and len(current_A) < 10:
                max_a = max(current_A)
                candidates = [max_a + 1, max_a + 2, max_a + 3, random.randint(1, max_a + 5)]
                random.shuffle(candidates)
                for c in candidates:
                    if c in current_A: continue
                    new_A = sorted(current_A + [c])
                    new_U = generate_sumset(new_A, current_B)
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        current_A = new_A
                        current_U = new_U
                        current_score = new_score
                        improved = True
                        no_improve = 0
                        break

            # Try adding to B
            if not improved and len(current_B) < 10:
                max_b = max(current_B)
                candidates = [max_b + 1, max_b + 2, max_b + 3, random.randint(1, max_b + 5)]
                random.shuffle(candidates)
                for c in candidates:
                    if c in current_B: continue
                    new_B = sorted(current_B + [c])
                    new_U = generate_sumset(current_A, new_B)
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        current_B = new_B
                        current_U = new_U
                        current_score = new_score
                        improved = True
                        no_improve = 0
                        break

            # Try removing from A (if size > 2)
            if not improved and len(current_A) > 2:
                idx = random.randint(1, len(current_A) - 1)
                new_A = [x for i, x in enumerate(current_A) if i != idx]
                new_U = generate_sumset(new_A, current_B)
                new_score = score_set(new_U)
                if new_score > current_score:
                    current_A = new_A
                    current_U = new_U
                    current_score = new_score
                    improved = True
                    no_improve = 0

            # Try removing from B (if size > 2)
            if not improved and len(current_B) > 2:
                idx = random.randint(1, len(current_B) - 1)
                new_B = [x for i, x in enumerate(current_B) if i != idx]
                new_U = generate_sumset(current_A, new_B)
                new_score = score_set(new_U)
                if new_score > current_score:
                    current_B = new_B
                    current_U = new_U
                    current_score = new_score
                    improved = True
                    no_improve = 0

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

        return current_U, current_score

    # Initial configurations for A and B
    # Strategy: A is small/dense, B is larger/sparse to create asymmetry
    initial_configs = []
    
    # 1. A = {0, 1, 2}, B = {0, 5, 10, 15}
    initial_configs.append(([0, 1, 2], [0, 5, 10, 15]))
    
    # 2. A = {0, 1}, B = {0, 3, 6, 9, 12}
    initial_configs.append(([0, 1], [0, 3, 6, 9, 12]))
    
    # 3. A = {0, 2, 4}, B = {0, 5, 10}
    initial_configs.append(([0, 2, 4], [0, 5, 10]))
    
    # 4. Random small sets
    for _ in range(5):
        size_a = random.randint(2, 5)
        size_b = random.randint(2, 5)
        A = [0] + sorted(random.sample(range(1, 20), size_a - 1))
        B = [0] + sorted(random.sample(range(1, 50), size_b - 1))
        initial_configs.append((A, B))
        
    # 5. Geometric-like A, Linear B
    A_geo = [0, 1, 3, 9]
    B_lin = [0, 10, 20, 30]
    initial_configs.append((A_geo, B_lin))

    for A_init, B_init in initial_configs:
        if time.time() > deadline - 10:
            break
            
        U_init = generate_sumset(A_init, B_init)
        s_init = score_set(U_init)
        
        if s_init > best_score:
            best_score = s_init
            best_set = U_init
            
        U_res, s_res = local_search_sumset(A_init, B_init, s_init)
        
        if s_res > best_score:
            best_score = s_res
            best_set = U_res

    return best_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
