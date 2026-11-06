import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    N = 256

    def check_add_validity(S_set, x):
        """
        Check if x can be added to S_set without creating a 3-AP.
        S_set is a set of integers.
        """
        for y in S_set:
            # Case 1: x, y, z => z = 2y - x
            z1 = 2 * y - x
            if z1 in S_set and z1 != x and z1 != y:
                return False
            
            # Case 2: y, x, z => z = 2x - y
            z2 = 2 * x - y
            if z2 in S_set and z2 != x and z2 != y:
                return False
            
            # Case 3: y, z, x => z = (x + y) / 2
            if (x + y) % 2 == 0:
                z3 = (x + y) // 2
                if z3 in S_set and z3 != x and z3 != y:
                    return False
        return True

    def greedy_construct():
        S = set()
        for i in range(1, N + 1):
            if check_add_validity(S, i):
                S.add(i)
        return S

    def calculate_score(S):
        return sum(1.0 / x for x in S)

    def calculate_score_frac(S):
        return sum(Fraction(1, x) for x in S)

    def local_search(S, max_iters=20000):
        """
        Perform local search by swapping larger elements with smaller ones.
        """
        S_list = list(S)
        S_set = set(S)
        current_score = calculate_score(S_set)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Find best swap: remove y, add x, where x < y
            # This increases score by 1/x - 1/y > 0
            best_gain = 0
            best_x = None
            best_y = None
            
            # Optimization: Only consider x not in S, y in S
            # And x < y
            # Sort candidates by potential gain (1/x) descending to find high gains earlier?
            # Or just iterate all.
            # To speed up, we can pre-compute 1/x
            inv = {i: 1.0/i for i in range(1, N+1)}
            
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                for y in S_list:
                    if y <= x:
                        continue
                    gain = inv[x] - inv[y]
                    if gain <= best_gain:
                        continue
                    # Check validity of S - {y} + {x}
                    S_temp = S_set - {y}
                    if check_add_validity(S_temp, x):
                        best_gain = gain
                        best_x = x
                        best_y = y
            
            if best_x is not None:
                S_set.remove(best_y)
                S_set.add(best_x)
                S_list = list(S_set)
                current_score += best_gain
                improved = True
        
        return S_set, current_score

    def perturb_and_search(S):
        """
        Remove a few elements and re-optimize.
        """
        S_perturbed = set(S)
        if len(S_perturbed) < 10:
            return S_perturbed, calculate_score(S_perturbed)
        
        # Remove 3-6 random elements
        num_remove = random.randint(3, 6)
        to_remove = random.sample(list(S_perturbed), num_remove)
        for r in to_remove:
            S_perturbed.remove(r)
            
        S_perturbed, score_p = local_search(S_perturbed)
        return S_perturbed, score_p

    best_score = 0.0
    best_S = set()
    
    # Initial greedy
    S = greedy_construct()
    S, score = local_search(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
        
    # Iterative improvement
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try a perturbation
        S_perturbed, score_p = perturb_and_search(best_S)
        
        if score_p > best_score:
            best_score = score_p
            best_S = S_perturbed
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Also try a fresh greedy restart occasionally to escape deep local minima
        if no_improve_count % 5 == 0:
            S_new = greedy_construct()
            S_new, score_new = local_search(S_new)
            if score_new > best_score:
                best_score = score_new
                best_S = set(S_new)
                no_improve_count = 0

    # Final polish: try to add any remaining elements that fit
    # (Though local search should have handled this, just in case)
    for x in range(1, N + 1):
        if x not in best_S:
            if check_add_validity(best_S, x):
                best_S.add(x)
                best_score = calculate_score(best_S)
                
    # Try one more round of local search on the final set
    final_S, final_score = local_search(best_S)
    if final_score > best_score:
        best_S = final_S
        best_score = final_score

    return list(best_S)
