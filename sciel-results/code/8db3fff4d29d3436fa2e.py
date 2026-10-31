"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with Sidon-like structures (maximize distinct differences)
    - Use local search to minimize |U+U| while preserving |U-U|
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def local_search(U, max_iters=1000, max_val=1000):
        """
        Enhanced local search with simulated annealing-like acceptance.
        Focuses on moves that increase |U-U| or decrease |U+U|.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        current_score = best_score
        current_U = list(U)
        
        # Simple cooling schedule
        temperature = 0.05
        cooling = 0.99
        
        for iteration in range(max_iters):
            U_set = set(current_U)
            n = len(current_U)
            
            # Generate a random move
            move_type = random.randint(0, 2)
            
            new_U = None
            if move_type == 0 and n > 1:
                # Single move: change one element (not 0)
                i = random.randint(1, n - 1)
                # Try a few random deltas
                for _ in range(5):
                    delta = random.randint(-50, 50)
                    new_val = current_U[i] + delta
                    if new_val < 0 or new_val > max_val:
                        continue
                    if new_val in U_set:
                        continue
                    new_U = list(current_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        new_U = None
                        continue
                    break
            elif move_type == 1 and n > 2:
                # Double move: change two elements
                i, j = random.sample(range(1, n), 2)
                for _ in range(5):
                    d1 = random.randint(-20, 20)
                    d2 = random.randint(-20, 20)
                    new_v1 = current_U[i] + d1
                    new_v2 = current_U[j] + d2
                    if new_v1 < 0 or new_v2 < 0 or new_v1 > max_val or new_v2 > max_val:
                        continue
                    if new_v1 in U_set or new_v2 in U_set:
                        continue
                    if new_v1 == new_v2:
                        continue
                    new_U = list(current_U)
                    new_U[i] = new_v1
                    new_U[j] = new_v2
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        new_U = None
                        continue
                    break
            elif move_type == 2:
                # Add or remove element
                if random.random() < 0.5 and n < 40:
                    # Add
                    cand = random.randint(1, max_val)
                    if cand not in U_set:
                        new_U = sorted(current_U + [cand])
                else:
                    # Remove
                    if n > 2:
                        i = random.randint(1, n - 1)
                        new_U = [x for j, x in enumerate(current_U) if j != i]
            
            if new_U is None:
                continue
                
            new_score = compute_score(new_U)
            
            # Accept if better, or with probability exp((new-old)/T)
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(current_U)
            else:
                # Accept worse move to escape local minima
                diff = new_score - current_score
                if diff < 0:
                    prob = math.exp(diff / temperature)
                    if random.random() < prob:
                        current_U = new_U
                        current_score = new_score
                        
            temperature *= cooling
            
        return best_U, best_score

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Known Sidon-like sequences (Mian-Chowla style)
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115, 131, 148])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86])
    
    # 2. Greedy Sidon Construction with Sum Bias
    # We want to build U such that differences are unique (Sidon) but sums collide.
    # This is tricky. Standard Sidon maximizes distinct diffs.
    # We will generate multiple random Sidon sets and let local search optimize.
    
    random.seed(42)
    for trial in range(20):
        n_target = random.randint(10, 25)
        max_val = random.randint(200, 1000)
        U = [0]
        U_set = {0}
        diff_counts = {}
        
        for _ in range(n_target - 1):
            found = False
            for _ in range(500):
                cand = random.randint(1, max_val)
                if cand in U_set:
                    continue
                ok = True
                for u in U_set:
                    if (cand - u) in diff_counts or (u - cand) in diff_counts:
                        ok = False
                        break
                if ok:
                    U.append(cand)
                    U_set.add(cand)
                    for u in U_set:
                        if u == cand: continue
                        diff_counts[cand - u] = 1
                        diff_counts[u - cand] = 1
                    found = True
                    break
            if not found:
                break
        if len(U) >= 5:
            candidates.append(U)
            
    # 3. Structured sets
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127, 255])
    candidates.append([0, 1, 4, 16, 64, 256, 1024])
    
    # Run local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        # Run local search with more iterations for better results
        searched_U, searched_score = local_search(U, max_iters=2000, max_val=2000)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
            
    # Final polish: run local search on the best found
    final_U, final_score = local_search(best_U, max_iters=3000, max_val=3000)
    if final_score > best_score:
        best_U = final_U
        best_score = final_score
        
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
