"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        # Compute sumset and difference set efficiently
        sumset = set()
        diffset = set()
        for i in range(n):
            a = U[i]
            for j in range(n):
                b = U[j]
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    # Phase 1: Try known good structured sets
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52],
    ]
    
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)
    
    # Scaled variants
    for gr in golomb_rulers[:6]:
        for scale in [2, 3, 4, 5, 6, 7, 8]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Local search with swap/add/remove moves from diverse seeds
    def local_search(initial_set, max_time_remaining=10):
        """Perform local search starting from initial_set."""
        current = sorted(set(initial_set))
        if 0 not in current or len(current) < 3:
            return current, compute_score(current)
        
        current_score = compute_score(current)
        start_ls = time.time()
        no_improve = 0
        max_no_improve = 300
        
        while no_improve < max_no_improve:
            if time.time() - start_ls > max_time_remaining:
                break
            
            improved = False
            max_cur = max(current)
            
            # Try adding an element
            if len(current) < 20:
                candidates_to_add = set()
                for el in current:
                    for delta in range(-5, 6):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 15):
                    candidates_to_add.add(new_el)
                # Also try elements in gaps
                for i in range(len(current) - 1):
                    gap = current[i+1] - current[i]
                    if gap > 2:
                        for new_el in range(current[i] + 1, current[i+1]):
                            candidates_to_add.add(new_el)
                
                for new_el in sorted(candidates_to_add):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    # Try a wider range of replacements
                    for new_el in range(max(1, old_el - 10), min(max_cur + 20, old_el + 11)):
                        if new_el == old_el or new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != len(current):
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element (if set is large)
            if len(current) > 4:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
        
        return current, current_score
    
    # Run local search from multiple diverse seeds
    seeds = []
    
    # Golomb rulers as seeds
    for gr in golomb_rulers:
        seeds.append(list(gr))
    
    # Powers of 2
    for n in range(3, 12):
        powers = [2**i for i in range(n)]
        seeds.append(powers)
    
    # Fibonacci-like
    fib = [0, 1]
    for _ in range(15):
        fib.append(fib[-1] + fib[-2])
    for n in range(3, 12):
        seeds.append(fib[:n])
    
    # Arithmetic progressions with various parameters
    for start in [1, 2, 3, 5]:
        for step in [1, 2, 3, 4, 5]:
            ap = [0]
            cur = start
            while cur <= 100 and len(ap) < 12:
                ap.append(cur)
                cur += step
            seeds.append(ap)
    
    # Random Sidon-like sets
    random.seed(42)
    for _ in range(10):
        size = random.randint(4, 10)
        max_u = random.randint(20, 80)
        # Greedy Sidon set construction
        sidon = [0]
        used_sums = {0}
        for _ in range(size - 1):
            best_next = -1
            for candidate in range(1, max_u + 1):
                if candidate in sidon:
                    continue
                valid = True
                for existing in sidon:
                    s = candidate + existing
                    if s in used_sums:
                        valid = False
                        break
                if valid:
                    best_next = candidate
                    break
            if best_next == -1:
                break
            sidon.append(best_next)
            for existing in sidon[:-1]:
                used_sums.add(best_next + existing)
        if len(sidon) >= 3:
            seeds.append(sorted(sidon))
    
    # Run local search from each seed
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        time_remaining = min(10, deadline - 5 - time.time())
        if time_remaining <= 0:
            break
        
        result_set, result_score = local_search(seed, max_time_remaining=time_remaining)
        
        if result_score > best_score:
            best_score = result_score
            best_set = list(result_set)
    
    # Phase 3: Final polish on the best set found
    if time.time() < deadline - 5:
        time_remaining = deadline - 5 - time.time()
        if time_remaining > 0:
            result_set, result_score = local_search(best_set, max_time_remaining=time_remaining)
            if result_score > best_score:
                best_score = result_score
                best_set = list(result_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
