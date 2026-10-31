"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
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

    def is_golomb(U):
        """Check if U is a Golomb ruler (all pairwise differences are unique)."""
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal Golomb rulers
    optimal_golomb = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }

    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    # Phase 1: Try known good structured sets (Golomb rulers and scaled variants)
    for n in range(3, 13):
        if n in optimal_golomb:
            gr = optimal_golomb[n]
            s = compute_score(gr)
            if s > best_score:
                best_score = s
                best_set = list(gr)
    
    # Scaled variants of small Golomb rulers
    for n in range(3, 9):
        if n in optimal_golomb:
            gr = optimal_golomb[n]
            for scale in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                scaled = [x * scale for x in gr]
                s = compute_score(scaled)
                if s > best_score:
                    best_score = s
                    best_set = list(scaled)
    
    # Phase 2: Local search with geometric insights
    # The key insight: maximize |U-U| (distinct differences) while minimizing |U+U| (distinct sums)
    # Golomb rulers maximize |U-U| = n(n-1)/2 + n (including 0)
    # But we can sometimes do better by allowing some sum collisions while keeping differences distinct
    
    def local_search(initial_set, max_time_remaining=10):
        """Perform local search starting from initial_set."""
        current = sorted(set(initial_set))
        if 0 not in current or len(current) < 3:
            return current, compute_score(current)
        
        current_score = compute_score(current)
        start_ls = time.time()
        no_improve = 0
        max_no_improve = 500
        rng = random.Random(42)
        
        while no_improve < max_no_improve:
            if time.time() - start_ls > max_time_remaining:
                break
            if time.time() > deadline - 3:
                break
            
            improved = False
            max_cur = max(current)
            n = len(current)
            
            # Try adding an element (if set is small)
            if n < 15:
                candidates_to_add = set()
                for el in current:
                    for delta in range(-3, 4):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 8):
                    candidates_to_add.add(new_el)
                # Also try elements in gaps
                for i in range(len(current) - 1):
                    gap = current[i+1] - current[i]
                    if gap > 2:
                        for new_el in range(current[i] + 1, current[i+1]):
                            candidates_to_add.add(new_el)
                
                # Try each candidate
                for new_el in sorted(candidates_to_add):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        improved = True
                        break
            
            # Try replacing an element with small shifts
            if n >= 3:
                max_cur = max(current)
                # Shuffle indices to avoid bias
                indices = list(range(1, n))
                rng.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline - 3:
                        break
                    old_el = current[i]
                    # Try small shifts
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_el = old_el + delta
                        if new_el < 1:
                            continue
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != n:
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element (if set is large enough)
            if n > 5:
                for i in range(1, n):
                    if time.time() > deadline - 3:
                        break
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
                # Occasional random perturbation to escape local optima
                if rng.random() < 0.15 and n > 3:
                    idx = rng.randint(1, n-1)
                    new_val = max(1, current[idx] + rng.randint(-4, 4))
                    if new_val not in current:
                        cand = list(current)
                        cand[idx] = new_val
                        cand = sorted(set(cand))
                        if 0 in cand and len(cand) == n:
                            s = compute_score(cand)
                            if s > current_score - 0.001:  # Accept slightly worse
                                current = list(cand)
                                current_score = s
            else:
                no_improve = 0
        
        return current, current_score
    
    # Run local search from multiple diverse seeds
    seeds = []
    
    # Golomb rulers as seeds
    for n in range(3, 13):
        if n in optimal_golomb:
            seeds.append(list(optimal_golomb[n]))
    
    # Powers of 2
    for n in range(3, 11):
        powers = [2**i for i in range(n)]
        seeds.append(powers)
    
    # Fibonacci-like
    fib = [0, 1]
    for _ in range(14):
        fib.append(fib[-1] + fib[-2])
    for n in range(3, 11):
        seeds.append(fib[:n])
    
    # Arithmetic progressions with various parameters
    for start in [1, 2, 3, 5]:
        for step in [1, 2, 3, 4, 5]:
            ap = [0]
            cur = start
            while cur <= 80 and len(ap) < 11:
                ap.append(cur)
                cur += step
            seeds.append(ap)
    
    # Random Sidon-like sets
    random.seed(42)
    for _ in range(12):
        size = random.randint(4, 10)
        max_u = random.randint(20, 70)
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
        
        time_remaining = min(8, deadline - 5 - time.time())
        if time_remaining <= 0:
            break
        
        result_set, result_score = local_search(seed, max_time_remaining=time_remaining)
        
        if result_score > best_score:
            best_score = result_score
            best_set = list(result_set)
    
    # Phase 3: Final polish on the best set found
    if time.time() < deadline - 3:
        time_remaining = deadline - 3 - time.time()
        if time_remaining > 0:
            result_set, result_score = local_search(best_set, max_time_remaining=time_remaining)
            if result_score > best_score:
                best_score = result_score
                best_set = list(result_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
