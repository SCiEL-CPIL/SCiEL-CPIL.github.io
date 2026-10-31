"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

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
        # For small n, use set-based approach
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
    
    def is_golomb_ruler(U):
        """Check if all pairwise differences are distinct (up to sign)."""
        U = sorted(U)
        n = len(U)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    # Phase 1: Try known good structured sets (Golomb rulers, powers, etc.)
    # Known optimal Golomb rulers
    golomb_rulers = [
        [0, 1, 3],           # size 3, max 3
        [0, 1, 4, 6],        # size 4, max 6
        [0, 1, 4, 10, 12],   # size 5, max 12
        [0, 1, 4, 10, 12, 17], # size 6, max 17
        [0, 1, 4, 10, 18, 23, 25], # size 7, max 25
        [0, 1, 4, 10, 18, 23, 25, 34], # size 8, max 34
        [0, 1, 4, 10, 18, 23, 25, 34, 42], # size 9, max 42
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52], # size 10, max 52
    ]
    
    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)
    
    # Also try scaled/shifted variants of Golomb rulers
    for gr in golomb_rulers[:6]:
        for scale in [2, 3, 4, 5]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)
    
    # Phase 2: Systematic search over small sets
    # For each size and max, try structured candidates
    for max_u in range(1, 50):
        if time.time() > deadline - 5:
            break
        
        for size in range(3, min(12, max_u + 1)):
            if time.time() > deadline - 5:
                break
            
            candidates = []
            
            # Golomb ruler candidates (try to construct)
            # Simple heuristic: greedy construction
            if size >= 3:
                # Try different starting points and growth patterns
                for start2 in [1, 2, 3, 4]:
                    if start2 >= max_u:
                        continue
                    gr_candidate = [0, start2]
                    used_diffs = {start2}
                    for _ in range(size - 2):
                        # Try to add next element that maintains Golomb property
                        best_next = -1
                        for candidate in range(gr_candidate[-1] + 1, max_u + 1):
                            valid = True
                            for existing in gr_candidate:
                                d = candidate - existing
                                if d in used_diffs:
                                    valid = False
                                    break
                            if valid:
                                best_next = candidate
                                break
                        if best_next == -1:
                            break
                        gr_candidate.append(best_next)
                        for existing in gr_candidate[:-1]:
                            used_diffs.add(gr_candidate[-1] - existing)
                    if len(gr_candidate) == size:
                        candidates.append(gr_candidate)
            
            # Powers of 2
            if max_u >= 2 ** (size - 1):
                geo = [2 ** i for i in range(size)]
                if geo[-1] <= max_u:
                    candidates.append(geo)
            
            # Arithmetic progression
            if (max_u) % (size - 1) == 0:
                step = max_u // (size - 1)
                if step >= 1:
                    ap = [i * step for i in range(size)]
                    candidates.append(ap)
            
            # Fibonacci-like
            fib = [0, 1]
            while len(fib) < size:
                fib.append(fib[-1] + fib[-2])
            if fib[-1] <= max_u:
                candidates.append(fib[:size])
            
            # Evaluate unique candidates
            seen = set()
            for cand in candidates:
                cand = sorted(set(cand))
                if len(cand) != size or 0 not in cand:
                    continue
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
    
    # Phase 3: Local search around best set found
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 500
        
        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            improved = False
            
            # Try adding an element
            if len(current) < 15:
                max_cur = max(current)
                # Try elements near existing ones and some spread
                candidates_to_add = set()
                for el in current:
                    for delta in range(-2, 3):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates_to_add.add(new_el)
                # Also try some larger elements
                for new_el in range(max_cur + 1, max_cur + 10):
                    candidates_to_add.add(new_el)
                
                for new_el in sorted(candidates_to_add):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    # Try nearby replacements
                    for new_el in range(max(1, old_el - 3), min(max_cur + 5, old_el + 4)):
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
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
    
    # Phase 4: Random restarts with Golomb-ruler-inspired initialization
    if time.time() < deadline - 5:
        random.seed(12345)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start from a random Golomb-ruler-like set
            size = random.randint(4, 10)
            max_u = random.randint(10, 60)
            
            # Greedy Golomb ruler construction
            gr = [0, 1]
            used_diffs = {1}
            for _ in range(size - 2):
                best_next = -1
                for candidate in range(2, max_u + 1):
                    valid = True
                    for existing in gr:
                        d = candidate - existing
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        best_next = candidate
                        break
                if best_next == -1:
                    break
                gr.append(best_next)
                for existing in gr[:-1]:
                    used_diffs.add(gr[-1] - existing)
            
            if len(gr) >= 3:
                s = compute_score(gr)
                if s > best_score:
                    best_score = s
                    best_set = list(gr)
    
    return list(best_set)
# EVOLVE-BLOCK-END
