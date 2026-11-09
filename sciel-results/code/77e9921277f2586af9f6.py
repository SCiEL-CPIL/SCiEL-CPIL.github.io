"""Program for integer sum-product extremal sets using structured sub-set union approach."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy for speed."""
        A = np.array(sorted(A), dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        prods = A[i_idx] * A[j_idx]
        unique_sums = np.unique(sums)
        unique_prods = np.unique(prods)
        return max(len(unique_sums), len(unique_prods))
    
    best_set = None
    best_score = float('inf')
    
    def update_best(cand):
        nonlocal best_set, best_score
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            return False
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = list(sorted(cand))
            return True
        return False
    
    # Phase 1: Enumerate structured candidates (AP + GP unions)
    candidates = []
    
    # 1. Pure geometric progressions
    for start in range(1, 15):
        for ratio in range(2, 6):
            cand = []
            val = start
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val *= ratio
            if len(cand) == n:
                candidates.append(sorted(cand))
    
    # 2. Pure arithmetic progressions
    for start in range(1, 15):
        for step in range(1, 30):
            cand = [start + i * step for i in range(n)]
            if max(cand) <= max_val:
                candidates.append(sorted(cand))
    
    # 3. Union of short AP and short GP (key structured approach)
    # AP part: a, a+d, a+2d, ..., a+(k-1)d
    # GP part: b, b*r, b*r^2, ..., b*r^(n-k-1)
    # Try various splits
    for k in range(5, 16):  # AP length k, GP length n-k
        gp_len = n - k
        if gp_len < 3:
            continue
        
        for a in range(1, 20):
            for d in range(1, 15):
                ap_part = [a + i * d for i in range(k)]
                if max(ap_part) > max_val:
                    continue
                
                for b in range(1, 20):
                    for r in range(2, 8):
                        gp_part = []
                        val = b
                        for i in range(gp_len):
                            if val > max_val:
                                break
                            gp_part.append(val)
                            val *= r
                        if len(gp_part) != gp_len:
                            continue
                        
                        # Combine and check for uniqueness
                        cand = ap_part + gp_part
                        if len(set(cand)) == n and max(cand) <= max_val and min(cand) >= 1:
                            candidates.append(sorted(cand))
    
    # 4. Two interleaved APs with different steps
    for a1 in range(1, 10):
        for d1 in range(1, 8):
            for a2 in range(1, 10):
                for d2 in range(1, 8):
                    if d1 == d2 and a1 == a2:
                        continue
                    cand = []
                    for i in range(n):
                        if i % 2 == 0:
                            val = a1 + (i // 2) * d1
                        else:
                            val = a2 + (i // 2) * d2
                        if 1 <= val <= max_val:
                            cand.append(val)
                        else:
                            cand = None
                            break
                    if cand and len(set(cand)) == n:
                        candidates.append(sorted(cand))
    
    # 5. Logarithmic-spaced sets (approximate GP with integer rounding)
    for base in [1, 2, 3, 4, 5]:
        for growth in [1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
            cand = []
            val = base
            for i in range(n):
                if val > max_val:
                    break
                cand.append(int(round(val)))
                val *= growth
            if len(cand) == n and len(set(cand)) == n and max(cand) <= max_val:
                candidates.append(sorted(cand))
    
    # 6. Mixed: first half AP, second half GP
    for a in range(1, 15):
        for d in range(1, 12):
            ap_part = [a + i * d for i in range(10)]
            if max(ap_part) > max_val:
                continue
            for b in range(1, 15):
                for r in range(2, 7):
                    gp_part = []
                    val = b
                    for i in range(10):
                        if val > max_val:
                            break
                        gp_part.append(val)
                        val *= r
                    if len(gp_part) != 10:
                        continue
                    cand = ap_part + gp_part
                    if len(set(cand)) == n and max(cand) <= max_val and min(cand) >= 1:
                        candidates.append(sorted(cand))
    
    # Evaluate all structured candidates
    for cand in candidates:
        if update_best(cand):
            pass  # Updated
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
    
    # Phase 2: Local search from best structured candidate
    # Use targeted moves that preserve structure
    current_set = list(best_set)
    current_score = best_score
    
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.3
    
    # Precompute indices for fast evaluation
    i_idx, j_idx = np.triu_indices(n)
    
    def evaluate_fast(A_list):
        A = np.array(sorted(A_list), dtype=np.int64)
        sums = A[i_idx] + A[j_idx]
        prods = A[i_idx] * A[j_idx]
        return max(len(np.unique(sums)), len(np.unique(prods)))
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
        
        new_set = list(current_set)
        move_type = random.random()
        
        if move_type < 0.35:
            # Nudge one element slightly (preserve local structure)
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            for _ in range(10):
                delta = random.randint(-8, 8)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if 1 <= new_val <= max_val and new_val not in new_set:
                    new_set[idx] = new_val
                    break
            if new_set == current_set:
                continue
                
        elif move_type < 0.55:
            # Swap two elements (preserve multiset of values)
            i, j = random.sample(range(n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            
        elif move_type < 0.75:
            # Replace one element with nearby value (bias towards small)
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            if random.random() < 0.6:
                new_val = random.randint(1, 80)
            else:
                new_val = random.randint(1, 400)
            attempts = 0
            while (new_val == old_val or new_val in new_set) and attempts < 15:
                if random.random() < 0.6:
                    new_val = random.randint(1, 80)
                else:
                    new_val = random.randint(1, 400)
                attempts += 1
            if new_val == old_val or new_val in new_set:
                continue
            new_set[idx] = new_val
            
        elif move_type < 0.9:
            # Structured move: shift a contiguous block
            block_len = random.randint(2, 5)
            start_idx = random.randint(0, n - block_len)
            shift = random.randint(-5, 5)
            if shift == 0:
                continue
            valid = True
            for i in range(start_idx, start_idx + block_len):
                new_val = new_set[i] + shift
                if new_val < 1 or new_val > max_val:
                    valid = False
                    break
            if not valid:
                continue
            # Check no collisions with other elements
            for i in range(start_idx, start_idx + block_len):
                new_val = new_set[i] + shift
                for j in range(n):
                    if j != i and new_set[j] == new_val:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            for i in range(start_idx, start_idx + block_len):
                new_set[i] += shift
        else:
            # Occasional random restart of one element to explore
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            new_val = random.randint(1, 500)
            attempts = 0
            while (new_val == old_val or new_val in new_set) and attempts < 20:
                new_val = random.randint(1, 500)
                attempts += 1
            if new_val == old_val or new_val in new_set:
                continue
            new_set[idx] = new_val
        
        new_score = evaluate_fast(new_set)
        
        delta_score = new_score - current_score
        if delta_score < 0:
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            if random.random() < math.exp(-delta_score / T):
                current_set = new_set
                current_score = new_score
        
        T *= cooling_rate
    
    return best_set
# EVOLVE-BLOCK-END
