import numpy as np
import time
import random
from itertools import product as iproduct

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe (90s limit)
    
    # 2. Objective Function - fast evaluation
    def evaluate(A):
        # A is a list of ints
        s_set = set()
        p_set = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                s_set.add(ai + aj)
                p_set.add(ai * aj)
        return max(len(s_set), len(p_set))
    
    def evaluate_delta(A, idx_old, old_val, new_val):
        """Compute new score after replacing A[idx_old] with new_val"""
        A_new = list(A)
        A_new[idx_old] = new_val
        return evaluate(A_new)
    
    # Strategy: Model set as union of structured sub-sets
    # - Arithmetic progression: a, a+d, a+2d, ..., a+(k-1)d for sumset collisions
    # - Geometric progression: b*r^0, b*r^1, ..., b*r^(m-1) for product-set collisions
    
    best_score = float('inf')
    best_A = None
    
    # Helper to build a candidate set from AP + GP parameters
    def build_from_params(ap_start, ap_step, ap_len, gp_base, gp_ratio, gp_len, extra_count):
        """
        Build a set from:
        - AP: ap_start, ap_start+ap_step, ..., ap_start+(ap_len-1)*ap_step
        - GP: gp_base, gp_base*gp_ratio, ..., gp_base*gp_ratio^(gp_len-1)
        - extras: fill remaining slots with smooth numbers
        """
        ap_terms = set()
        for i in range(ap_len):
            val = ap_start + i * ap_step
            if 1 <= val <= 10000:
                ap_terms.add(val)
        
        gp_terms = set()
        val = gp_base
        for i in range(gp_len):
            if 1 <= val <= 10000:
                gp_terms.add(val)
            else:
                break
            val *= gp_ratio
        
        # Combine and ensure distinctness
        combined = ap_terms | gp_terms
        if len(combined) > 20:
            # Need to reduce - take first 20 sorted
            combined = set(sorted(combined)[:20])
        
        # Add extras if needed
        if len(combined) < 20:
            # Use small smooth numbers not in combined
            smooth_pool = []
            for x in range(1, 10001):
                if x not in combined:
                    # Check if x is 5-smooth
                    temp = x
                    for p in [2, 3, 5]:
                        while temp % p == 0:
                            temp //= p
                    if temp == 1:
                        smooth_pool.append(x)
                    if len(smooth_pool) >= 20 - len(combined):
                        break
            
            for x in smooth_pool[:20 - len(combined)]:
                combined.add(x)
        
        if len(combined) != 20:
            # Fallback: just use smallest 20
            return sorted(list(combined)[:20])
        
        return sorted(combined)
    
    # Generate candidate parameter combinations
    candidate_sets = []
    
    # AP lengths: 5-12, GP lengths: 5-12, total should be ~20
    for ap_len in range(4, 13):
        for gp_len in range(4, 13):
            if ap_len + gp_len > 22:  # Allow some overlap
                continue
            if ap_len + gp_len < 10:
                continue
            
            # AP parameters
            for ap_start in range(1, 50):
                for ap_step in range(1, 20):
                    # GP parameters
                    for gp_base in range(1, 30):
                        for gp_ratio in [2, 3, 4, 5, 6]:
                            A = build_from_params(ap_start, ap_step, ap_len, gp_base, gp_ratio, gp_len, 0)
                            if len(set(A)) == 20 and all(1 <= x <= 10000 for x in A):
                                candidate_sets.append(A)
            
            if len(candidate_sets) > 5000:
                break
        if len(candidate_sets) > 5000:
            break
    
    # Also add some pure AP and pure GP candidates
    for start in range(1, 30):
        for step in range(1, 15):
            A = [start + i * step for i in range(20)]
            if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                candidate_sets.append(sorted(A))
    
    for base in range(1, 20):
        for ratio in [2, 3, 4]:
            A = []
            val = base
            for i in range(20):
                if val <= 10000:
                    A.append(val)
                else:
                    break
                val *= ratio
            if len(A) == 20 and len(set(A)) == 20:
                candidate_sets.append(sorted(A))
    
    # Evaluate all candidates and keep top ones
    scored_candidates = []
    for A in candidate_sets:
        score = evaluate(A)
        scored_candidates.append((score, A))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Initialize with top candidates
    for score, A in scored_candidates[:10]:
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    # Now do local search on the best candidates
    # Use a pool of good replacement values
    # Generate a broader pool for local search
    pool_set = set()
    for x in range(1, 10001):
        temp = x
        for p in [2, 3, 5, 7]:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            pool_set.add(x)
    pool = sorted(pool_set)
    
    # Local search with simulated annealing
    if best_A is not None:
        current_A = list(best_A)
        current_score = best_score
        
        temp = 50.0
        cooling_rate = 0.9998
        no_improve = 0
        max_no_improve = 5000
        
        while temp > 0.5 and no_improve < max_no_improve:
            if time.time() > deadline:
                break
            
            # Generate neighbor: replace one element
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            
            # Choose new value from pool, biased towards values that create collisions
            # Try values close to existing ones for sum collisions
            candidates = []
            for _ in range(20):
                new_val = random.choice(pool)
                if new_val != old_val and new_val not in current_A:
                    candidates.append(new_val)
            
            if not candidates:
                continue
            
            new_val = random.choice(candidates)
            
            # Apply move
            current_A[idx] = new_val
            new_score = evaluate(current_A)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_A[idx] = old_val
                no_improve += 1
            
            temp *= cooling_rate
    
    return best_A
