import time
import math
import random
import numpy as np

def _score_set_numpy(U):
    """Faster scoring using numpy bit operations."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Sumset via numpy
    U_arr = np.array(U, dtype=np.int64)
    
    # For n up to ~200, O(n^2) with numpy vectorization is fast
    # Sumset: compute all a+b
    # Use broadcasting for small n
    if n <= 200:
        # Vectorized: create matrix of all sums
        # Memory: n*n which for n=200 is 40000, fine
        sums_matrix = U_arr[:, None] + U_arr[None, :]
        P = len(np.unique(sums_matrix))
        
        diffs_matrix = U_arr[:, None] - U_arr[None, :]
        D = len(np.unique(diffs_matrix))
    else:
        # Fallback for larger n
        sums_set = set()
        diffs_set = set()
        for a in U:
            for b in U:
                sums_set.add(a + b)
                diffs_set.add(a - b)
        P = len(sums_set)
        D = len(diffs_set)
    
    if P == 0:
        return 0.0
    ratio = D / P
    if ratio <= 1:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Greedy construction with different penalty parameters
    # For each lambda, build a set greedily maximizing (new_diffs - lambda * new_sums)
    
    lambda_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    for lam in lambda_values:
        if time.time() > deadline - 10:
            break
        
        U = [0]
        # Start with a small seed
        U = [0, 1, 3]
        
        current_sums = set()
        current_diffs = set()
        for a in U:
            for b in U:
                current_sums.add(a + b)
                current_diffs.add(a - b)
        
        current_score = _score_set_numpy(U)
        if current_score > best_score:
            best_score = current_score
            best_set = U[:]
        
        # Greedily add elements
        max_u = max(U)
        for step in range(50):
            if time.time() > deadline - 10:
                break
            
            # Candidate range: from 1 to max_u + some margin
            upper = max_u + max(10, max_u // 3)
            
            best_candidate = None
            best_gain = -float('inf')
            
            # Sample candidates to keep this fast
            candidates = set()
            # Near existing elements
            for u in U:
                for offset in range(1, max(5, max_u // 10) + 1):
                    c = u + offset
                    if 1 <= c <= upper and c not in U:
                        candidates.add(c)
                    c = u - offset
                    if 1 <= c <= upper and c not in U:
                        candidates.add(c)
            # Random candidates
            for _ in range(50):
                c = random.randint(1, upper)
                if c not in U:
                    candidates.add(c)
            # Some structured candidates
            for _ in range(20):
                c = random.randint(1, upper)
                candidates.add(c)
            
            for cand in candidates:
                if cand in U:
                    continue
                
                # Compute marginal gain
                new_sums_count = 0
                new_diffs_count = 0
                
                for a in U:
                    # New sums: cand + a
                    s = cand + a
                    if s not in current_sums:
                        new_sums_count += 1
                    # New diffs: cand - a and a - cand
                    d1 = cand - a
                    if d1 not in current_diffs:
                        new_diffs_count += 1
                    d2 = a - cand
                    if d2 not in current_diffs:
                        new_diffs_count += 1
                
                # Also cand + cand and cand - cand = 0
                s2 = cand + cand
                if s2 not in current_sums:
                    new_sums_count += 1
                # cand - cand = 0, already in diffs since 0-0=0
                
                gain = new_diffs_count - lam * new_sums_count
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            
            if best_candidate is None or best_gain < 0:
                break
            
            # Add the best candidate
            U.append(best_candidate)
            U = sorted(set(U))
            
            # Update sums and diffs
            for a in U:
                for b in U:
                    current_sums.add(a + b)
                    current_diffs.add(a - b)
            
            new_score = _score_set_numpy(U)
            if new_score > best_score:
                best_score = new_score
                best_set = U[:]
            
            max_u = max(U)
        
        # Phase 2: Local refinement of the greedily constructed set
        if time.time() > deadline - 5:
            break
        
        current_U = U[:]
        current_score = _score_set_numpy(current_U)
        
        temp = 0.3
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            
            new_U = list(current_U)
            move_type = random.random()
            
            if move_type < 0.4 and len(current_U) > 2:
                # Replace an element
                removable = [x for x in current_U if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    max_val = max(current_U)
                    new_val = max(1, old_val + random.randint(-max(3, old_val//5), max(3, old_val//5)))
                    new_U.remove(old_val)
                    if new_val not in new_U:
                        new_U.append(new_val)
            elif move_type < 0.7 and len(current_U) < 80:
                # Add element
                max_val = max(current_U)
                base = random.choice(current_U)
                new_val = base + random.randint(1, max(5, max_val // 10))
                if new_val not in new_U:
                    new_U.append(new_val)
            else:
                # Remove element
                removable = [x for x in current_U if x != 0]
                if removable:
                    new_U.remove(random.choice(removable))
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
            
            new_score = _score_set_numpy(new_U)
            
            if new_score >= current_score:
                current_score = new_score
                current_U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_U[:]
            else:
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = math.exp(delta / max(temp, 1e-10))
                    if random.random() < accept_prob:
                        current_score = new_score
                        current_U = new_U
            
            temp *= 0.999
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = current_U[:]
    
    # Phase 3: Final polish with very fine-tuned local search
    if time.time() < deadline - 3:
        current_U = best_set[:]
        current_score = _score_set_numpy(current_U)
        
        temp = 0.1
        for it in range(5000):
            if time.time() > deadline - 3:
                break
            
            new_U = list(current_U)
            move_type = random.random()
            
            if move_type < 0.5 and len(current_U) > 2:
                # Fine-tune: shift by ±1
                removable = [x for x in current_U if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    new_val = old_val + random.choice([-1, 1])
                    if new_val > 0 and new_val not in current_U:
                        new_U.remove(old_val)
                        new_U.append(new_val)
            elif move_type < 0.8 and len(current_U) < 100:
                # Add near existing
                max_val = max(current_U)
                base = random.choice(current_U)
                new_val = base + random.randint(1, max(2, max_val // 20))
                if new_val not in new_U:
                    new_U.append(new_val)
            else:
                # Remove
                removable = [x for x in current_U if x != 0]
                if removable:
                    new_U.remove(random.choice(removable))
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
            
            new_score = _score_set_numpy(new_U)
            
            if new_score >= current_score:
                current_score = new_score
                current_U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_U[:]
            else:
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = math.exp(delta / max(temp, 1e-10))
                    if random.random() < accept_prob:
                        current_score = new_score
                        current_U = new_U
            
            temp *= 0.9995
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_U[:]
    
    # Phase 4: High-precision final polish (from inspiration)
    if time.time() < deadline - 2:
        current_U = best_set[:]
        current_score = _score_set_numpy(current_U)
        best_score_local = current_score
        best_U_local = current_U[:]
        
        temp = 0.05
        cooling = 0.9998
        for it in range(10000):
            if time.time() > deadline - 2:
                break
            
            new_U = list(current_U)
            move_type = random.random()
            
            if move_type < 0.4 and len(current_U) > 2:
                # Fine-tune an element
                removable = [x for x in new_U if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    step = max(1, old_val // 20)
                    new_val = max(1, old_val + random.randint(-step, step))
                    new_U.remove(old_val)
                    if new_val not in new_U:
                        new_U.append(new_val)
            elif move_type < 0.7 and len(current_U) < 50:
                # Add element near existing ones
                max_val = max(current_U) if current_U else 0
                if current_U:
                    base = random.choice(current_U)
                    new_val = base + random.randint(1, max(3, max_val // 20))
                    if new_val not in new_U:
                        new_U.append(new_val)
            else:
                # Remove element
                removable = [x for x in new_U if x != 0]
                if removable:
                    idx = random.choice(removable)
                    new_U.remove(idx)
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
            
            new_score = _score_set_numpy(new_U)
            
            if new_score >= current_score:
                current_score = new_score
                current_U = new_U
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_U_local = current_U[:]
            else:
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = math.exp(delta / max(temp, 1e-10))
                    if random.random() < accept_prob:
                        current_score = new_score
                        current_U = new_U
            
            temp = max(0.01, temp * cooling)
        
        if best_score_local > best_score:
            best_score = best_score_local
            best_set = best_U_local[:]
    
    return sorted(best_set)
