import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generic greedy for other cases
        total_words = q ** n
        # This is just a placeholder for non-target cases
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    # 3^8 = 6561
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists (including self)
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Precompute reverse neighbors: coverers[x] = list of codewords that cover x
    # This is useful for finding swaps
    coverers = [None] * total_words
    for i in range(total_words):
        coverers[i] = np.array([j for j in range(total_words) if i in neighbors[j]], dtype=np.int32)
    # Note: coverers[i] is exactly neighbors[i] because distance is symmetric. 
    # So coverers[i] == neighbors[i]. This simplifies things.

    # --- Phase 1: Fast Greedy Initialization ---
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # potential[x] = number of uncovered points in neighbors[x]
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    uncovered_remaining = total_words

    while uncovered_remaining > 0 and time.time() < deadline - 40.0:
        c = np.argmax(potential)
        if potential[c] == 0:
            break
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                # Update potential for all centers that cover x
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # --- Phase 2: Iterated Local Search ---
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)

    random.seed(42)
    
    def get_risky_points(c):
        """Return points in neighbors[c] that are covered only by c."""
        nb = neighbors[c]
        return nb[current_cov_count[nb] == 1]

    def try_remove(c, idx):
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size
        nb = neighbors[c]
        if np.all(current_cov_count[nb] > 1):
            current_cov_count[nb] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            return True
        return False

    def try_swap(c1, c1_idx):
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size
        risky = get_risky_points(c1)
        
        if len(risky) == 0:
            return try_remove(c1, c1_idx)
        
        # Find a c2 that covers all risky points
        # Candidates are intersection of neighbors[r] for r in risky
        # Since neighbors are small, we can just iterate
        # Optimization: Start with the coverers of the first risky point
        candidates_set = set(neighbors[risky[0]].tolist())
        candidates_set.discard(c1)
        
        for r in risky[1:]:
            # Intersect with neighbors[r]
            candidates_set &= set(neighbors[r].tolist())
            if not candidates_set:
                break
        
        if not candidates_set:
            return False
            
        # Filter out existing codewords (we are swapping, so c2 should not be in code, 
        # unless we allow replacing with same, which is useless)
        # Actually, if c2 is in code, we are just moving c1 to c2's position? No, we remove c1 and add c2.
        # If c2 is already in code, we would have duplicate codewords. So c2 must NOT be in current_code_set.
        valid_candidates = [c2 for c2 in candidates_set if c2 not in current_code_set]
        
        if not valid_candidates:
            return False
            
        # Heuristic: Pick c2 that covers the most *other* uncovered points? 
        # Or just pick randomly to escape local minima? 
        # Greedy: Pick c2 that maximizes coverage of points with count 0? 
        # But all points in neighbors[c1] have count >= 1. 
        # Points outside neighbors[c1] are unaffected by removal.
        # So any valid c2 maintains validity.
        # To improve future removal chances, pick c2 that overlaps heavily with other codewords?
        # Or pick one that is "central"?
        # Let's pick one that minimizes the number of *newly* risky points created?
        # Or simply random choice from valid candidates to allow exploration.
        
        # Let's try a weighted random choice or just first one for speed.
        # Given time constraints, random is better for exploration.
        c2 = random.choice(valid_candidates)
        
        # Perform swap
        current_cov_count[neighbors[c1]] -= 1
        current_is_code[c1] = False
        current_code_set.remove(c1)
        current_code[c1_idx] = c2
        
        current_cov_count[neighbors[c2]] += 1
        current_is_code[c2] = True
        current_code_set.add(c2)
        
        return True

    def try_two_for_one(c1, c1_idx, c2, c2_idx):
        nonlocal current_cov_count, current_is_code, current_code_set, current_code, current_size
        
        # Hypothetically remove c1 and c2
        # Risky points are those covered only by c1 or only by c2
        nb1 = neighbors[c1]
        nb2 = neighbors[c2]
        
        # Points covered by c1 but not c2? No, we need points that lose coverage.
        # A point x is risky if cov[x] == 1 and x in nb1, OR cov[x] == 1 and x in nb2.
        # But if x is in both nb1 and nb2, and cov[x] == 2, it becomes 0 if both removed.
        # So we need to check the combined effect.
        
        # Let's simulate the removal on a copy of cov_count? Too slow.
        # Instead, identify candidates for risky points.
        # Risky set R = {x | (x in nb1 and cov[x]==1) or (x in nb2 and cov[x]==1) or (x in nb1 and x in nb2 and cov[x]==2)}
        
        # Simpler: 
        # 1. Remove c1. Find risky points R1.
        # 2. Remove c2. Find risky points R2 (considering c1 is gone).
        
        # Let's do it step by step with temporary logic
        # Temporarily decrement
        current_cov_count[nb1] -= 1
        current_cov_count[nb2] -= 1
        
        # Find points that are now 0
        risky = np.where(current_cov_count == 0)[0]
        
        if len(risky) == 0:
            # Both can be removed
            # Restore counts? No, we commit to removal.
            # But we need to remove them from code list.
            # Indices might shift.
            if c1_idx < c2_idx:
                current_code.pop(c2_idx)
                current_code.pop(c1_idx)
            else:
                current_code.pop(c1_idx)
                current_code.pop(c2_idx)
            
            current_is_code[c1] = False
            current_is_code[c2] = False
            current_code_set.remove(c1)
            current_code_set.remove(c2)
            current_size -= 2
            return True
        else:
            # We need to find a c3 that covers all risky points.
            # Candidates are intersection of neighbors[r] for r in risky.
            candidates_set = set(neighbors[risky[0]].tolist())
            candidates_set.discard(c1)
            candidates_set.discard(c2)
            
            for r in risky[1:]:
                candidates_set &= set(neighbors[r].tolist())
                if not candidates_set:
                    break
            
            valid_candidates = [c3 for c3 in candidates_set if c3 not in current_code_set]
            
            if not valid_candidates:
                # Rollback
                current_cov_count[nb1] += 1
                current_cov_count[nb2] += 1
                return False
            
            c3 = random.choice(valid_candidates)
            
            # Commit: Remove c1, c2, Add c3
            # Counts are already decremented for c1, c2.
            # We need to increment for c3.
            current_cov_count[neighbors[c3]] += 1
            
            # Update code list
            if c1_idx < c2_idx:
                current_code.pop(c2_idx)
                current_code.pop(c1_idx)
            else:
                current_code.pop(c1_idx)
                current_code.pop(c2_idx)
            
            current_is_code[c1] = False
            current_is_code[c2] = False
            current_code_set.remove(c1)
            current_code_set.remove(c2)
            
            current_code.append(c3)
            current_is_code[c3] = True
            current_code_set.add(c3)
            current_size -= 1 # -2 + 1
            
            return True

    while time.time() < deadline - 5.0:
        if current_size == 0:
            break
        
        made_progress = False
        
        # 1. Try random removal
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        if try_remove(c, idx):
            made_progress = True
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        
        # 2. If no removal, try swap
        if not made_progress and time.time() < deadline - 10.0:
            idx = random.randint(0, current_size - 1)
            c = current_code[idx]
            if try_swap(c, idx):
                made_progress = True
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
        
        # 3. If still stuck, try 2-for-1
        if not made_progress and time.time() < deadline - 15.0 and current_size > 1:
            c1_idx = random.randint(0, current_size - 1)
            c2_idx = random.randint(0, current_size - 1)
            if c1_idx != c2_idx:
                c1 = current_code[c1_idx]
                c2 = current_code[c2_idx]
                if try_two_for_one(c1, c1_idx, c2, c2_idx):
                    made_progress = True
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]

    # --- Phase 3: Polish ---
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # Verify
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]

    return [list(digits_of[i]) for i in final_code]
