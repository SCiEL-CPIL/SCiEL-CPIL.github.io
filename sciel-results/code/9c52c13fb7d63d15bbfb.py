import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 113.0  # Safety margin (150s limit, use 113s)
    N = 32
    MAX_VAL = 512
    
    # Bitmask representation: bit i is set if (i+1) is in the set
    # So value v corresponds to bit (v-1)
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m
    
    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, 257):
            if d * 2 + 1 > MAX_VAL:
                break
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count
    
    def delta_score(mask, old_bit, new_bit):
        """Compute change in score when replacing old_bit with new_bit."""
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, 257):
            if d * 2 + 1 > MAX_VAL:
                break
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta
    
    def evaluate_set(S):
        m = list_to_mask(S)
        return score_mask(m)
    
    # Generate diverse initial candidates
    def gen_candidates():
        candidates = []
        
        # Consecutive integers
        candidates.append(list(range(1, 33)))
        
        # Arithmetic progressions
        for step in [2, 3, 4, 5, 6, 7, 8]:
            base = 1
            lst = [base + i * step for i in range(N)]
            if max(lst) <= MAX_VAL:
                candidates.append(lst)
        
        # Structured sets with small gaps
        for base in [1, 5, 10, 20, 50, 100]:
            lst = []
            for i in range(N):
                val = base + i * 3 + (i % 2)
                if 1 <= val <= MAX_VAL and val not in lst:
                    lst.append(val)
            while len(lst) < N:
                lst.append(random.randint(1, MAX_VAL))
            candidates.append(lst[:N])
        
        # Random sets
        for _ in range(20):
            candidates.append(random.sample(range(1, MAX_VAL + 1), N))
        
        # Sets based on quadratic residues or similar structures
        for mod in [7, 11, 13, 17]:
            lst = []
            for v in range(1, MAX_VAL + 1):
                if (v * v) % mod == 1 or (v * v) % mod == 0:
                    lst.append(v)
                if len(lst) >= N:
                    break
            while len(lst) < N:
                lst.append(random.randint(1, MAX_VAL))
            candidates.append(lst[:N])
        
        return candidates
    
    def local_search(mask, max_iter=500, time_limit=None):
        """Hill climbing using delta scoring."""
        s = score_mask(mask)
        s_set = set(mask_to_list(mask))
        
        for _ in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            elems = list(s_set)
            random.shuffle(elems)
            for old in elems:
                old_bit = 1 << (old - 1)
                best_new = None
                best_s = s
                # Try several random replacements
                for _ in range(12):
                    new = random.randint(1, MAX_VAL)
                    if new == old or new in s_set:
                        continue
                    new_bit = 1 << (new - 1)
                    delta = delta_score(mask, old_bit, new_bit)
                    new_s = s + delta
                    if new_s > best_s:
                        best_s = new_s
                        best_new = new
                if best_new is not None:
                    new_bit = 1 << (best_new - 1)
                    mask = mask ^ old_bit ^ new_bit
                    s_set.discard(old)
                    s_set.add(best_new)
                    s = best_s
                    improved = True
            if not improved:
                break
        return mask, s
    
    def targeted_local_search(mask, max_iter=200, time_limit=None):
        """Local search that also tries to complete 3-APs."""
        s = score_mask(mask)
        s_set = set(mask_to_list(mask))
        
        for _ in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            
            # Try targeted moves: for pairs in set, try to complete 3-AP
            elems = sorted(s_set)
            for i in range(len(elems)):
                a = elems[i]
                for j in range(i + 1, len(elems)):
                    b = elems[j]
                    d = b - a
                    c = a + 2 * d
                    if c > MAX_VAL:
                        break
                    if c not in s_set:
                        # Try removing some element and adding c
                        # Remove a random element that's not a or b
                        removable = [x for x in s_set if x != a and x != b]
                        if removable:
                            x_to_remove = random.choice(removable)
                            old_bit = 1 << (x_to_remove - 1)
                            new_bit = 1 << (c - 1)
                            delta = delta_score(mask, old_bit, new_bit)
                            if delta > 0:
                                mask = mask ^ old_bit ^ new_bit
                                s_set.discard(x_to_remove)
                                s_set.add(c)
                                s += delta
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                # Try standard random swaps
                for old in list(s_set):
                    old_bit = 1 << (old - 1)
                    for _ in range(8):
                        new = random.randint(1, MAX_VAL)
                        if new == old or new in s_set:
                            continue
                        new_bit = 1 << (new - 1)
                        delta = delta_score(mask, old_bit, new_bit)
                        if delta > 0:
                            mask = mask ^ old_bit ^ new_bit
                            s_set.discard(old)
                            s_set.add(new)
                            s += delta
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        return mask, s
    
    # Generate initial candidates and find best
    candidates = gen_candidates()
    best_mask = None
    best_score = 0
    
    for cand in candidates:
        m = list_to_mask(cand)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_mask = m
    
    print(f"Initial best: {best_score}", file=sys.stderr)
    
    # Multi-start local search
    num_starts = 8
    for start_i in range(num_starts):
        if time.time() > deadline:
            break
        
        if start_i == 0:
            current_mask = best_mask
            current_score = best_score
        else:
            # Perturb best
            current_mask = best_mask
            current_score = best_score
            s_set = set(mask_to_list(current_mask))
            num_swaps = random.randint(3, 7)
            for _ in range(num_swaps):
                old = random.choice(list(s_set))
                new = random.randint(1, MAX_VAL)
                if new not in s_set:
                    old_bit = 1 << (old - 1)
                    new_bit = 1 << (new - 1)
                    current_mask = current_mask ^ old_bit ^ new_bit
                    s_set.discard(old)
                    s_set.add(new)
            current_score = score_mask(current_mask)
        
        # Alternate between targeted and standard local search
        for phase in range(3):
            if time.time() > deadline - 5:
                break
            if phase % 2 == 0:
                current_mask, current_score = targeted_local_search(
                    current_mask, max_iter=300, time_limit=deadline - 5
                )
            else:
                current_mask, current_score = local_search(
                    current_mask, max_iter=500, time_limit=deadline - 5
                )
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                print(f"New best: {best_score} at start {start_i}, phase {phase}", file=sys.stderr)
    
    # Final polish: intensive local search on best
    if time.time() < deadline - 3:
        best_mask, best_score = local_search(best_mask, max_iter=2000, time_limit=deadline - 2)
    
    result = mask_to_list(best_mask)
    assert len(result) == N, f"Expected {N} elements, got {len(result)}"
    assert all(1 <= x <= MAX_VAL for x in result), "Values out of range"
    assert len(set(result)) == N, "Duplicate values"
    
    return result
