"""Improved program for 32-element set with many 3-AP differences."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in set."""
    s_set = set(s)
    diffs = set()
    n = len(s)
    sorted_s = sorted(s)
    for i in range(n):
        for j in range(i + 1, n):
            d = sorted_s[j] - sorted_s[i]
            if d <= 0:
                continue
            if sorted_s[i] + 2 * d in s_set:
                diffs.add(d)
    return len(diffs)


def _try_add_to_complete_aps(s, s_set, n_target=32, max_range=512):
    """Try to add elements that complete many 3-APs."""
    s_list = list(s)
    s_set = set(s)
    for _ in range(200):
        if len(s_list) >= n_target:
            break
        best_val = None
        best_gain = 0
        # Candidate values: try to complete APs
        candidates = set()
        for a in s_list:
            for b in s_list:
                if a == b:
                    continue
                # a, b, c where c = 2b - a
                c = 2 * b - a
                if 1 <= c <= max_range and c not in s_set:
                    candidates.add(c)
                # a, c, b where c = 2a - b (c is middle)
                c = 2 * a - b
                if 1 <= c <= max_range and c not in s_set:
                    candidates.add(c)
                # c, a, b where c = 2a - b... already covered
                # a, b, c with a as first: need c = a + 2(b-a) = 2b - a (covered)
                # c, a, b with b as last: need c = 2a - b (covered)
        
        # Also try random candidates
        for _ in range(50):
            c = random.randint(1, max_range)
            if c not in s_set:
                candidates.add(c)
        
        for c in candidates:
            if c in s_set:
                continue
            # Count how many new APs this would create
            new_aps = 0
            for a in s_list:
                for b in s_list:
                    if a == b or a == c or b == c:
                        continue
                    # Check if (a, b, c) or (c, a, b) or (a, c, b) or (b, c, a) etc. form AP
                    # Actually just check all triples involving c
                    pass
            
            # Simpler: just check if adding c creates any new AP
            # An AP (x, x+d, x+2d) is new if it wasn't there before
            # With c added, new APs must include c
            for a in s_list:
                for b in s_list:
                    if a == b:
                        continue
                    # Check if (a, b, c) is an AP
                    if b - a == c - b:  # a, b, c
                        new_aps += 1
                    # Check if (a, c, b) is an AP
                    if c - a == b - c:
                        new_aps += 1
                    # Check if (c, a, b) is an AP
                    if a - c == b - a:
                        new_aps += 1
                    # Check if (b, a, c) is an AP
                    if a - b == c - a:
                        new_aps += 1
                    # Check if (b, c, a) is an AP
                    if c - b == a - c:
                        new_aps += 1
                    # Check if (c, b, a) is an AP
                    if b - c == a - b:
                        new_aps += 1
            
            if new_aps > best_gain:
                best_gain = new_aps
                best_val = c
        
        if best_val is not None:
            s_list.append(best_val)
            s_set.add(best_val)
        else:
            # Pick a random value
            while True:
                c = random.randint(1, max_range)
                if c not in s_set:
                    s_list.append(c)
                    s_set.add(c)
                    break
    
    return s_list


def _local_search(s, max_iter=5000, deadline=None):
    """Local search: swap elements to improve score."""
    s_list = list(s)
    s_set = set(s_list)
    current_score = _score_set(s_list)
    best_score = current_score
    best_list = list(s_list)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random element to replace
        idx = random.randint(0, len(s_list) - 1)
        old_val = s_list[idx]
        
        # Try replacing with a value that completes APs
        candidates = set()
        for a in s_list:
            for b in s_list:
                if a == b:
                    continue
                c = 2 * b - a
                if 1 <= c <= 512 and c != old_val:
                    candidates.add(c)
                c = 2 * a - b
                if 1 <= c <= 512 and c != old_val:
                    candidates.add(c)
        
        # Add some random candidates
        for _ in range(10):
            c = random.randint(1, 512)
            if c != old_val:
                candidates.add(c)
        
        best_candidate = None
        best_candidate_score = current_score
        
        for c in candidates:
            if c in s_set and c != old_val:
                continue
            # Temporarily replace
            s_list[idx] = c
            s_set.discard(old_val)
            s_set.add(c)
            new_score = _score_set(s_list)
            s_list[idx] = old_val
            s_set.discard(c)
            s_set.add(old_val)
            
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = c
        
        if best_candidate is not None:
            s_list[idx] = best_candidate
            s_set.discard(old_val)
            s_set.add(best_candidate)
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_list = list(s_list)
        else:
            # No improvement, try a random move
            while True:
                c = random.randint(1, 512)
                if c not in s_set or c == old_val:
                    break
            s_list[idx] = c
            s_set.discard(old_val)
            s_set.add(c)
            current_score = _score_set(s_list)
    
    return best_list, best_score


def _simulated_annealing(s, max_iter=20000, t_start=10.0, t_end=0.1, deadline=None):
    """Simulated annealing."""
    s_list = list(s)
    s_set = set(s_list)
    current_score = _score_set(s_list)
    best_score = current_score
    best_list = list(s_list)
    
    n_iter = max_iter
    for i in range(n_iter):
        if deadline and time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (i / max(1, n_iter - 1))
        
        idx = random.randint(0, len(s_list) - 1)
        old_val = s_list[idx]
        
        # Generate candidate
        candidates = set()
        for a in s_list:
            for b in s_list:
                if a == b:
                    continue
                c = 2 * b - a
                if 1 <= c <= 512 and c != old_val:
                    candidates.add(c)
                c = 2 * a - b
                if 1 <= c <= 512 and c != old_val:
                    candidates.add(c)
        
        for _ in range(5):
            c = random.randint(1, 512)
            if c != old_val:
                candidates.add(c)
        
        if not candidates:
            continue
        
        c = random.choice(list(candidates))
        if c in s_set and c != old_val:
            continue
        
        s_list[idx] = c
        s_set.discard(old_val)
        s_set.add(c)
        new_score = _score_set(s_list)
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / max(t, 1e-10)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_list = list(s_list)
        else:
            s_list[idx] = old_val
            s_set.discard(c)
            s_set.add(old_val)
    
    return best_list, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = list(range(1, 33))
    best_score = _score_set(best_set)
    
    # Multi-start with different initial strategies
    random.seed(42)
    
    for start_iter in range(30):
        if time.time() > deadline - 5:
            break
        
        # Different initializations
        if start_iter % 3 == 0:
            # Consecutive
            s_init = list(range(1, 33))
        elif start_iter % 3 == 1:
            # Arithmetic progression with step 2
            s_init = [1 + 2 * i for i in range(32)]
        else:
            # Random
            s_init = random.sample(range(1, 513), 32)
        
        # Try to improve via targeted addition
        s_improved = _try_add_to_complete_aps(s_init, set(s_init), n_target=32, max_range=512)
        score_improved = _score_set(s_improved)
        
        if score_improved > best_score:
            best_score = score_improved
            best_set = list(s_improved)
        
        # Local search polish
        if time.time() > deadline - 10:
            break
        
        s_ls, score_ls = _local_search(s_improved, max_iter=2000, deadline=deadline)
        if score_ls > best_score:
            best_score = score_ls
            best_set = list(s_ls)
        
        # Simulated annealing polish
        if time.time() > deadline - 15:
            break
        
        s_sa, score_sa = _simulated_annealing(s_improved, max_iter=5000, deadline=deadline)
        if score_sa > best_score:
            best_score = score_sa
            best_set = list(s_sa)
    
    return best_set
# EVOLVE-BLOCK-END
