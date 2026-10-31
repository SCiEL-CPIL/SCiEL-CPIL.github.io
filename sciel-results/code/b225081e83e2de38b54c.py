import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150s kill

def _score_from_vector(vec, max_val):
    """
    Compute the score for a set represented by a binary indicator vector.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    n = np.sum(vec)
    if n < 2 or max_val < 1:
        return 0.0
    
    L = len(vec)
    required_len = 2 * (max_val + 1)
    if L < required_len:
        vec = np.pad(vec, (0, required_len - L), mode='constant')
        L = len(vec)
        
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    f = np.fft.fft(vec, n=fft_len)
    
    # Sumset: |U+U|
    conv = np.fft.ifft(f * f)
    sums = np.abs(conv.real) > 1e-9
    num_sums = int(np.sum(sums))
    
    # Differenceset: |U-U|
    corr = np.fft.ifft(np.conj(f) * f)
    diffs = np.abs(corr.real) > 1e-9
    num_diffs = int(np.sum(diffs))
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / float(num_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _list_to_vec(lst, length):
    """Convert list of indices to binary vector of given length."""
    vec = np.zeros(length, dtype=np.float64)
    for i in lst:
        if i < length:
            vec[i] = 1.0
    return vec

def _greedy_search(u_init, max_val_limit, time_limit):
    """
    Greedy hill-climbing search.
    At each step, evaluate a set of candidate moves (insert, remove, move)
    and pick the one that maximizes the score.
    """
    current_u = sorted(set(u_init))
    if 0 not in current_u:
        current_u.append(0)
    current_u = sorted(current_u)
    
    current_max = max(current_u)
    # Pre-allocate a large enough vector for scoring
    fixed_vec_len = 2 * (max_val_limit + 1)
    
    current_vec = _list_to_vec(current_u, fixed_vec_len)
    current_score = _score_from_vector(current_vec, current_max)
    
    best_score = current_score
    best_u = current_u[:]
    
    start_time = time.time()
    
    # Candidate moves generation
    # We will try:
    # 1. Adding a new random element
    # 2. Moving an existing element to a new random position
    # 3. Removing an element (if size > 2)
    
    while time.time() < time_limit:
        # Generate a batch of candidate moves
        candidates = []
        
        # 1. Add new element
        for _ in range(10):
            new_val = random.randint(1, max_val_limit)
            if new_val not in current_u:
                new_u = current_u + [new_val]
                new_u.sort()
                candidates.append(new_u)
        
        # 2. Move existing element
        for _ in range(10):
            if len(current_u) < 2: continue
            idx = random.randint(0, len(current_u) - 1)
            old_val = current_u[idx]
            new_val = random.randint(1, max_val_limit)
            if new_val == old_val or new_val in current_u:
                continue
            new_u = current_u[:]
            new_u[idx] = new_val
            new_u.sort()
            candidates.append(new_u)
        
        # 3. Remove element (only if large)
        if len(current_u) > 10:
            for _ in range(5):
                idx = random.randint(1, len(current_u) - 1) # Don't remove 0
                new_u = current_u[:]
                new_u.pop(idx)
                new_u.sort()
                candidates.append(new_u)
        
        if not candidates:
            break
            
        # Evaluate candidates
        best_candidate_score = current_score
        best_candidate_u = current_u
        
        for cand in candidates:
            cand_max = max(cand)
            cand_vec = _list_to_vec(cand, fixed_vec_len)
            cand_score = _score_from_vector(cand_vec, cand_max)
            
            if cand_score > best_candidate_score:
                best_candidate_score = cand_score
                best_candidate_u = cand
                
        # Accept the best move
        if best_candidate_score > current_score:
            current_u = best_candidate_u
            current_score = best_candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best_u = current_u[:]
        else:
            # No improvement found in this batch, try a different random seed or break
            # To avoid getting stuck, we can force a random jump occasionally
            if random.random() < 0.1:
                # Random restart from current set with perturbation
                new_u = current_u[:]
                for _ in range(3):
                    idx = random.randint(0, len(new_u) - 1)
                    new_val = random.randint(1, max_val_limit)
                    if new_val not in new_u:
                        new_u[idx] = new_val
                new_u.sort()
                new_u = list(set(new_u))
                if 0 not in new_u: new_u.append(0)
                new_u.sort()
                current_u = new_u
                current_max = max(current_u)
                current_vec = _list_to_vec(current_u, fixed_vec_len)
                current_score = _score_from_vector(current_vec, current_max)
            else:
                # Stagnation, break out to try a different initial config
                break
                
    return best_u, best_score

def _generate_initial_sets(n, max_val):
    """Generate diverse initial sets."""
    sets = []
    
    # 1. Exponential growth
    u_exp = [0]
    cur = 1
    while len(u_exp) < n and cur <= max_val:
        u_exp.append(cur)
        cur = int(cur * 1.5) + 1
    sets.append(u_exp)
    
    # 2. Fibonacci-like
    u_fib = [0, 1]
    while len(u_fib) < n:
        nxt = u_fib[-1] + u_fib[-2]
        if nxt > max_val: break
        u_fib.append(nxt)
    sets.append(u_fib)
    
    # 3. Random sorted
    others = random.sample(range(1, max_val + 1), min(n - 1, max_val))
    u_rand = [0] + sorted(others)
    sets.append(u_rand)
    
    # 4. Golomb-like (greedy distinct differences)
    u_golomb = [0]
    diffs = set()
    next_val = 1
    while len(u_golomb) < n and next_val <= max_val:
        valid = True
        for x in u_golomb:
            d = next_val - x
            if d in diffs or -d in diffs:
                valid = False
                break
        if valid:
            u_golomb.append(next_val)
            for x in u_golomb[:-1]:
                diffs.add(next_val - x)
                diffs.add(x - next_val)
        next_val += 1
    sets.append(u_golomb)
    
    return sets

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy:
    # 1. Generate diverse initial sets.
    # 2. Refine using Greedy Hill Climbing with FFT-based scoring.
    # 3. Keep the best result.
    
    # We focus on a few key configurations to ensure we have time for deep search
    configs = [
        # (n, M_limit, description)
        (20, 50000, "small_n"),
        (40, 200000, "med_n"),
        (60, 500000, "large_n"),
    ]
    
    for n, M_limit, desc in configs:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Allocate remaining time divided among initial sets
        time_per_set = max(1.0, (_DEADLINE - 5.0 - time.time()) / 4)
        
        initial_sets = _generate_initial_sets(n, M_limit)
        
        for u_init in initial_sets:
            if time.time() > _DEADLINE - 5.0:
                break
            
            if 0 not in u_init:
                u_init = [0] + u_init
            u_init = sorted(set(u_init))
            
            # Limit size to avoid O(N^2) blowup in move evaluation if it gets too big
            if len(u_init) > 100:
                others = [x for x in u_init if x != 0]
                random.shuffle(others)
                u_init = [0] + others[:100]
                
            u_opt, score_opt = _greedy_search(u_init, M_limit, time_per_set)
            
            if score_opt > best_score:
                best_score = score_opt
                best_set = u_opt
                
    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
# EVOLVE-BLOCK-END
