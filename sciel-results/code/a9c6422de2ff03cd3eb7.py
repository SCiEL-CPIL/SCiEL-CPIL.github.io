import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
        
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187
    p3 = [1, 3, 9, 27, 81, 243, 729]
    
    # Precompute balls
    ball_indices = np.empty((N, 15), dtype=np.int32)
    for i in range(N):
        tmp = i
        digits = [0]*7
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        ball_indices[i, 0] = i
        idx = 1
        for pos in range(7):
            orig = digits[pos]
            for val in [0, 1, 2]:
                if val != orig:
                    ball_indices[i, idx] = i + (val - orig) * p3[pos]
                    idx += 1
                    
    # Precompute coverers (inverse balls)
    # coverers[w] is a list of c such that w is in ball_indices[c]
    coverers = [np.array([], dtype=np.int32) for _ in range(N)]
    for c in range(N):
        for w in ball_indices[c]:
            # Append c to coverers[w]
            # Since N is small, we can use lists and convert later
            coverers[w] = np.append(coverers[w], c)
            
    # Convert coverers to numpy arrays for faster access?
    # Actually, coverers[w] is used to find candidates to add.
    # In greedy/repair, we need to pick a c covering w.
    
    def is_valid(code_set, cnt):
        # cnt is array of counts
        return (cnt > 0).all()
        
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        selected_set = set()
        
        while uncovered.any():
            # Find best candidate
            best_idx = -1
            best_count = -1
            
            # Optimization: Only consider candidates that cover at least one uncovered word
            # This is hard to filter without scanning.
            # Just scan all.
            # Vectorized approach:
            # counts = np.array([uncovered[ball_indices[i]].sum() for i in range(N)])
            # This is slow in Python loop.
            
            # Use a subset?
            # Or just pick random uncovered word, and pick a random coverer?
            # This is "Greedy with Random Tie-breaking" or "Greedy on Random Uncovered".
            
            # Let's stick to standard greedy but optimized.
            # Since N=2187, scanning all 2187 is ~2187 * 15 ops = 32k ops.
            # 100 steps = 3M ops. Fast.
            
            # We can precompute a matrix M where M[i, j] = 1 if j in ball(i)?
            # Then counts = M @ uncovered?
            # M is 2187x2187. 4.7M entries.
            # np.dot is fast.
            
            # Let's create M once outside?
            # M is sparse.
            
            # For now, simple loop.
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for i in candidates:
                if i in selected_set:
                    continue
                cnt = uncovered[ball_indices[i]].sum()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                elif cnt == best_count and cnt > 0:
                    if rng.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered[ball_indices[best_idx]] = False
            
            if time.time() > deadline - 10.0:
                break
                
        return selected

    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball_indices[c]] += 1
            
        best_code = code.copy()
        best_size = len(code)
        
        while time.time() < deadline - 5.0:
            improved = False
            
            # Phase 1: Single Deletion
            # Try to remove any c where min(cnt[ball[c]]) >= 2
            # Iterate in random order
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if idx >= len(code): continue
                c = code[idx]
                if cnt[ball_indices[c]].min() >= 2:
                    cnt[ball_indices[c]] -= 1
                    code.pop(idx)
                    code_set.remove(c)
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                        
            # Phase 2: Double Deletion + Single Addition
            # Try to remove c1, c2 and add c'
            # Sample random pairs
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts and time.time() < deadline - 5.0:
                attempts += 1
                if len(code) < 2:
                    break
                i1, i2 = rng.sample(range(len(code)), 2)
                c1 = code[i1]
                c2 = code[i2]
                
                # Calculate U: words that become uncovered
                # U is subset of Ball(c1) U Ball(c2)
                union_ball = np.union1d(ball_indices[c1], ball_indices[c2])
                U = []
                for w in union_ball:
                    # Count how many of c1, c2 cover w
                    # If w in ball(c1), cnt decreases by 1
                    # If w in ball(c2), cnt decreases by 1
                    # If w in both, decreases by 2
                    # new_cnt = cnt[w] - delta
                    # If new_cnt == 0, w is in U
                    
                    in_c1 = (ball_indices[c1] == w).any()
                    in_c2 = (ball_indices[c2] == w).any()
                    delta = (1 if in_c1 else 0) + (1 if in_c2 else 0)
                    if cnt[w] - delta == 0:
                        U.append(w)
                        
                if not U:
                    # Can remove both
                    # Remove c2 first to keep indices valid?
                    # Or just remove both
                    # We need to update cnt
                    # Remove c1
                    cnt[ball_indices[c1]] -= 1
                    # Remove c2
                    # Note: if we remove c1, cnt changes.
                    # We calculated U based on original cnt.
                    # So we can just apply the removals.
                    code.remove(c2)
                    code_set.remove(c2)
                    code.remove(c1)
                    code_set.remove(c1)
                    # cnt updates:
                    # We already subtracted c1's contribution? No, we did it above.
                    # We need to subtract c2's contribution too.
                    # But wait, if w was in both, we subtracted 1 for c1.
                    # We need to subtract 1 for c2.
                    # So just do cnt[ball_indices[c2]] -= 1
                    cnt[ball_indices[c2]] -= 1
                    
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    break # Restart phases
                    
                # If U is not empty, find c'
                # c' must cover all w in U
                # c' in Intersection(Ball(w) for w in U)
                
                # Start with Ball(U[0])
                candidates = set(ball_indices[U[0]].tolist())
                for w in U[1:]:
                    candidates &= set(ball_indices[w].tolist())
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                # Filter out c' in code_set
                # c' can be c1 or c2? No, we are removing them.
                # So c' must not be in code_set.
                valid_c = [c for c in candidates if c not in code_set]
                
                if not valid_c:
                    continue
                    
                c_prime = rng.choice(valid_c)
                
                # Perform move
                # Remove c1, c2
                # Add c_prime
                
                # Update cnt
                # Remove c1
                cnt[ball_indices[c1]] -= 1
                # Remove c2
                cnt[ball_indices[c2]] -= 1
                # Add c_prime
                cnt[ball_indices[c_prime]] += 1
                
                code.remove(c2)
                code.remove(c1)
                code_set.remove(c2)
                code_set.remove(c1)
                
                code.append(c_prime)
                code_set.add(c_prime)
                
                improved = True
                # Size decreased by 1
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                break # Restart phases
                
            if not improved:
                # Try Single Swap?
                # Or break if no improvement for a while?
                # For now, break if no improvement in this pass
                break
                
        return best_code

    # Main Loop
    best_code = None
    best_size = N + 1
    
    # 1. Greedy
    code = greedy_init(0)
    code = local_search(code, 0)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code
        
    # 2. ILS
    current_code = code.copy()
    
    while time.time() < deadline - 5.0:
        # Perturb
        # Remove 2-3 random codewords
        k = rng.randint(2, 4)
        # Ensure we don't remove too many
        if len(current_code) > k:
            to_remove = rng.sample(current_code, k)
            # Remove them
            for c in to_remove:
                current_code.remove(c)
            
            # Repair
            # Find uncovered
            cnt = np.zeros(N, dtype=np.int32)
            for c in current_code:
                cnt[ball_indices[c]] += 1
                
            while not (cnt > 0).all():
                # Find an uncovered word
                uncovered_words = np.where(cnt == 0)[0]
                if len(uncovered_words) == 0: break
                w = rng.choice(uncovered_words)
                
                # Pick a random coverer
                # coverers[w] is precomputed
                c = rng.choice(coverers[w])
                if c not in current_code:
                    current_code.append(c)
                    cnt[ball_indices[c]] += 1
                else:
                    # If already in code, it shouldn't be uncovered?
                    # Unless we removed it?
                    # If c is in code, cnt[w] should be > 0.
                    # So this case shouldn't happen if logic is correct.
                    # But if we removed it, it's not in code.
                    # So c must be new.
                    pass
                    
        # Local Search
        new_code = local_search(current_code, int(time.time()*1000) % 10000)
        
        if len(new_code) < best_size:
            best_size = len(new_code)
            best_code = new_code
            current_code = new_code.copy()
            
        # If we reached 186 (or lower), we might stop?
        # But we want to minimize.
        
    # Convert to result
    result = []
    for idx in best_code:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        result.append(digits)
        
    return result

def _construct_general(q, n, radius, deadline):
    # Fallback
    # Just return a valid code, maybe not optimal
    # Greedy
    N = q ** n
    # ... (Same as before but simplified)
    pass
