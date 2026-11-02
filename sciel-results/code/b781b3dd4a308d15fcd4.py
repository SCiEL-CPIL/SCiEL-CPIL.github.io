def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fft(best_U)
    
    random.seed(42)
    
    # Strategy 1: Generate diverse initial candidates
    candidates = []
    
    # Golomb Rulers (good for large diffset)
    def golomb_ruler(n):
        # Simple greedy or known small rulers
        # For n=4: 0,1,4,6
        # For n=5: 0,1,4,9,11
        # We'll just use random sparse sets for now, or known sequences
        pass
        
    # Random sparse sets with varying density
    for trial in range(100):
        n = random.randint(5, 50)
        # Choose max_val to keep ratio interesting
        # If n is large, max_val should be large enough to avoid collisions in sums
        # But not too large to keep log(2*maxU+1) small?
        # Score = 1 + log(diff/sum) / log(2*maxU+1)
        # We want diff/sum large.
        # Max diff is 2*maxU+1. Max sum is 2*maxU+1.
        # If U is sparse, diff ~ n^2, sum ~ n^2.
        # Ratio ~ 1. Log(1) = 0. Score 1.
        # We need diff > sum.
        # Diffset is symmetric around 0, Sumset is positive.
        # |U-U| is usually larger than |U+U| for sparse sets?
        # Actually, for U={0, 1, ..., n-1}, |U+U| = 2n-1, |U-U| = 2n-1. Ratio 1.
        # For U={0, 1, 3, 7, ...} (powers of 2 minus 1?), sums are unique?
        # If sums are unique, |U+U| = n(n+1)/2.
        # Diffs: u_i - u_j. If u_i are superincreasing, diffs are unique?
        # If u_i = 2^i, diffs are unique? 2^i - 2^j. Yes, all distinct.
        # |U-U| = n^2 - n + 1 (including 0).
        # Ratio = (n^2-n+1) / (n(n+1)/2) = 2(n-1+1/n) / (n+1) ~ 2.
        # Log(2) / log(2*maxU+1).
        # maxU = 2^(n-1).
        # Denom ~ n log 2.
        # Score ~ 1 + log(2) / (n log 2) = 1 + 1/n.
        # This is > 1.
        # So superincreasing sequences are good.
        
        # Let's try superincreasing-like sets
        if trial % 2 == 0:
            # Geometric
            base = random.randint(2, 5)
            n = random.randint(4, 15)
            U = [base ** i for i in range(n)]
            U.insert(0, 0)
            # Shuffle to break symmetry? No, keep sorted
            U.sort()
            candidates.append(U)
        else:
            # Random sparse
            n = random.randint(5, 30)
            max_val = random.randint(100, 1000)
            U = [0]
            remaining = list(range(1, max_val + 1))
            random.shuffle(remaining)
            for _ in range(n - 1):
                if remaining:
                    U.append(remaining.pop())
            U.sort()
            candidates.append(U)

    # Evaluate initial candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set_fft(U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Local Search
    U = list(best_U)
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        n = len(U)
        maxU = max(U) if U else 0
        
        # Generate moves
        moves = []
        
        # 1. Add element
        if n < 100:
            for _ in range(20):
                if random.random() < 0.5:
                    # Near existing
                    base = random.choice(U)
                    offset = random.randint(-10, 10)
                    new_val = base + offset
                else:
                    # Random
                    new_val = random.randint(1, maxU * 2 + 10)
                
                if new_val > 0 and new_val not in U:
                    new_U = U + [new_val]
                    new_U.sort()
                    moves.append(new_U)
        
        # 2. Remove element
        if n > 3:
            for _ in range(5):
                idx = random.randint(1, n - 1)
                new_U = U[:idx] + U[idx+1:]
                moves.append(new_U)
        
        # 3. Replace element
        if n >= 3:
            for _ in range(10):
                idx = random.randint(1, n - 1)
                new_val = random.randint(1, maxU * 2 + 10)
                if new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        # 4. Shift element
        if n >= 3:
            for _ in range(10):
                idx = random.randint(1, n - 1)
                delta = random.choice([-2, -1, 1, 2, 5, 10])
                new_val = U[idx] + delta
                if new_val > 0 and new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U.sort()
                    moves.append(new_U)

        if not moves:
            no_improve_count += 1
            continue
        
        # Evaluate moves
        current_best_score = best_score
        current_best_U = U
        
        for new_U in moves:
            if time.time() > deadline:
                break
            s = _score_set_fft(new_U)
            if s > current_best_score:
                current_best_score = s
                current_best_U = new_U
        
        if current_best_U != U:
            U = current_best_U
            best_score = current_best_score
            best_U = list(U)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck, try a bigger jump (restart from best or random)
            if no_improve_count % 10 == 0:
                # Perturb best
                U = list(best_U)
                if len(U) > 3:
                    idx = random.randint(1, len(U)-1)
                    delta = random.randint(-20, 20)
                    new_val = U[idx] + delta
                    if new_val > 0 and new_val not in U:
                        U[idx] = new_val
                        U.sort()
                else:
                    # Add random
                    maxU = max(U)
                    new_val = random.randint(1, maxU + 100)
                    if new_val not in U:
                        U.append(new_val)
                        U.sort()

    return best_U
