import random
import time

def construct(q, n, radius):
    # Handle the specific case K_3(6,1) with optimization
    if q == 3 and n == 6 and radius == 1:
        return _solve_k3_6_1()
    
    # Fallback for other cases
    return _generic_greedy(q, n, radius)

def _solve_k3_6_1():
    N = 729
    n = 6
    
    # Precompute words
    def get_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % 3)
            idx //= 3
        return w
        
    idx_to_word = [get_word(i) for i in range(N)]
    
    # Precompute balls
    balls = [0] * N
    for idx in range(N):
        w = idx_to_word[idx][:] # Copy
        mask = 1 << idx
        # Self is already in mask? No, 1<<idx is self.
        # Wait, mask starts with self.
        # Neighbors:
        for i in range(n):
            orig = w[i]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[i] = new_val
                # Calculate index
                n_idx = 0
                for k in range(n):
                    n_idx = n_idx * 3 + w[k]
                mask |= (1 << n_idx)
            w[i] = orig
        balls[idx] = mask
        
    # Greedy Initialization
    uncovered = (1 << N) - 1
    code = []
    in_code = [False] * N
    
    # Optimization: Precompute popcount for masks? No, int.bit_count() is C-speed.
    
    # To speed up greedy, we can just iterate all N each time.
    # N=729, iters ~75. 729*75 = 54k checks. Very fast.
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for idx in range(N):
            if in_code[idx]:
                continue
            count = (balls[idx] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break # Should not happen if uncovered != 0
        code.append(best_idx)
        in_code[best_idx] = True
        uncovered &= ~balls[best_idx]
        
    # Local Search
    counts = [0] * N
    
    def add_to_counts(idx, delta):
        mask = balls[idx]
        m = mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            counts[j] += delta
            m ^= lsb

    for idx in code:
        add_to_counts(idx, 1)
        
    def is_redundant(idx):
        mask = balls[idx]
        m = mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if counts[j] == 1:
                return False
            m ^= lsb
        return True

    # Phase 1: Remove all redundancies
    improved = True
    while improved:
        improved = False
        for i in range(len(code)):
            if is_redundant(code[i]):
                add_to_counts(code[i], -1)
                in_code[code[i]] = False
                code.pop(i)
                improved = True
                break

    # Phase 2: Swap and Perturb
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    while time.time() < deadline:
        # Try to find a swap
        # Find c with min unique bits
        min_unique = N
        best_c_idx = -1
        best_c_pos = -1
        
        for pos, c in enumerate(code):
            mask = balls[c]
            m = mask
            unique_count = 0
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if counts[j] == 1:
                    unique_count += 1
                m ^= lsb
            if unique_count < min_unique:
                min_unique = unique_count
                best_c_idx = c
                best_c_pos = pos
                
        if best_c_idx == -1:
            break
            
        # Calculate unique mask for best_c_idx
        unique_mask = 0
        mask = balls[best_c_idx]
        m = mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if counts[j] == 1:
                unique_mask |= (1 << j)
            m ^= lsb
            
        # Find c_prime
        found_swap = False
        for c_prime in range(N):
            if in_code[c_prime]:
                continue
            if (unique_mask & ~balls[c_prime]) == 0:
                # Swap
                add_to_counts(best_c_idx, -1)
                in_code[best_c_idx] = False
                code[best_c_pos] = c_prime
                in_code[c_prime] = True
                add_to_counts(c_prime, 1)
                found_swap = True
                break
        
        if not found_swap:
            # Perturb: Add random word, then remove redundancies
            # Pick random word not in code
            # Simple way: pick random idx, if in_code, retry or just pick another
            # Since N is small, we can just pick random until we find one not in code
            # Or generate list of not-in-code (expensive)
            # Let's just pick random idx.
            
            # To avoid bias, we can pick a random idx from 0..N-1
            # If it's in code, it won't help. 
            # Better: maintain a list of free indices?
            # For now, just pick random.
            
            attempts = 0
            while attempts < 10:
                c_rand = random.randint(0, N - 1)
                if not in_code[c_rand]:
                    break
                attempts += 1
            else:
                break # All in code? Impossible since |code| < N
                
            code.append(c_rand)
            in_code[c_rand] = True
            add_to_counts(c_rand, 1)
            
            # Remove redundancies
            local_improved = True
            while local_improved:
                local_improved = False
                for i in range(len(code)):
                    if is_redundant(code[i]):
                        add_to_counts(code[i], -1)
                        in_code[code[i]] = False
                        code.pop(i)
                        local_improved = True
                        break

    # Convert to list of lists
    return [idx_to_word[idx] for idx in code]

def _generic_greedy(q, n, radius):
    # Fallback
    N = q ** n
    if N > 10000:
        return []
        
    def get_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w
        
    def encode(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    balls = [0] * N
    for idx in range(N):
        w = get_word(idx)
        mask = 0
        for j in range(N):
            w2 = get_word(j)
            if hamming_dist(w, w2) <= radius:
                mask |= (1 << j)
        balls[idx] = mask
        
    uncovered = (1 << N) - 1
    code = []
    in_code = [False] * N
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for idx in range(N):
            if in_code[idx]:
                continue
            count = (balls[idx] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        code.append(best_idx)
        in_code[best_idx] = True
        uncovered &= ~balls[best_idx]
        
    return [get_word(idx) for idx in code]
