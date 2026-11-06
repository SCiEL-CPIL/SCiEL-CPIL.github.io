import random
import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Target range
    MIN_T = 128
    MAX_T = 256
    N = 128
    UNIVERSE_SIZE = N + 1  # 0 to 128
    
    # We represent A as a boolean numpy array of size 129
    # or a set. Numpy is faster for counting if we use convolution?
    # Convolution of A with A gives ordered pairs.
    # Let A_vec be 1s at positions in A.
    # conv = np.convolve(A_vec, A_vec)
    # conv[i] is the number of ordered pairs (a,b) such that a+b=i.
    # We need unordered pairs with a <= b.
    # For i, count_unordered = (conv[i] + (1 if i is even and i/2 in A else 0)) // 2
    # Wait, conv[i] counts (a,b) and (b,a).
    # If a != b, they are counted twice.
    # If a == b, they are counted once.
    # So count_unordered = (conv[i] + delta) / 2 where delta is 1 if a==b is possible and present.
    # Specifically, delta = 1 if i is even and (i//2) is in A, else 0.
    
    # This convolution approach is O(N log N) using FFT, or O(N^2) direct.
    # N=129 is small. Direct convolution is 129*129 = 16k ops. Very fast.
    # But we can just use numpy convolve which is optimized.
    
    def get_counts(A_vec):
        # A_vec is boolean array of size 129
        conv = np.convolve(A_vec, A_vec)
        # conv is length 257
        # We care about indices 128 to 256
        counts = np.zeros(MAX_T - MIN_T + 1, dtype=np.int32)
        for i in range(MIN_T, MAX_T + 1):
            c = conv[i]
            # Check if i/2 is in A
            if i % 2 == 0:
                mid = i // 2
                if 0 <= mid <= N and A_vec[mid]:
                    c += 1
            counts[i - MIN_T] = c // 2
        return counts

    def score(A_vec):
        counts = get_counts(A_vec)
        uncovered = np.sum(counts == 0)
        if uncovered > 0:
            return 10000 + uncovered * 100 + np.max(counts)
        return np.max(counts)

    # Initial random solution
    best_score = float('inf')
    best_A = None
    
    # Try a few initial sizes
    for k in [25, 30, 35, 40]:
        # Random initialization
        A_set = set(random.sample(range(UNIVERSE_SIZE), k))
        A_vec = np.zeros(UNIVERSE_SIZE, dtype=np.int32)
        for x in A_set:
            A_vec[x] = 1
        
        s = score(A_vec)
        if s < best_score:
            best_score = s
            best_A = A_vec.copy()
            
        # Local search
        current_A = A_vec.copy()
        current_score = s
        no_improve = 0
        
        while time.time() < deadline - 1.0:
            # Pick a move
            # 1. Add
            # 2. Remove
            # 3. Swap
            # 4. Fix uncovered
            
            move_type = random.random()
            
            new_A = current_A.copy()
            
            if move_type < 0.3:
                # Add
                # Prefer adding elements that help cover low counts?
                # Just random for now
                x = random.randint(0, N)
                if new_A[x] == 0:
                    new_A[x] = 1
                else:
                    continue
            elif move_type < 0.6:
                # Remove
                # Only remove if it doesn't break coverage?
                # Check if any target becomes 0
                # This is expensive to check naively.
                # Let's just try and evaluate.
                # Find a candidate to remove. 
                # Prefer removing elements that contribute to high multiplicity targets.
                # Or just random.
                candidates = np.where(current_A == 1)[0]
                if len(candidates) == 0:
                    continue
                x = random.choice(candidates)
                new_A[x] = 0
            elif move_type < 0.8:
                # Swap
                candidates_in = np.where(current_A == 1)[0]
                candidates_out = np.where(current_A == 0)[0]
                if len(candidates_in) == 0 or len(candidates_out) == 0:
                    continue
                x_in = random.choice(candidates_in)
                x_out = random.choice(candidates_out)
                new_A[x_in] = 0
                new_A[x_out] = 1
            else:
                # Fix uncovered
                counts = get_counts(current_A)
                uncovered_indices = np.where(counts == 0)[0]
                if len(uncovered_indices) == 0:
                    continue
                target_offset = random.choice(uncovered_indices)
                T = target_offset + MIN_T
                # Find a pair a+b=T
                # a in [0, 128], b = T-a in [0, 128]
                # a <= b
                valid_a = []
                for a in range(0, N+1):
                    b = T - a
                    if 0 <= b <= N and a <= b:
                        valid_a.append(a)
                if not valid_a:
                    continue
                a = random.choice(valid_a)
                b = T - a
                # Add a and b if not present
                added = False
                if new_A[a] == 0:
                    new_A[a] = 1
                    added = True
                if new_A[b] == 0:
                    new_A[b] = 1
                    added = True
                if not added:
                    continue

            new_score = score(new_A)
            
            if new_score < current_score:
                current_A = new_A
                current_score = new_score
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A.copy()
            else:
                no_improve += 1
                # If stuck for too long, restart or perturb?
                if no_improve > 100:
                    break

    # Convert best_A back to list
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return [i for i in range(UNIVERSE_SIZE) if best_A[i] == 1]
