import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: greedy for generality
        total = q ** n
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q

        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = powers[i]
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)

        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []

        while True:
            if time.time() > deadline - 5:
                break
            max_idx = int(np.argmax(counts))
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(total):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break

        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17

    # Precompute balls as lists of integers
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)

    # Step 1: Deterministic algebraic construction using product code
    # Product of ternary Golay code G(11,6) and ternary repetition code [3,1,3]
    # This yields a (64,8,2) ternary code with 486 codewords
    
    # Generate ternary Golay code G(11,6)
    # The ternary Golay code is a [11,6,5] code over GF(3)
    # We'll use a systematic generator matrix
    
    # Generator matrix for ternary Golay code (systematic form)
    # This is one valid generator matrix for the ternary Golay code
    G_golay = np.array([
        [1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 2],
        [0, 1, 0, 0, 0, 0, 2, 1, 0, 2, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 2, 1, 2],
        [0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 1],
        [0, 0, 0, 0, 1, 0, 2, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 2, 0, 2, 2]
    ], dtype=np.int32) % 3
    
    # Generate all 3^6 = 729 codewords of the Golay code
    golay_codewords = []
    for msg in range(729):
        digits = []
        tmp = msg
        for i in range(6):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        msg_vec = np.array(digits, dtype=np.int32)
        cw = (G_golay @ msg_vec) % 3
        golay_codewords.append(cw)
    
    # Generate ternary repetition code [3,1,3]
    # Codewords: (0,0,0), (1,1,1), (2,2,2)
    rep_codewords = [np.array([0,0,0]), np.array([1,1,1]), np.array([2,2,2])]
    
    # Product code: for each pair (g, r) where g is in Golay, r is in rep
    # The product code has length 11*3 = 33, but we need length 8
    # Wait, this doesn't directly give us length 8. Let me reconsider.
    
    # Actually, the known construction for K_3(8,1) = 486 uses a different approach.
    # Let me use a simpler deterministic construction that's known to work.
    
    # Alternative: Use a greedy approach with a clever ordering based on algebraic structure
    # Or use the fact that 486 = 2 * 3^5 = 2 * 243
    
    # Let me try a different algebraic construction:
    # Consider the code as a subset of {0,1,2}^8
    # Use a parity-check based construction
    
    # Actually, let me just use a well-known greedy with a good initial ordering
    # and then apply aggressive local search
    
    # Step 1: Greedy construction with algebraic-inspired ordering
    # Order codewords by their "weight" or some algebraic property
    
    # Compute a hash-like value for ordering
    def algebraic_order(idx):
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        # Sum of digits weighted by position
        val = sum(d * (i + 1) for i, d in enumerate(digits))
        # Add some nonlinearity
        val2 = sum(d * d * (i + 1) for i, d in enumerate(digits))
        return val * 1000 + val2
    
    order = sorted(range(total), key=algebraic_order)
    
    covered = np.zeros(total, dtype=bool)
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    selected = []
    
    # Greedy with algebraic ordering as tie-breaker
    for _ in range(600):  # Safety limit
        if time.time() > deadline - 30:
            break
        max_count = -1
        best_idx = -1
        for idx in order:
            if counts[idx] > max_count:
                max_count = counts[idx]
                best_idx = idx
                if max_count == ball_size:
                    break
        if best_idx == -1 or max_count == 0:
            break
        selected.append(best_idx)
        for y in balls[best_idx]:
            if not covered[y]:
                covered[y] = True
                for i in balls[y]:
                    counts[i] -= 1
        if np.all(covered):
            break
    
    best_size = len(selected)
    best_solution = selected[:]
    
    # Step 2: Aggressive local search - remove redundant codewords
    current_set = set(best_solution)
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in current_set:
        for y in balls[cw]:
            cover_count[y] += 1
    
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and time.time() < deadline - 20 and iteration < max_iterations:
        improved = False
        iteration += 1
        candidates = list(current_set)
        random.shuffle(candidates)
        for cw in candidates:
            if cw not in current_set:
                continue
            # Check if redundant
            redundant = True
            for y in balls[cw]:
                if cover_count[y] == 1:
                    redundant = False
                    break
            if redundant:
                for y in balls[ccw]:
                    cover_count[y] -= 1
                current_set.remove(cw)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
    
    # Step 3: Try swap-based local search
    if time.time() < deadline - 10:
        current_set = set(best_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for cw in current_set:
            for y in balls[cw]:
                cover_count[y] += 1
        
        # Try to find codewords that can be replaced by fewer codewords
        # For each codeword, check if its covered points are covered by others
        # This is essentially the same as redundancy check, but we can try
        # to find pairs that can be merged
        
        # Alternative: try adding a random codeword and then removing redundants
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Add a random codeword
            random_cw = random.randint(0, total - 1)
            if random_cw in current_set:
                continue
            
            # Tentatively add it
            for y in balls[random_cw]:
                cover_count[y] += 1
            current_set.add(random_cw)
            
            # Remove redundants
            improved_local = True
            while improved_local and time.time() < deadline - 10:
                improved_local = False
                candidates = list(current_set)
                random.shuffle(candidates)
                for cw in candidates:
                    if cw not in current_set:
                        continue
                    redundant = True
                    for y in balls[cw]:
                        if cover_count[y] == 1:
                            redundant = False
                            break
                    if redundant:
                        for y in balls[cw]:
                            cover_count[y] -= 1
                        current_set.remove(cw)
                        improved_local = True
            
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_solution = list(current_set)
            else:
                # Revert
                current_set.remove(random_cw)
                for y in balls[random_cw]:
                    cover_count[y] -= 1
    
    # Convert best_solution to list of digit lists
    if best_solution is None:
        best_solution = []

    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)

    return result
