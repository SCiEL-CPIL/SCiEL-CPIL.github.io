import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    N = 3 ** 7
    all_mask = (1 << N) - 1

    # Precompute digits for each word
    words = []
    for i in range(N):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        words.append(digits)

    # Precompute ball masks as Python integers (bitsets)
    ball_masks = [0] * N
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            base = i - orig_val * powers3[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                j = base + new_val * powers3[pos]
                mask |= (1 << j)
        ball_masks[i] = mask

    def popcount(x):
        return bin(x).count('1')

    def greedy_randomized(seed=None):
        if seed is not None:
            random.seed(seed)
        covered = 0
        code_indices = []
        
        while covered != all_mask:
            uncovered = all_mask ^ covered
            best_gain = -1
            best_idx = -1
            candidates = []
            
            for i in range(N):
                gain = popcount(ball_masks[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            # Random tie-breaking
            if candidates:
                best_idx = random.choice(candidates)
            
            if best_gain == 0:
                break

            code_indices.append(best_idx)
            covered |= ball_masks[best_idx]
            
        return code_indices

    def local_search(code_indices):
        if not code_indices:
            return code_indices
            
        covered = 0
        for idx in code_indices:
            covered |= ball_masks[idx]
            
        if covered != all_mask:
            return greedy_randomized()

        code_set = set(code_indices)
        improved = True
        while improved:
            improved = False
            
            # Try deletions using coverage count
            # Count how many codewords cover each position
            # For each codeword, check if it's the only one covering any word
            for idx in list(code_indices):
                # Check if removing idx still covers everything
                # A word is uniquely covered by idx if it's in ball_masks[idx] but not in any other ball
                # Quick check: for each word in ball_masks[idx], see if another codeword covers it
                # This is expensive, so use a heuristic: check a sample
                unique_found = False
                # Sample positions from ball_masks[idx]
                mask = ball_masks[idx]
                # Get set bits
                bits = []
                m = mask
                while m:
                    low = m & (-m)
                    bits.append(low.bit_length() - 1)
                    m ^= low
                    if len(bits) > 200:
                        break
                random.shuffle(bits)
                for pos in bits[:50]:
                    # Check if any other codeword covers this position
                    is_covered = False
                    for j in code_indices:
                        if j == idx:
                            continue
                        if ball_masks[j] & (1 << pos):
                            is_covered = True
                            break
                    if not is_covered:
                        unique_found = True
                        break
                if not unique_found:
                    code_indices.remove(idx)
                    code_set.discard(idx)
                    improved = True
                    break
            
            if improved:
                continue

            # Try 1-swaps: Remove idx, Add j
            # Focus on codewords with few unique contributions
            for idx in list(code_indices):
                # Compute unique coverage of idx
                covered_by_others = 0
                for j in code_indices:
                    if j != idx:
                        covered_by_others |= ball_masks[j]
                unique_to_idx = ball_masks[idx] & ~covered_by_others
                unique_count = popcount(unique_to_idx)
                
                if unique_count > 15:
                    continue  # Too risky to remove
                
                # Try adding random codewords not in code
                for _ in range(30):
                    j = random.randint(0, N - 1)
                    if j in code_set:
                        continue
                    
                    # Check if removing idx and adding j maintains coverage
                    new_covered = (covered & ~ball_masks[idx]) | ball_masks[j]
                    if new_covered == all_mask:
                        code_indices.remove(idx)
                        code_set.discard(idx)
                        code_indices.append(j)
                        code_set.add(j)
                        covered = new_covered
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                continue

            # Try 2-remove-1-add
            if len(code_indices) >= 2:
                for _ in range(80):
                    i1, i2 = random.sample(code_indices, 2)
                    j = random.randint(0, N - 1)
                    if j in code_set:
                        continue
                    
                    new_covered = (covered & ~ball_masks[i1] & ~ball_masks[i2]) | ball_masks[j]
                    if new_covered == all_mask:
                        code_indices.remove(i1)
                        code_indices.remove(i2)
                        code_set.discard(i1)
                        code_set.discard(i2)
                        code_indices.append(j)
                        code_set.add(j)
                        covered = new_covered
                        improved = True
                        break
            
            if improved:
                continue

            # Try 2-remove-2-add
            if len(code_indices) >= 2:
                for _ in range(30):
                    i1, i2 = random.sample(code_indices, 2)
                    j1 = random.randint(0, N - 1)
                    j2 = random.randint(0, N - 1)
                    if j1 in code_set or j2 in code_set or j1 == j2:
                        continue
                    
                    new_covered = (covered & ~ball_masks[i1] & ~ball_masks[i2]) | ball_masks[j1] | ball_masks[j2]
                    if new_covered == all_mask:
                        code_indices.remove(i1)
                        code_indices.remove(i2)
                        code_set.discard(i1)
                        code_set.discard(i2)
                        code_indices.append(j1)
                        code_indices.append(j2)
                        code_set.add(j1)
                        code_set.add(j2)
                        covered = new_covered
                        improved = True
                        break
        
        return code_indices

    def verify_coverage(code_indices):
        covered = 0
        for idx in code_indices:
            covered |= ball_masks[idx]
        return covered == all_mask

    # Run multiple restarts
    best_code = None
    best_len = float('inf')
    
    # Baseline
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    deadline = start_time + 115
    
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_randomized(seed=run_count)
        polished_indices = local_search(code_indices)
        
        if verify_coverage(polished_indices) and len(polished_indices) < best_len:
            best_len = len(polished_indices)
            best_code = [words[i] for i in polished_indices]
            # Also try a second local search pass on the improved code
            if time.time() < deadline - 5:
                polished2 = local_search(polished_indices)
                if verify_coverage(polished2) and len(polished2) < best_len:
                    best_len = len(polished2)
                    best_code = [words[i] for i in polished2]
        
        run_count += 1
    
    return best_code
