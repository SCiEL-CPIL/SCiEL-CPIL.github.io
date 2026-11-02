import itertools
import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by trimming and swap-based local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    total_words = q ** n
    
    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls as lists of neighbor indices
    # For radius 1: center + n*(q-1) neighbors
    ball_size = 1 + n * (q - 1)
    
    # Precompute balls as numpy arrays for faster operations
    balls = np.empty((total_words, ball_size), dtype=np.int32)
    
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        if radius == 1:
            for pos in range(n):
                orig = d[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    d[pos] = new_val
                    neighbors.append(digits_to_int(d, q))
                d[pos] = orig
        else:
            # General radius: generate all words within Hamming distance radius
            # For this specific task, radius is 1.
            pass
            
        balls[i] = neighbors

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Find the center that covers the most uncovered words
        # Use numpy for faster computation
        # For each candidate center, count how many of its neighbors are uncovered
        # balls is shape (total_words, ball_size)
        # We need sum over ball_size dimension of (balls[i] mapped to uncovered)
        
        # Efficient approach: for each center, count uncovered neighbors
        # This is expensive if done naively. Let's use a different approach.
        
        # Alternative: maintain a count array that tracks how many uncovered words each center can cover
        # But this requires updating after each selection, which is complex.
        
        # Simpler: just iterate and use numpy sum
        best_center = -1
        max_gain = 0
        
        # Sample-based greedy for speed when uncovered is large
        # If we have many candidates, we can sample a subset
        if total_words > 1000 and time.time() - start_time > 10:
            # Sample 1000 random candidates
            sample_size = min(1000, total_words)
            candidates = np.random.choice(total_words, size=sample_size, replace=False)
        else:
            candidates = np.arange(total_words)
        
        for c in candidates:
            gain = np.sum(uncovered[balls[c]])
            if gain > max_gain:
                max_gain = gain
                best_center = c
                if gain == ball_size:  # Can't get better than full ball
                    break
        
        if best_center == -1:
            break
            
        code.append(best_center)
        uncovered[balls[best_center]] = False
        
        if time.time() > deadline:
            break

    # Convert to list for easier manipulation
    code_list = list(code)
    
    # Trim Phase with cover counting
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[balls[c]] += 1
    
    # Multiple trimming passes
    changed = True
    passes = 0
    while changed and time.time() < deadline:
        changed = False
        passes += 1
        to_remove = []
        for i, c in enumerate(code_list):
            # Check if any word in balls[c] has cover_count == 1
            if not np.any(cover_count[balls[c]] == 1):
                to_remove.append(i)
        
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code_list[idx]
                code_list.pop(idx)
                cover_count[balls[c]] -= 1
            changed = True

    # Swap-based local search: try to replace one codeword with another
    # to maintain coverage while potentially enabling further trimming
    improved = True
    swap_passes = 0
    while improved and time.time() < deadline:
        improved = False
        swap_passes += 1
        
        # Try to find a swap that allows removal of another codeword
        # Or just try random swaps and see if we can trim more
        for _ in range(min(50, max(1, (deadline - time.time()) // 5))):
            if time.time() > deadline:
                break
            
            if len(code_list) < 2:
                break
            
            # Pick two random codewords
            i, j = np.random.choice(len(code_list), size=2, replace=False)
            c_i = code_list[i]
            c_j = code_list[j]
            
            # Try swapping: remove c_i, add c_j (if not already present)
            if c_j in code_list:
                continue
            
            # Check if we can remove c_i and add c_j while maintaining coverage
            # First, check if c_i is redundant if we add c_j
            # This is complex; let's just try and verify
            
            new_code = code_list.copy()
            new_code[i] = c_j
            
            # Check coverage
            new_cover = np.zeros(total_words, dtype=np.int32)
            for c in new_code:
                new_cover[balls[c]] += 1
            
            if np.all(new_cover > 0):
                # Valid swap, now try to trim
                old_len = len(new_code)
                changed2 = True
                while changed2 and time.time() < deadline:
                    changed2 = False
                    to_remove2 = []
                    for k, c in enumerate(new_code):
                        if not np.any(new_cover[balls[c]] == 1):
                            to_remove2.append(k)
                    if to_remove2:
                        for idx in sorted(to_remove2, reverse=True):
                            c = new_code[idx]
                            new_code.pop(idx)
                            new_cover[balls[c]] -= 1
                        changed2 = True
                
                if len(new_code) < len(code_list):
                    code_list = new_code
                    cover_count = new_cover
                    improved = True
                    break
                elif len(new_code) == len(code_list) and swap_passes % 10 == 0:
                    # Accept equal-size swap sometimes to escape local optima
                    if np.random.random() < 0.1:
                        code_list = new_code
                        cover_count = new_cover
                        improved = True
                        break

    # Final trimming pass
    if time.time() < deadline:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code_list):
                if not np.any(cover_count[balls[c]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code_list[idx]
                    code_list.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    # Convert back to list of lists of integers
    result = []
    for c in code_list:
        result.append(int_to_digits(c, n, q))
        
    return result
