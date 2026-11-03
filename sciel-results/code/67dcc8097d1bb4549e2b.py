import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using tetracode product initialization and 
    inverse-index based local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

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

    # Precompute balls
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Tetracode Product Initialization
    # Split positions into two halves
    half_n = n // 2
    pos1 = list(range(half_n))
    pos2 = list(range(half_n, n))
    
    # Generate all half-words
    half_words = []
    for i in range(q ** half_n):
        half_words.append(int_to_digits(i, half_n, q))
    
    # Generate full words by combining half-words
    full_words = []
    for hw1 in half_words:
        for hw2 in half_words:
            full = [0] * n
            for i in range(half_n):
                full[pos1[i]] = hw1[i]
                full[pos2[i]] = hw2[i]
            full_words.append(digits_to_int(full, q))
            
    # Greedy set cover on the full_words
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Sort full_words by some heuristic? Or just iterate?
    # Let's iterate in order.
    for fw in full_words:
        if time.time() > deadline - 30:
            break
        # Check if fw covers any uncovered words
        fw_ball = balls[fw]
        if np.any(uncovered[fw_ball]):
            code.append(fw)
            uncovered[fw_ball] = False
            cover_count[fw_ball] += 1
            
    # Trim
    changed = True
    while changed and time.time() < deadline:
        changed = False
        to_remove = []
        for i, c in enumerate(code):
            if np.any(cover_count[balls[c]] == 1):
                continue
            to_remove.append(i)
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                cover_count[balls[c]] -= 1
            changed = True

    # Build inverse index
    inv_idx = [[] for _ in range(total_words)]
    for c in code:
        for w in balls[c]:
            inv_idx[w].append(c)

    # Local Search with Inverse Index
    code_set = set(code)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle code order
        code_indices = list(range(len(code)))
        np.random.shuffle(code_indices)
        
        for ci in code_indices:
            if time.time() > deadline - 5:
                break
            i = code[ci]
            i_ball = balls[i]
            # Words only covered by i
            only_mask = cover_count[i_ball] == 1
            only_by_i = i_ball[only_mask]
            
            if len(only_by_i) == 0:
                # Can remove i
                code.pop(ci)
                cover_count[i_ball] -= 1
                code_set.discard(i)
                # Update inv_idx
                for w in i_ball:
                    inv_idx[w].remove(i)
                improved = True
                break
                
            # Find candidates that cover all of only_by_i
            # Use the first word to get a small candidate set
            w0 = only_by_i[0]
            candidates = inv_idx[w0]
            
            found = False
            for j in candidates:
                if j in code_set:
                    continue
                # Check if j covers all of only_by_i
                valid = True
                for w in only_by_i:
                    if cover_count[w] == 0 or j not in inv_idx[w]:
                        # j must be in inv_idx[w] to cover w
                        # Actually, we just need to check if w is in balls[j]
                        # But we don't have a fast way to check that without a set.
                        # Let's use the fact that if j is in inv_idx[w0], and we check the rest...
                        # Wait, the check `j not in inv_idx[w]` is O(1) if inv_idx[w] is a set.
                        # Let's convert inv_idx to sets for O(1) membership.
                        # But we need to update them.
                        # Let's just check if w is in balls[j] by using a precomputed set for balls?
                        # No, that's too much memory.
                        # Let's just check if j is in inv_idx[w].
                        # If j is in inv_idx[w0], it covers w0.
                        # For other w in only_by_i, we need to check if j covers w.
                        # We can check if j is in inv_idx[w].
                        pass
                # Let's re-implement the check properly.
                # We need to check if balls[j] contains all words in only_by_i.
                # We can do this by checking if j is in inv_idx[w] for all w in only_by_i.
                # This is O(len(only_by_i)) which is small.
                valid = True
                for w in only_by_i:
                    if j not in inv_idx[w]:
                        valid = False
                        break
                if valid:
                    # Valid swap!
                    # Remove i
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[i_ball] -= 1
                    cover_count[balls[j]] += 1
                    # Update inv_idx
                    for w in i_ball:
                        inv_idx[w].remove(i)
                    for w in balls[j]:
                        inv_idx[w].append(j)
                    improved = True
                    found = True
                    break
                    
            if found:
                break

    # Final trim
    changed = True
    while changed and time.time() < deadline:
        changed = False
        to_remove = []
        for i, c in enumerate(code):
            if np.any(cover_count[balls[c]] == 1):
                continue
            to_remove.append(i)
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                cover_count[balls[c]] -= 1
            changed = True

    # Convert to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result
