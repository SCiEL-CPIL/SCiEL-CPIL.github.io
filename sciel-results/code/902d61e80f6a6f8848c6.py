import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(start_time, deadline)
        
    return _generic_greedy(q, n, radius, start_time, deadline)

def _solve_ternary_6_1(start_time, deadline):
    q = 3
    n = 6
    radius = 1
    total_words = q ** n  # 729
    ball_size = 1 + n * (q - 1)  # 1 + 6*2 = 13
    
    # Precompute powers of q
    q_powers = [1]
    for i in range(1, n):
        q_powers.append(q_powers[-1] * q)
    
    # Precompute all words' digit representations
    digits_of = [[0] * n for _ in range(total_words)]
    for w in range(total_words):
        temp = w
        for pos in range(n):
            digits_of[w][pos] = temp % q
            temp //= q
    
    # Precompute balls as bitmasks
    balls = [0] * total_words
    for w in range(total_words):
        mask = 1 << w
        for pos in range(n):
            orig_val = digits_of[w][pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w + (new_val - orig_val) * q_powers[pos]
                mask |= (1 << new_w)
        balls[w] = mask
    
    # Precompute ball as list of covered words for faster intersection
    ball_lists = []
    for w in range(total_words):
        covered = []
        mask = balls[w]
        for i in range(total_words):
            if mask & (1 << i):
                covered.append(i)
        ball_lists.append(covered)
    
    # Greedy Set Cover with multiple restarts
    best_code = None
    best_size = total_words  # worst case
    
    # Try several different greedy strategies
    for attempt in range(5):
        if time.time() > deadline - 20:  # Leave time for local search
            break
            
        uncovered_mask = (1 << total_words) - 1
        code = []
        
        if attempt == 0:
            # Standard greedy: pick word covering most uncovered
            while uncovered_mask:
                if time.time() > deadline - 20:
                    break
                best_word = -1
                max_covered = 0
                for w in range(total_words):
                    cnt = (uncovered_mask & balls[w]).bit_count()
                    if cnt > max_covered:
                        max_covered = cnt
                        best_word = w
                        if cnt == uncovered_mask.bit_count():
                            break
                if best_word == -1:
                    break
                code.append(best_word)
                uncovered_mask &= ~balls[best_word]
        else:
            # Greedy with random tie-breaking
            random.seed(attempt * 12345 + 67890)
            while uncovered_mask:
                if time.time() > deadline - 20:
                    break
                best_word = -1
                max_covered = 0
                # Find all words with max coverage, pick randomly
                candidates = []
                for w in range(total_words):
                    cnt = (uncovered_mask & balls[w]).bit_count()
                    if cnt > max_covered:
                        max_covered = cnt
                        candidates = [w]
                    elif cnt == max_covered and cnt > 0:
                        candidates.append(w)
                if not candidates:
                    break
                best_word = random.choice(candidates)
                code.append(best_word)
                uncovered_mask &= ~balls[best_word]
        
        # Local search: try to remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    break
                # Check if removing code[i] still covers everything
                # Compute union of all balls except code[i]
                union_mask = 0
                for j in range(len(code)):
                    if j != i:
                        union_mask |= balls[code[j]]
                if union_mask == (1 << total_words) - 1:
                    # Can remove code[i]
                    code.pop(i)
                    changed = True
                    break
        
        # Local search: try swapping one codeword with another to reduce size
        # This is harder; try: for each codeword, see if there's a word that covers 
        # all its unique contributions plus more
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    break
                # Find words uniquely covered by code[i]
                others_union = 0
                for j in range(len(code)):
                    if j != i:
                        others_union |= balls[code[j]]
                unique_to_i = balls[code[i]] & ~others_union
                if unique_to_i == 0:
                    # code[i] is redundant, remove it
                    code.pop(i)
                    improved = True
                    break
                # Try to find a replacement that covers unique_to_i and more
                # This is expensive; skip for now or do limited search
                # For now, just try a few random candidates
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    candidate = random.randint(0, total_words - 1)
                    if candidate in code:
                        continue
                    # Check if replacing code[i] with candidate still covers everything
                    new_union = 0
                    for j in range(len(code)):
                        if j == i:
                            new_union |= balls[candidate]
                        else:
                            new_union |= balls[code[j]]
                    if new_union == (1 << total_words) - 1:
                        # Also check if this reduces total unique coverage
                        # (it doesn't directly, but might enable further removals)
                        code[i] = candidate
                        improved = True
                        break
                if improved:
                    break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we hit the known optimum, stop
            if best_size <= 73:
                break
    
    # Convert to list of digit lists
    if best_code is None:
        best_code = []
    
    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        result.append(digits)
    
    return result

def _to_digits(num, q, n):
    digits = []
    for _ in range(n):
        digits.append(num % q)
        num //= q
    return digits

def _generic_greedy(q, n, radius, start_time, deadline):
    total_words = q ** n
    if total_words > 10000:
        return []
        
    words = list(range(total_words))
    
    # Precompute balls
    balls = [0] * total_words
    for w in range(total_words):
        mask = 1 << w
        # Generate all neighbors within radius
        # For simplicity, only handle radius=1 for now
        if radius == 1:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            q_powers = [1]
            for i in range(1, n):
                q_powers.append(q_powers[-1] * q)
            for pos in range(n):
                orig_val = digits[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = w + (new_val - orig_val) * q_powers[pos]
                    mask |= (1 << new_w)
        balls[w] = mask
    
    uncovered_mask = (1 << total_words) - 1
    code = []
    
    while uncovered_mask:
        if time.time() > deadline:
            break
        best_word = -1
        max_covered = 0
        for w in range(total_words):
            cnt = (uncovered_mask & balls[w]).bit_count()
            if cnt > max_covered:
                max_covered = cnt
                best_word = w
        if best_word == -1:
            break
        code.append(best_word)
        uncovered_mask &= ~balls[best_word]
    
    result = []
    for w in code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        result.append(digits)
    
    return result
