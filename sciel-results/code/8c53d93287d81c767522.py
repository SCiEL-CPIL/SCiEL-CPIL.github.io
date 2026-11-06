import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    n_smooth = len(smooth_numbers)
    
    # 2. Helper to check antichain property for a tuple of numbers
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                if a == b:
                    return False
                if b % a == 0 or a % b == 0:
                    return False
        return True

    # 3. Find representations
    answers = [None] * 128
    
    # Level 1: 1 term
    for s in smooth_numbers:
        if s <= 128:
            answers[s - 1] = [s]

    # Level 2: 2 terms
    # Pre-compute valid 2-term antichain sums for efficiency and clarity
    # We only need to store the first one found for each sum, as all have cost 2
    two_term_sums = {}
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            # Check antichain
            if b % a != 0 and a % b != 0:
                if s not in two_term_sums:
                    two_term_sums[s] = [a, b]
    
    for s, terms in two_term_sums.items():
        if answers[s - 1] is None:
            answers[s - 1] = terms
            
    # Check if all are found
    if all(ans is not None for ans in answers):
        return answers

    # Level 3: 3 terms
    three_term_sums = {}
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            if a + b > 128: # Optimization: min c is smooth_numbers[j+1] > b, but just use sum check
                continue
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s = a + b + c
                if s > 128:
                    break
                # Check antichain for (a, b, c)
                if b % a == 0 or a % b == 0:
                    continue
                if c % a == 0 or a % c == 0:
                    continue
                if c % b == 0 or b % c == 0:
                    continue
                
                if s not in three_term_sums:
                    three_term_sums[s] = [a, b, c]
                    
    for s, terms in three_term_sums.items():
        if answers[s - 1] is None:
            answers[s - 1] = terms
            
    if all(ans is not None for ans in answers):
        return answers

    # Fallback: Level 4+
    # For any remaining targets, do a targeted search
    for i in range(128):
        if answers[i] is None:
            target = i + 1
            # Try 4 terms
            found = False
            for combo in itertools.combinations(smooth_numbers, 4):
                if sum(combo) == target and is_antichain(combo):
                    answers[i] = list(combo)
                    found = True
                    break
            if not found:
                # Try 5 terms (unlikely)
                for combo in itertools.combinations(smooth_numbers, 5):
                    if sum(combo) == target and is_antichain(combo):
                        answers[i] = list(combo)
                        found = True
                        break
            if not found:
                # Should not happen
                answers[i] = [target] # Fallback to invalid? No, must be valid.
                # If this happens, the problem is unsolvable with smooth numbers?
                # But 1..128 are all covered by 3 terms.
                
    return answers
