import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0
    
    total_words = q ** n
    
    # Precompute powers
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    def int_to_base(x, base, length):
        digits = [0] * length
        for i in range(length - 1, -1, -1):
            digits[i] = x % base
            x //= base
        return digits

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        digits = int_to_base(w, q, n)
        neighbors[w].append(w)
        for i in range(n):
            for new_digit in range(q):
                if new_digit != digits[i]:
                    neighbors[w].append(w + (new_digit - digits[i]) * powers[i])
                    
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy attempts with random tie-breaking
    while time.time() < deadline:
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_w = -1
            max_covered = 0
            candidates = []
            
            # Find all words that cover the max number of uncovered words
            # To avoid full scan if possible? No, full scan is fast.
            for w in range(total_words):
                covered = 0
                for nb in neighbors[w]:
                    if nb in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_w = w
                    candidates = [w]
                elif covered == max_covered and covered > 0:
                    candidates.append(w)
                    
            if best_w == -1:
                break
                
            # Randomly pick from candidates to diversify
            chosen_w = random.choice(candidates)
            code.append(chosen_w)
            uncovered.difference_update(neighbors[chosen_w])
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    if best_code is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for w in best_code:
        result.append(int_to_base(w, q, n))
        
    return result
