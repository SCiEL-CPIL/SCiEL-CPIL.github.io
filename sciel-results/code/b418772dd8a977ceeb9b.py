"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 80.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    answers = []
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Fallback: use brute force with relaxed constraints if needed
            # But this shouldn't happen with proper search
            break
        answer = None
        # Try fewer terms first to minimize total summands
        for count in (1, 2, 3):
            if time.time() - start_time > timeout:
                break
            candidates = [x for x in smooth if x <= n]
            for terms in combinations(candidates, count):
                if sum(terms) == n and all(b % a != 0 for a, b in combinations(terms, 2)):
                    answer = list(terms)
                    break
            if answer is not None:
                break
        answers.append(answer)
    return answers
# EVOLVE-BLOCK-END
