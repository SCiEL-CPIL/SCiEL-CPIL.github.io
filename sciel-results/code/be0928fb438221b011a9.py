"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    A = list(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Initial candidate: small consecutive integers
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    # Try geometric sequences with different bases
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]:
        if time.time() > deadline:
            break
        A = []
        for i in range(20):
            val = base ** i
            if val > 10000:
                break
            A.append(val)
        if len(A) == 20 and _is_valid(A):
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A
    
    # Try random geometric-like sequences
    random.seed(42)
    for _ in range(10000):
        if time.time() > deadline:
            break
        # Generate a random geometric-like sequence
        ratio = random.uniform(1.1, 3.0)
        start = random.randint(1, 5)
        A = []
        for i in range(20):
            val = int(start * (ratio ** i))
            if val > 10000:
                break
            if val not in A and val > 0:
                A.append(val)
        if len(A) == 20 and _is_valid(A):
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A
    
    # Try random sets with small numbers
    for _ in range(5000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 100), 20)
        if _is_valid(A):
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A
    
    # Local optimization: try to improve by swapping elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_A = sorted(best_A)
        for i in range(len(best_A)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_A)):
                if time.time() > deadline:
                    break
                # Try replacing best_A[i] with a random value
                new_val = random.randint(1, 10000)
                if new_val in best_A:
                    continue
                new_A = best_A.copy()
                new_A[i] = new_val
                new_A = sorted(new_A)
                if _is_valid(new_A):
                    s = _score(new_A)
                    if s < best_score:
                        best_score = s
                        best_A = new_A
                        improved = True
                        break
    
    return best_A
# EVOLVE-BLOCK-END
