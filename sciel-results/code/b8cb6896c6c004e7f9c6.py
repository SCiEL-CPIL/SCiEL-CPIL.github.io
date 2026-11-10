import time
import random
from itertools import combinations

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    A = list(A)
    sumset = set()
    prodset = set()
    for i, a in enumerate(A):
        for b in A[i:]:
            sumset.add(a + b)
            prodset.add(a * b)
    return max(len(sumset), len(prodset))

def _arithmetic_progression(start, d, n=20):
    return [start + i * d for i in range(n)]

def _local_search(A, time_limit=55):
    """Local search: try replacing one element with another."""
    A = list(A)
    best_score = _calc_score(A)
    start = time.time()
    while time.time() - start < time_limit:
        improved = False
        for i in range(len(A)):
            if time.time() - start > time_limit:
                break
            old = A[i]
            # Try replacing A[i] with nearby values
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = old + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A:
                    continue
                A[i] = new_val
                s = _calc_score(A)
                if s < best_score:
                    best_score = s
                    improved = True
                A[i] = old
        if not improved:
            # Try larger moves
            for i in range(len(A)):
                if time.time() - start > time_limit:
                    break
                old = A[i]
                for _ in range(20):
                    new_val = random.randint(1, 10000)
                    if new_val in A:
                        continue
                    A[i] = new_val
                    s = _calc_score(A)
                    if s < best_score:
                        best_score = s
                        improved = True
                    A[i] = old
            if not improved:
                break
    return A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_A = list(range(1, 21))
    best_score = _calc_score(best_A)
    
    # Try arithmetic progressions
    for start in range(1, 100):
        for d in range(1, 10):
            if time.time() > deadline:
                break
            A = _arithmetic_progression(start, d, 20)
            if A[-1] > 10000:
                continue
            s = _calc_score(A)
            if s < best_score:
                best_score = s
                best_A = A
    
    # Local search from best AP
    if time.time() < deadline:
        A_ls, score_ls = _local_search(best_A, time_limit=deadline - time.time())
        if score_ls < best_score:
            best_score = score_ls
            best_A = A_ls
    
    # Try more random starts with local search
    random.seed(42)
    while time.time() < deadline - 5:
        # Random set
        A = sorted(random.sample(range(1, 10001), 20))
        s = _calc_score(A)
        if s < best_score:
            best_score = s
            best_A = A
        elif s < best_score + 5:
            # Close enough to try local search
            if time.time() < deadline - 10:
                A_ls, score_ls = _local_search(A, time_limit=min(10, deadline - time.time()))
                if score_ls < best_score:
                    best_score = score_ls
                    best_A = A_ls
    
    return best_A
