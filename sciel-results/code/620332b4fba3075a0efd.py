import time
import random
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Start with geometric sequence of powers of 2 (max 14 fit in 10000)
    # Plus some additional numbers
    base = [2**i for i in range(14)]  # 1,2,4,...,8192
    # Need 6 more numbers. Try adding multiples that create collisions
    # Add numbers that are sums of existing elements to create sum collisions
    candidates = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    best_A = base[:14] + candidates[:6]
    best_A = sorted(set(best_A))
    if len(best_A) < 20:
        # Fill with larger numbers
        for x in range(26, 10001):
            if len(best_A) >= 20:
                break
            if x not in best_A:
                best_A.append(x)
        best_A = sorted(best_A)
    
    best_score, best_sum_size, best_prod_size = compute_metrics(best_A)
    
    # Local search: try swapping elements
    random.seed(42)
    
    def try_improve(A, score, sum_sz, prod_sz):
        """Try to improve A by swapping one element."""
        n = len(A)
        # Try replacing each element with a candidate
        A_set = set(A)
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            old = A[i]
            # Try candidates: nearby numbers, or numbers that create collisions
            candidates = set()
            # Numbers near old
            for delta in range(-20, 21):
                c = old + delta
                if 1 <= c <= 10000 and c not in A_set:
                    candidates.add(c)
            # Numbers that are sums/differences of pairs in A
            for j in range(n):
                if j != i:
                    s = A[j] * 2 - old  # reflection
                    if 1 <= s <= 10000 and s not in A_set:
                        candidates.add(s)
            
            for new_val in list(candidates):
                A[i] = new_val
                s_score, s_sz, p_sz = compute_metrics(A)
                A[i] = old
                if s_score < score:
                    return A[:], s_score, s_sz, p_sz
        return None
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        result = try_improve(best_A, best_score, best_sum_size, best_prod_size)
        if result:
            best_A, best_score, best_sum_size, best_prod_size = result
            improved = True
    
    # Try random restarts with different initial sets if time permits
    if time.time() < deadline - 15.0:
        for attempt in range(50):
            if time.time() > deadline - 10.0:
                break
            # Generate random set biased toward small numbers
            A = set()
            A.add(1)
            while len(A) < 20:
                # Bias toward small numbers
                if random.random() < 0.7:
                    A.add(random.randint(1, 50))
                else:
                    A.add(random.randint(1, 10000))
            A = sorted(A)
            if len(set(A)) != 20:
                continue
            score, sz, psz = compute_metrics(A)
            if score < best_score:
                best_A = A
                best_score = score
                best_sum_size = sz
                best_prod_size = psz
                # Local search from here
                improved2 = True
                while improved2 and time.time() < deadline - 10.0:
                    improved2 = False
                    result = try_improve(best_A, best_score, best_sum_size, best_prod_size)
                    if result:
                        best_A, best_score, best_sum_size, best_prod_size = result
                        improved2 = True
    
    return best_A
