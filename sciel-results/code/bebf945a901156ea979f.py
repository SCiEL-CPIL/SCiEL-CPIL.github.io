import time
import math
from itertools import combinations

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Theoretical best found: {0, 1, 3, 9}
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline
    
    # Try structured candidates with geometric growth
    for base in [2, 3, 4, 5, 6, 7]:
        for n in range(3, 10):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            try_set(U)
    
    # Try {0, 1, a, a*b} type sets
    for a in range(2, 50):
        if time.time() > deadline:
            break
        for b in range(2, 30):
            if time.time() > deadline:
                break
            c = a * b
            if c > 1000:
                break
            try_set([0, 1, a, c])
    
    # Try {0, 1, a, b, c} with geometric-like spacing
    for a in range(2, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 50):
            if time.time() > deadline:
                break
            for c in range(b + 1, 150):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 100
    for a in range(1, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 60):
            if time.time() > deadline:
                break
            for c in range(b + 1, 101):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 50
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            for c in range(b + 1, 35):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 51):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Try 6-element sets with geometric growth
    for a in range(1, 6):
        if time.time() > deadline:
            break
        for b in range(a + 1, 12):
            if time.time() > deadline:
                break
            for c in range(b + 1, 20):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 30):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 50):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    # Try sets with one large element and small dense part
    for small_max in range(5, 25):
        if time.time() > deadline:
            break
        for large in range(small_max + 5, small_max * 8 + 1):
            if time.time() > deadline:
                break
            # Small dense part: {0, 1, 2, ..., small_max} plus large
            U = list(range(small_max + 1)) + [large]
            try_set(U)
    
    # Try {0, 1, 2, ..., k} + {m} for various k, m
    for k in range(3, 20):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 8 + 1):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            try_set(U)
    
    # Try arithmetic progressions
    for a in range(1, 15):
        if time.time() > deadline:
            break
        for d in range(1, 20):
            if time.time() > deadline:
                break
            for n in range(4, 10):
                U = [a * i * d for i in range(n)]
                U[0] = 0
                try_set(U)
    
    # Try sets of the form {0, 1, p, p^2} for prime p
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes:
        if time.time() > deadline:
            break
        try_set([0, 1, p, p * p])
        if p * p * p < 5000:
            try_set([0, 1, p, p * p, p * p * p])
    
    # Try sets with Fibonacci-like growth
    fib_sets = []
    a, b = 1, 1
    for _ in range(10):
        fib_sets.append(a)
        a, b = b, a + b
    for n in range(3, 8):
        if time.time() > deadline:
            break
        U = [0] + fib_sets[:n]
        try_set(U)
    
    # Try sets with Lucas-like growth
    lucas = [2, 1]
    for _ in range(10):
        lucas.append(lucas[-1] + lucas[-2])
    for n in range(3, 8):
        if time.time() > deadline:
            break
        U = [0] + lucas[:n]
        try_set(U)
    
    # Try sets with powers of 2
    for n in range(3, 12):
        if time.time() > deadline:
            break
        U = [2**i for i in range(n)]
        try_set(U)
    
    # Try sets with powers of 3
    for n in range(3, 8):
        if time.time() > deadline:
            break
        U = [3**i for i in range(n)]
        try_set(U)
    
    # Try sets with powers of 4
    for n in range(3, 6):
        if time.time() > deadline:
            break
        U = [4**i for i in range(n)]
        try_set(U)
    
    # Try mixed growth patterns
    for r1 in [2, 3]:
        for r2 in [2, 3, 4]:
            for n in range(4, 8):
                if time.time() > deadline:
                    break
                U = [0, 1]
                x = 1
                for i in range(n - 2):
                    x *= r1 if i % 2 == 0 else r2
                    U.append(x)
                try_set(U)
    
    # Try sets with one element much larger than the rest
    for base_size in range(3, 8):
        if time.time() > deadline:
            break
        for max_val in range(50, 200):
            if time.time() > deadline:
                break
            # Try {0, 1, 2, ..., base_size-1, max_val}
            U = list(range(base_size)) + [max_val]
            try_set(U)
            # Try {0, 1, 3, ..., 3^(base_size-2), max_val}
            U = [0, 1]
            x = 3
            for _ in range(base_size - 2):
                U.append(x)
                x *= 3
            U.append(max_val)
            try_set(U)
    
    return best_U
