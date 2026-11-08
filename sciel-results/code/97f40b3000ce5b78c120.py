import time
import math
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Compute sum set
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):  # i<=j since addition is commutative
            sum_set.add(U[i] + U[j])
    
    # Compute difference set
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0.0
    
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0  # or slightly above
    
    log_ratio = math.log(ratio)
    log_max = math.log(2 * maxU + 1)
    
    if log_max <= 0:
        return 1.0
    
    return 1.0 + log_ratio / log_max

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def try_set(U):
        nonlocal best_score, best_set
        if time.time() > deadline - 5:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Try the seed
    try_set([0, 1, 3])
    
    # Exhaustive search for small sets
    # Size 4: {0, a, b, c} with 1 <= a < b < c <= 30
    for a in range(1, 31):
        for b in range(a + 1, 31):
            for c in range(b + 1, 31):
                if time.time() > deadline - 5:
                    return best_set
                try_set([0, a, b, c])
    
    # Size 5: {0, a, b, c, d} with 1 <= a < b < c < d <= 25
    for a in range(1, 26):
        for b in range(a + 1, 26):
            for c in range(b + 1, 26):
                for d in range(c + 1, 26):
                    if time.time() > deadline - 5:
                        return best_set
                    try_set([0, a, b, c, d])
    
    # Size 6: {0, a, b, c, d, e} with 1 <= a < b < c < d < e <= 20
    for a in range(1, 21):
        for b in range(a + 1, 21):
            for c in range(b + 1, 21):
                for d in range(c + 1, 21):
                    for e in range(d + 1, 21):
                        if time.time() > deadline - 5:
                            return best_set
                        try_set([0, a, b, c, d, e])
    
    # Try some structured sets
    # Geometric progressions
    for base in [2, 3, 4, 5]:
        for n in range(3, 10):
            if time.time() > deadline - 5:
                return best_set
            U = [0]
            v = 1
            for i in range(n):
                U.append(v)
                v *= base
            if U[-1] < 100000:
                try_set(U)
    
    # Try powers of 2
    for n in range(3, 15):
        if time.time() > deadline - 5:
            return best_set
        U = [0] + [2**i for i in range(n)]
        if U[-1] < 1000000:
            try_set(U)
    
    # Try sets with one large gap
    for gap in [2, 3, 5, 8, 13, 21]:
        for n in range(2, 8):
            if time.time() > deadline - 5:
                return best_set
            # {0, 1, gap, 2*gap, 3*gap, ...}
            U = [0, 1]
            for i in range(2, n + 2):
                U.append(i * gap)
            if U[-1] < 50000:
                try_set(U)
    
    # Random search
    import random
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 5:
            return best_set
        n = random.randint(4, 8)
        max_val = random.randint(10, 100)
        U = [0] + sorted(random.sample(range(1, max_val), n - 1))
        try_set(U)
    
    return best_set
