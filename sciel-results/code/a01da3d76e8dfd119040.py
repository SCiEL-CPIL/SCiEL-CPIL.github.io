import random
import time

N = 256

def is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        
        # Check if x can be added
        if 2 * x in sums:
            continue
            
        valid = True
        for a in S:
            if (x + a) in sums:
                valid = False
                break
        
        if valid:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 25  # Safety margin before 30s limit
    
    best = None
    best_size = N + 1
    
    # Try deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if all(x % p != 0 for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251])], # Primes
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_sidon(order, N)
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
    # Try many random orderings
    random.seed(42)
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    if best is None:
        # Fallback to ascending greedy
        order = list(range(1, N + 1))
        best = greedy_sidon(order, N)
    
    return best
