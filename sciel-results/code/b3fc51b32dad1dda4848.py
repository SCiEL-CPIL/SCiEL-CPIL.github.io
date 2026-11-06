"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
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
    deadline = start_time + 75  # Safety margin before 90s limit
    
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    rng = random.Random(42)
    
    def try_candidate(A):
        nonlocal best_A, best_score
        if not _is_valid(A, 20, 10000):
            return
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = sorted(A)
    
    def local_search(A, max_iters=2000):
        """Random perturbation local search."""
        A = sorted(A[:])
        current_score = _score(A)
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 3:
            improved = False
            it += 1
            # Randomly pick an index to perturb
            i = rng.randint(0, 19)
            # Try small deltas
            deltas = rng.sample([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], 3)
            for d in deltas:
                new_A = A[:]
                new_A[i] += d
                if 1 <= new_A[i] <= 10000 and len(set(new_A)) == 20:
                    s = _score(new_A)
                    if s < current_score:
                        current_score = s
                        A = new_A
                        improved = True
                        break
        return A, current_score
    
    def local_search_structured(A, max_iters=500):
        """Structured local search: try swapping elements with nearby values."""
        A = sorted(A[:])
        current_score = _score(A)
        improved = True
        it = 0
        while improved and it < max_iters and time.time() < deadline - 3:
            improved = False
            it += 1
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_A = A[:]
                        new_A[j] = A[j] + delta
                        if 1 <= new_A[j] <= 10000 and len(set(new_A)) == 20:
                            s = _score(new_A)
                            if s < current_score:
                                current_score = s
                                A = new_A
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return A, current_score
    
    # Phase 1: Generate diverse initial candidates
    # Arithmetic progressions
    for start in [1, 2, 3]:
        for diff in [1, 2, 3, 5, 7, 10]:
            A = [start + i * diff for i in range(20)]
            if max(A) <= 10000:
                try_candidate(A)
    
    # Geometric sequences
    for base in [2, 3, 4]:
        A = []
        val = 1
        for _ in range(20):
            if val > 10000:
                break
            A.append(val)
            val = val * base
        if len(A) < 20:
            last = A[-1] if A else 1
            for _ in range(20 - len(A)):
                last = last * 2
                while last in A:
                    last += 1
                if last > 10000:
                    last = 10000 - len(A)
                A.append(last)
        if len(A) == 20:
            try_candidate(A)
    
    # Mixed: geometric then arithmetic
    for base in [2, 3]:
        for length in range(5, 15):
            A = [base ** i for i in range(length)]
            if max(A) > 10000 or len(A) < 3:
                continue
            last = A[-1]
            for i in range(20 - len(A)):
                last += 1
                if last > 10000:
                    break
                if last not in A:
                    A.append(last)
            if len(A) == 20:
                try_candidate(A)
    
    # Start points from inspiration
    start_points = [
        list(range(1, 21)),
        [2**i for i in range(13)] + [8193, 8194, 8195, 8196, 8197, 8198],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 100, 200, 300, 400, 500, 600],
        list(range(1, 10001, 500))[:20],
    ]
    
    for sp in start_points:
        if len(sp) == 20:
            try_candidate(sp)
    
    # Phase 2: Local search on best candidate and diverse starting points
    for sp in start_points[:3]:
        if time.time() < deadline - 10:
            improved_A, improved_score = local_search(sp, max_iters=1000)
            if improved_score < best_score:
                best_score = improved_score
                best_A = improved_A
    
    # Phase 3: More random restarts with local search
    while time.time() < deadline - 5:
        A = rng.sample(range(1, 10001), 20)
        try_candidate(A)
        if time.time() < deadline - 5:
            improved_A, improved_score = local_search(A, max_iters=300)
            if improved_score < best_score:
                best_score = improved_score
                best_A = improved_A
    
    # Phase 4: Final structured local search on best
    if time.time() < deadline - 2:
        improved_A, improved_score = local_search_structured(best_A, max_iters=200)
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
    
    # Phase 5: One more random local search round
    if time.time() < deadline - 1:
        improved_A, improved_score = local_search(best_A, max_iters=500)
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
    
    return sorted(best_A)
# EVOLVE-BLOCK-END
