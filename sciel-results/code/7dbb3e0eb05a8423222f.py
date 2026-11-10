import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Helper functions
    def sumset_size(A):
        s = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                s.add(A[i] + A[j])
        return len(s)
    
    def prodset_size(A):
        s = set()
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                s.add(A[i] * A[j])
        return len(s)
    
    def max_size(A):
        return max(sumset_size(A), prodset_size(A))
    
    def evaluate(A):
        ss = sumset_size(A)
        ps = prodset_size(A)
        return max(ss, ps), ss, ps
    
    # Initial greedy construction
    # Strategy: build a set that has small sumset and small product set
    # Key insight: geometric sequences have small product sets (products of powers)
    # But we need distinct integers <= 10000
    
    # Try a geometric-like sequence with small base
    # For product set: if A = {b^i}, then A*A = {b^(i+j)} which is small
    # For sumset: sums of powers of b can be many
    
    # Let's try base 2: {1, 2, 4, 8, ..., 2^19} but 2^19 > 10000
    # 2^13 = 8192, so we can have at most 14 elements from powers of 2 up to 10000
    # Need 20 elements, so pure geometric won't work
    
    # Alternative: use a mixed approach
    # Start with a good initial set and improve via local search
    
    # Initial candidate: try different starting sets
    candidates = []
    
    # Candidate 1: Powers of 2 up to 10000, then fill with nearby values
    powers_of_2 = [2**i for i in range(14)]  # 1,2,4,...,8192
    # Need 6 more values
    # Add values that don't create too many new sums/products
    # Try adding odd multiples or other structured values
    for extra in range(1, 100):
        if extra not in powers_of_2 and extra <= 10000:
            candidates.append(powers_of_2 + [extra])
            if len(candidates) >= 10:
                break
    
    # Candidate 2: Arithmetic progression
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(2, 22)))
    candidates.append([i*3+1 for i in range(20)])
    
    # Candidate 3: Mixed geometric-arithmetic
    # Small geometric block + arithmetic extension
    for base in [2, 3]:
        geo = []
        v = 1
        while v <= 10000 and len(geo) < 12:
            geo.append(v)
            v *= base
        # Fill remaining with arithmetic
        remaining = 20 - len(geo)
        start = geo[-1] + 1
        arith = [start + i for i in range(remaining)]
        if all(x <= 10000 for x in arith):
            candidates.append(geo + arith)
    
    # Candidate 4: Random-ish structured sets
    import random
    rng = random.Random(42)
    for _ in range(20):
        # Generate a random set with some structure
        # Start with a geometric core
        core_size = rng.randint(8, 15)
        base = rng.choice([2, 3, 4, 5])
        core = []
        v = 1
        while v <= 10000 and len(core) < core_size:
            core.append(v)
            v *= base
        # Add random values
        remaining = 20 - len(core)
        extras = set()
        while len(extras) < remaining:
            x = rng.randint(1, 10000)
            if x not in core:
                extras.add(x)
        candidates.append(core + list(extras))
    
    # Evaluate all candidates and pick the best
    best_A = None
    best_score = float('inf')
    for A in candidates:
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        score, ss, ps = evaluate(A)
        if score < best_score:
            best_score = score
            best_A = sorted(A)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = max_size(best_A)
    
    # Local search: try swapping one element for another
    def local_search(A, current_score, deadline):
        A = sorted(A)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(A)
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = A[i]
                # Try replacing A[i] with a new value
                # Search over a range of possible new values
                # To be efficient, try values near existing ones and some spread
                candidates_to_try = set()
                # Neighbors
                for delta in range(-50, 51):
                    nv = old_val + delta
                    if 1 <= nv <= 10000 and nv != old_val and nv not in A:
                        candidates_to_try.add(nv)
                # Some random probes
                for _ in range(20):
                    nv = random.randint(1, 10000)
                    if nv != old_val and nv not in A:
                        candidates_to_try.add(nv)
                
                for nv in candidates_to_try:
                    if time.time() > deadline:
                        break
                    new_A = A[:i] + [nv] + A[i+1:]
                    new_A = sorted(new_A)
                    if len(set(new_A)) != 20:
                        continue
                    new_score, _, _ = evaluate(new_A)
                    if new_score < current_score:
                        current_score = new_score
                        A = new_A
                        improved = True
                        break  # Restart the outer loop with the new A
            # Also try 2-swaps if time permits
            if time.time() < deadline - 10:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline:
                            break
                        old_i, old_j = A[i], A[j]
                        # Try replacing both
                        for nv_i in [old_i-1, old_i+1, old_i-2, old_i+2]:
                            if nv_i < 1 or nv_i > 10000 or nv_i == old_j:
                                continue
                            for nv_j in [old_j-1, old_j+1, old_j-2, old_j+2]:
                                if nv_j < 1 or nv_j > 10000 or nv_j == old_i or nv_j == nv_i:
                                    continue
                                new_A = A[:i] + [nv_i] + A[i+1:j] + [nv_j] + A[j+1:]
                                new_A = sorted(new_A)
                                if len(set(new_A)) != 20:
                                    continue
                                new_score, _, _ = evaluate(new_A)
                                if new_score < current_score:
                                    current_score = new_score
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
    
    best_A, best_score = local_search(best_A, best_score, deadline)
    
    # Try multiple restarts with different initial points
    if time.time() < deadline - 5:
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            # Generate a new random candidate
            rng2 = random.Random(100 + restart)
            A_new = []
            # Mix of geometric and random
            base = rng2.choice([2, 3, 4])
            v = 1
            while v <= 10000 and len(A_new) < 10:
                A_new.append(v)
                v *= base
            while len(A_new) < 20:
                x = rng2.randint(1, 10000)
                if x not in A_new:
                    A_new.append(x)
            A_new = sorted(A_new)
            if len(set(A_new)) != 20:
                continue
            score_new, _, _ = evaluate(A_new)
            A_new, score_new = local_search(A_new, score_new, deadline)
            if score_new < best_score:
                best_score = score_new
                best_A = A_new
    
    return sorted(best_A)
