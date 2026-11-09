import time
import random

def _is_smooth(n, allowed_primes):
    """Check if all prime factors of n are in allowed_primes."""
    if n <= 1:
        return True
    temp = n
    for p in allowed_primes:
        while temp % p == 0:
            temp //= p
    return temp == 1

def construct():
    deadline = time.time() + 85
    N = 12
    M = 210  # 2 * 3 * 5 * 7
    
    # We search for x_i in [1, 47] such that x_i + x_j is 7-smooth.
    # Then a_i = 210 * x_i <= 10000.
    # Prime factors of a_i + a_j = 210(x_i + x_j) will be subset of {2,3,5,7} U P(x_i+x_j).
    # If x_i+x_j is 7-smooth, union is {2,3,5,7}. Size 4.
    
    allowed_base = {2, 3, 5, 7}
    
    # Precompute valid pairs for x in 1..47
    max_x = 47
    valid_pairs = {}
    for i in range(1, max_x + 1):
        valid_pairs[i] = set()
        for j in range(i + 1, max_x + 1):
            if _is_smooth(i + j, allowed_base):
                valid_pairs[i].add(j)
                valid_pairs[j].add(i)
                
    # Greedy search for a clique of size N
    best_clique = None
    best_score = 100
    
    # Try multiple random starts
    # Increase attempts to ensure we find the clique if it exists
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
            
        # Randomly pick a starting node
        start = random.randint(1, max_x)
        current = [start]
        # Greedily add nodes that connect to all current nodes
        candidates = list(range(1, max_x + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if c in current:
                continue
            # Check if c connects to all in current
            ok = True
            for d in current:
                if c not in valid_pairs[d]:
                    ok = False
                    break
            if ok:
                current.append(c)
                if len(current) >= N:
                    break
                    
        if len(current) >= N:
            # Verify and score
            pr = set()
            for i in range(N):
                for j in range(i + 1, N):
                    s = current[i] + current[j]
                    # Factor s
                    temp = s
                    for p in [2, 3, 5, 7]:
                        while temp % p == 0:
                            pr.add(p)
                            temp //= p
                    if temp > 1:
                        # Should not happen if valid_pairs was correct
                        t = temp
                        f = 11
                        while f * f <= t:
                            while t % f == 0:
                                pr.add(f)
                                t //= f
                            f += 2
                        if t > 1:
                            pr.add(t)
            sc = len(pr)
            if sc < best_score:
                best_score = sc
                best_clique = current[:]
            if best_score <= 4:
                break
                
    if best_clique is not None:
        result = [M * x for x in best_clique[:N]]
        if len(set(result)) == N and all(1 <= x <= 10000 for x in result):
            return result
            
    # Fallback: Local search to minimize prime count
    def score_scaled(xs):
        pr = set()
        for i in range(N):
            for j in range(i + 1, N):
                t = xs[i] + xs[j]
                temp = t
                f = 2
                while f * f <= temp:
                    while temp % f == 0:
                        pr.add(f)
                        temp //= f
                    f += 1 if f == 2 else 2
                if temp > 1:
                    pr.add(temp)
        return len(pr)
        
    best_x = list(range(1, N + 1))
    best_s = score_scaled([M * v for v in best_x])
    
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        x = best_x[:]
        i = random.randint(0, N - 1)
        old = x[i]
        new = random.randint(1, 47)
        if new in x:
            continue
        x[i] = new
        s = score_scaled([M * v for v in x])
        if s < best_s:
            best_s = s
            best_x = x[:]
            
    result = [M * x for x in best_x]
    if len(set(result)) == N and all(1 <= x <= 10000 for x in result):
        return result
        
    return [M * x for x in list(range(1, N + 1))]
