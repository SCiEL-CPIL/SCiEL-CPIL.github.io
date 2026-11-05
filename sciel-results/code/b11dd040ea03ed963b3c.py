import time
import random
import math
import heapq

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(math.isqrt(p)) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    L_max = 10000
    
    # State
    r = [random.randint(0, p - 1) for p in primes]
    cov = [0] * (L_max + 1)
    
    # Initialize coverage
    for i, p in enumerate(primes):
        x = r[i]
        while x <= L_max:
            if x > 0:
                cov[x] += 1
            x += p
            
    # Min-heap of bad positions
    bad_heap = [x for x in range(1, L_max + 1) if cov[x] < 2]
    heapq.heapify(bad_heap)
    
    def get_L():
        while bad_heap:
            x = bad_heap[0]
            if cov[x] < 2:
                return x - 1
            else:
                heapq.heappop(bad_heap)
        return L_max

    current_L = get_L()
    best_L = current_L
    best_r = r.copy()
    
    start_time = time.time()
    deadline = start_time + 110
    
    def apply_change(idx, new_r):
        nonlocal current_L, best_L, best_r
        p = primes[idx]
        old_r = r[idx]
        if old_r == new_r:
            return
        
        # Remove old
        x = old_r
        while x <= L_max:
            if x > 0:
                cov[x] -= 1
                if cov[x] < 2:
                    heapq.heappush(bad_heap, x)
            x += p
            
        # Add new
        r[idx] = new_r
        x = new_r
        while x <= L_max:
            if x > 0:
                cov[x] += 1
            x += p
        
        current_L = get_L()
        
        if current_L > best_L:
            best_L = current_L
            best_r = r.copy()
            
        # Prune heap if too large
        if len(bad_heap) > 5 * L_max:
            bad_heap[:] = [x for x in range(1, L_max + 1) if cov[x] < 2]
            heapq.heapify(bad_heap)

    def greedy_extend():
        nonlocal current_L
        while time.time() < deadline and current_L < L_max:
            # Find first bad position
            x_fail = None
            while bad_heap:
                x = bad_heap[0]
                if cov[x] < 2:
                    x_fail = x
                    break
                else:
                    heapq.heappop(bad_heap)
            
            if x_fail is None:
                break
                
            # Try to fix x_fail
            fixed = False
            # Try primes in random order to avoid bias
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for i in indices:
                p = primes[i]
                if x_fail % p == r[i]:
                    continue
                    
                # Check safety
                safe = True
                y = r[i]
                while y <= current_L:
                    if cov[y] < 3:
                        safe = False
                        break
                    y += p
                    
                if safe:
                    apply_change(i, x_fail % p)
                    fixed = True
                    break
                    
            if not fixed:
                break

    # Phase 1: Initial Greedy
    greedy_extend()
    
    # Phase 2: SA
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = r[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        old_L = current_L
        apply_change(i, new_r)
        new_L = current_L
        
        delta = new_L - old_L
        
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            pass
        else:
            apply_change(i, old_r)
            
        T *= cooling
        
        # Periodic Greedy
        if random.random() < 0.05:
            greedy_extend()
            
    return best_r
