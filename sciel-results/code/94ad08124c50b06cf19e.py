import time
import random

def construct():
    deadline = time.time() + 55.0
    n = 20
    max_val = 10000
    
    def evaluate(A):
        sums = set()
        prods = set()
        # A is sorted list of n distinct ints
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    def gen_grid(p1, p2, d1, d2):
        """Generate d1*d2 grid of p1^i * p2^j"""
        vals = set()
        for i in range(d1):
            for j in range(d2):
                v = (p1 ** i) * (p2 ** j)
                if v <= max_val:
                    vals.add(v)
        vals = list(vals)
        vals.sort()
        return vals[:n] if len(vals) >= n else None

    candidates = []
    
    # 1. 2^i * 3^j grids
    for d1 in range(2, 6):
        for d2 in range(2, 6):
            if d1 * d2 >= n:
                g = gen_grid(2, 3, d1, d2)
                if g: candidates.append(g)
    # 2. 2^i * 5^j grids
    for d1 in range(2, 6):
        for d2 in range(2, 6):
            if d1 * d2 >= n:
                g = gen_grid(2, 5, d1, d2)
                if g: candidates.append(g)
    # 3. 3^i * 5^j grids
    for d1 in range(2, 6):
        for d2 in range(2, 6):
            if d1 * d2 >= n:
                g = gen_grid(3, 5, d1, d2)
                if g: candidates.append(g)
    
    # 4. First 20 5-smooth numbers
    smooths = []
    i2 = i3 = i5 = 0
    last = 0
    while len(smooths) < n + 10:
        next_2 = smooths[i2] * 2 if i2 < len(smooths) else 10**9
        next_3 = smooths[i3] * 3 if i3 < len(smooths) else 10**9
        next_5 = smooths[i5] * 5 if i5 < len(smooths) else 10**9
        m = min(next_2, next_3, next_5)
        if m > max_val: break
        if m != last:
            smooths.append(m)
            last = m
        if m == next_2: i2 += 1
        if m == next_3: i3 += 1
        if m == next_5: i5 += 1
    if len(smooths) >= n:
        candidates.append(smooths[:n])
        
    # 5. First 20 7-smooth numbers?
    # Might be too sparse, but let's try a mixed pool for search.
    
    best_set = None
    best_score = float('inf')
    
    # Search Pool: The union of all candidate elements + some random smooths
    # This allows the search to move between structures.
    pool = set()
    for c in candidates:
        pool.update(c)
    
    # Add more 5-smooths to the pool
    i2 = i3 = i5 = 0
    last = 0
    s_list = []
    while len(s_list) < 100:
        next_2 = s_list[i2] * 2 if i2 < len(s_list) else 10**9
        next_3 = s_list[i3] * 3 if i3 < len(s_list) else 10**9
        next_5 = s_list[i5] * 5 if i5 < len(s_list) else 10**9
        m = min(next_2, next_3, next_5)
        if m > max_val: break
        if m != last:
            s_list.append(m)
            last = m
        if m == next_2: i2 += 1
        if m == next_3: i3 += 1
        if m == next_5: i5 += 1
    pool.update(s_list)
    pool = list(pool)
    
    for start_set in candidates:
        if time.time() > deadline:
            break
        
        current = start_set[:]
        current_score = evaluate(current)
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.9995
        # Max 10000 steps per restart or until time limit
        steps = 0
        while time.time() < deadline and steps < 20000:
            steps += 1
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            
            # 80% from pool, 20% random
            if random.random() < 0.8:
                new_val = random.choice(pool)
            else:
                new_val = random.randint(1, max_val)
                
            if new_val in current:
                continue
                
            current[idx] = new_val
            new_score = evaluate(current)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
            else:
                current[idx] = old_val
                
            T *= cooling
            if T < 0.1:
                break
                
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set
