import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s out of 60s
    
    n = 20
    max_val = 10000
    
    def score(A):
        A = list(set(A))
        if len(A) != n:
            return float('inf')
        if max(A) > max_val:
            return float('inf')
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_set = list(range(1, n+1))
    
    # Helper to generate GP
    def gen_gp(ratio, start, length):
        A = []
        val = start
        for _ in range(length):
            A.append(val)
            val *= ratio
        return A
    
    # Generate 2,3-smooth numbers (Hamming numbers) up to max_val
    smooth_numbers = []
    a = b = c = 0
    while len(smooth_numbers) < 200:
        next_val = min(2**a if a < 20 else float('inf'),
                       3**b if b < 20 else float('inf'),
                       6**(a//2) * 2**(a%2) if a < 20 else float('inf'))
        # Better: generate all 2^i * 3^j
        break
    
    # Generate 2,3-smooth numbers properly
    smooth = set()
    for i in range(15):
        for j in range(15):
            val = (2**i) * (3**j)
            if val <= max_val:
                smooth.add(val)
    smooth_list = sorted(smooth)
    
    # Candidate 1: Single GP
    for ratio in [2, 3]:
        for start in [1, 2]:
            A = gen_gp(ratio, start, n)
            if max(A) <= max_val:
                s = score(A)
                if s < best_score:
                    best_score = s
                    best_set = A
    
    # Candidate 2: Two clusters GP
    for ratio in [2, 3]:
        for start1 in [1, 2]:
            for start2 in [3, 4, 5, 10]:
                A1 = gen_gp(ratio, start1, 10)
                A2 = gen_gp(ratio, start2, 10)
                A = A1 + A2
                if len(set(A)) == n and max(A) <= max_val:
                    s = score(A)
                    if s < best_score:
                        best_score = s
                        best_set = A
    
    # Candidate 3: Subsets of 2,3-smooth numbers
    # Pick 20 distinct smooth numbers with good structure
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        # Mix of low and high smooth numbers
        A = random.sample(smooth_list, n)
        s = score(A)
        if s < best_score:
            best_score = s
            best_set = list(A)
    
    # Also try structured subsets: consecutive powers
    for base_combo in [(2,3), (3,2), (2,2,3), (3,3,2)]:
        A = []
        for i in range(n):
            # Cycle through bases
            base = base_combo[i % len(base_combo)]
            exp = i // len(base_combo) + 1
            val = base ** exp
            if val <= max_val and val not in A:
                A.append(val)
        if len(A) == n and max(A) <= max_val:
            s = score(A)
            if s < best_score:
                best_score = s
                best_set = list(A)
    
    # Local search with targeted neighbor generation
    def local_search(A, time_limit):
        A = list(A)
        current_score = score(A)
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            i = random.randint(0, n-1)
            old_i = A[i]
            
            # Generate targeted neighbors:
            # 1. Smooth numbers near current value
            # 2. Slightly perturbed current value
            # 3. Random from smooth numbers
            candidates = set()
            
            # Smooth numbers within factor 2 of current
            for sn in smooth_list:
                if 0.5 * old_i <= sn <= 2.0 * old_i:
                    candidates.add(sn)
            
            # Small perturbations
            for delta in [-3, -2, -1, 1, 2, 3]:
                nv = old_i + delta
                if 1 <= nv <= max_val:
                    candidates.add(nv)
            
            # Some random smooth numbers
            for _ in range(5):
                candidates.add(random.choice(smooth_list))
            
            # Some random numbers
            for _ in range(3):
                candidates.add(random.randint(1, max_val))
            
            # Remove existing elements (except A[i])
            existing = set(A)
            existing.discard(old_i)
            candidates = [c for c in candidates if c not in existing]
            
            random.shuffle(candidates)
            
            for new_val in candidates[:10]:  # Try up to 10 candidates
                A[i] = new_val
                s = score(A)
                if s < current_score:
                    current_score = s
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    A[i] = old_i
            
            if not improved:
                no_improve_count += 1
                # Occasional larger move: replace two elements
                if no_improve_count % 10 == 0:
                    j = random.randint(0, n-1)
                    if j != i:
                        old_j = A[j]
                        new_i_val = random.choice(smooth_list)
                        new_j_val = random.choice(smooth_list)
                        if new_i_val not in A and new_j_val not in A and new_i_val != new_j_val:
                            A[i] = new_i_val
                            A[j] = new_j_val
                            s = score(A)
                            if s < current_score:
                                current_score = s
                                improved = True
                                no_improve_count = 0
                            else:
                                A[i] = old_i
                                A[j] = old_j
            
            if time.time() > time_limit - 5:
                break
                
        return A, current_score

    # Run local search on best found so far
    if time.time() < deadline - 10:
        best_set, best_score = local_search(best_set, deadline - 10)
        
    return best_set
