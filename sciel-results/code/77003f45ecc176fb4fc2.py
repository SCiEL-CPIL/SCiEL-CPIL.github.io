import numpy as np
import time
import random

def _square_nnz(poly):
    if len(poly) == 0:
        return 0
    c = np.array(poly, dtype=np.int64)
    sq = np.convolve(c, c)
    return np.count_nonzero(sq)

def _is_valid(poly):
    if len(poly) > 41:
        return False
    if len(poly) == 0 or poly[-1] == 0:
        return False
    nnz = sum(1 for x in poly if x != 0)
    if nnz != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _local_search(poly, time_limit=2.0):
    start_time = time.time()
    current = poly[:]
    current_nnz = _square_nnz(current)
    
    while time.time() - start_time < time_limit:
        improved = False
        # 1. Sign flips
        nz_indices = [i for i, x in enumerate(current) if x != 0]
        for i in nz_indices:
            new_poly = current[:]
            new_poly[i] = -new_poly[i]
            if _is_valid(new_poly):
                nnz = _square_nnz(new_poly)
                if nnz < current_nnz:
                    current = new_poly
                    current_nnz = nnz
                    improved = True
                    break
        if improved:
            continue

        # 2. Value changes
        for i in nz_indices:
            old_val = current[i]
            for delta in [-1, 1, -2, 2]:
                new_val = old_val + delta
                if new_val == 0 or abs(new_val) > 64:
                    continue
                new_poly = current[:]
                new_poly[i] = new_val
                if _is_valid(new_poly):
                    nnz = _square_nnz(new_poly)
                    if nnz < current_nnz:
                        current = new_poly
                        current_nnz = nnz
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue

        # 3. Move term
        if nz_indices:
            i = random.choice(nz_indices)
            empty_slots = [j for j in range(41) if current[j] == 0]
            if empty_slots:
                j = random.choice(empty_slots)
                new_poly = current[:]
                new_poly[j] = new_poly[i]
                new_poly[i] = 0
                # Trim check is done in _is_valid
                if _is_valid(new_poly):
                    nnz = _square_nnz(new_poly)
                    if nnz < current_nnz:
                        current = new_poly
                        current_nnz = nnz
                        improved = True
        if not improved:
            break
            
    return current, current_nnz

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_poly = None
    best_nnz = float('inf')
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = []
    
    # Random
    for _ in range(50):
        poly = [0] * 41
        degrees = random.sample(range(41), 12)
        for d in degrees:
            poly[d] = random.choice([-1, 1])
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        if _is_valid(poly):
            candidates.append(poly)
            
    # Structured: Products
    for _ in range(50):
        # A
        deg_a = random.randint(0, 20)
        a_terms = random.sample(range(deg_a + 1), random.randint(2, 4))
        a = [0] * (deg_a + 1)
        for d in a_terms:
            a[d] = random.choice([-1, 1])
        # B
        deg_b = random.randint(0, 20)
        b_terms = random.sample(range(deg_b + 1), random.randint(2, 4))
        b = [0] * (deg_b + 1)
        for d in b_terms:
            b[d] = random.choice([-1, 1])
        prod = np.convolve(np.array(a), np.array(b))
        prod_list = prod.tolist()
        while len(prod_list) > 1 and prod_list[-1] == 0:
            prod_list.pop()
        if _is_valid(prod_list):
            candidates.append(prod_list)

    # Evaluate
    for c in candidates:
        if time.time() > deadline:
            break
        nnz = _square_nnz(c)
        if nnz < best_nnz:
            best_nnz = nnz
            best_poly = c[:]
            
    # Search
    while time.time() < deadline:
        if random.random() < 0.5 and best_poly is not None:
            poly = best_poly[:]
            for _ in range(random.randint(1, 2)):
                nz = [i for i, x in enumerate(poly) if x != 0]
                if not nz: continue
                i = random.choice(nz)
                op = random.random()
                if op < 0.3:
                    poly[i] = -poly[i]
                elif op < 0.6:
                    poly[i] += random.choice([-1, 1])
                    if abs(poly[i]) > 64 or poly[i] == 0:
                        poly[i] = 1
                else:
                    empty = [j for j in range(41) if poly[j] == 0]
                    if empty:
                        j = random.choice(empty)
                        poly[j] = poly[i]
                        poly[i] = 0
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        else:
            poly = [0] * 41
            degrees = random.sample(range(41), 12)
            for d in degrees:
                poly[d] = random.choice([-1, 1])
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
                
        if not _is_valid(poly):
            continue
            
        nnz = _square_nnz(poly)
        
        if nnz < best_nnz:
            best_nnz = nnz
            best_poly = poly[:]
            if time.time() < deadline - 5:
                improved_poly, improved_nnz = _local_search(poly, time_limit=1.0)
                if improved_nnz < best_nnz:
                    best_nnz = improved_nnz
                    best_poly = improved_poly[:]
                    
    if best_poly is None:
        best_poly = [1]*12
        
    return best_poly
