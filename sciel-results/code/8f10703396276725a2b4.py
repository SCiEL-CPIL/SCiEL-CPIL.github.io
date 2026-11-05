import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try n from 26 down to 12? 
    # Or 12 up to 26?
    # Since we have fast cost, we can try 26 first.
    # If 26 fails, try 25, etc.
    # Actually, starting with 26 is risky if it's hard.
    # But with fast cost, it's the best bet for high score.
    
    # Let's try 26, 25, 24... down to 12.
    # If we find a valid one, return it.
    # We should spend most time on 26.
    
    time_per_n = 10.0
    # Give more time to larger n?
    # Let's just iterate down.
    
    for n in range(26, 11, -1):
        if time.time() > deadline:
            break
            
        # Adjust time budget
        remaining = deadline - time.time()
        t_budget = min(time_per_n, remaining * 0.8) # Keep 20% for next
        
        success, colors = local_search_fast(n, t_budget)
        
        if success:
            best_n = n
            best_colors = colors
            # If we found 26, it's likely the max. 
            # We can stop or try 27? 
            # 27 is "new", so unlikely. 
            # Let's stop if we find 26 to save time for safety?
            # Or continue to see if 27 is possible?
            # Given the difficulty, 26 is a huge win.
            # I will break if n == 26 to ensure we return within time.
            if n == 26:
                break

    return {'n': best_n, 'colors': best_colors}

def local_search_fast(n, time_budget):
    m = n * (n - 1) // 2
    colors = np.random.randint(0, 5, m)
    
    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * n - i*(i+1)//2 + (j - i - 1)

    # Initialize A and C
    A = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    C = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
    bad_count = [0] * 5
    
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        A[c][i, j] = 1
        A[c][j, i] = 1
        
    for c in range(5):
        C[c] = A[c] @ A[c]
        # Count bad pairs (u < v, C[c][u, v] > 1)
        # Use upper triangle
        tri = np.triu(C[c], k=1)
        bad_count[c] = np.sum(tri > 1)
        
    total_bad = sum(bad_count)
    if total_bad == 0:
        return True, colors.tolist()
        
    T = 5.0
    T_min = 0.01
    cool = 0.995
    
    start = time.time()
    while time.time() - start < time_budget:
        idx = random.randint(0, m - 1)
        i, j = edges[idx]
        
        old_c = colors[idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
            
        # Calculate delta cost
        delta = 0
        
        # Remove from old_c
        # i, j
        for v in range(n):
            if v == i or v == j: continue
            
            # Update i-v
            d = A[old_c][j, v]
            old_val = C[old_c][i, v]
            new_val = old_val - d
            if i < v:
                if old_val > 1 and new_val <= 1: delta -= 1
                elif old_val <= 1 and new_val > 1: delta += 1
            C[old_c][i, v] = new_val
            C[old_c][v, i] = new_val
            
            # Update j-v
            d = A[old_c][i, v]
            old_val = C[old_c][j, v]
            new_val = old_val - d
            if j < v:
                if old_val > 1 and new_val <= 1: delta -= 1
                elif old_val <= 1 and new_val > 1: delta += 1
            C[old_c][j, v] = new_val
            C[old_c][v, j] = new_val
            
        A[old_c][i, j] = 0
        A[old_c][j, i] = 0
        bad_count[old_c] += delta
        
        # Add to new_c
        delta = 0
        for v in range(n):
            if v == i or v == j: continue
            
            # Update i-v
            d = A[new_c][j, v]
            old_val = C[new_c][i, v]
            new_val = old_val + d
            if i < v:
                if old_val > 1 and new_val <= 1: delta -= 1
                elif old_val <= 1 and new_val > 1: delta += 1
            C[new_c][i, v] = new_val
            C[new_c][v, i] = new_val
            
            # Update j-v
            d = A[new_c][i, v]
            old_val = C[new_c][j, v]
            new_val = old_val + d
            if j < v:
                if old_val > 1 and new_val <= 1: delta -= 1
                elif old_val <= 1 and new_val > 1: delta += 1
            C[new_c][j, v] = new_val
            C[new_c][v, j] = new_val
            
        A[new_c][i, j] = 1
        A[new_c][j, i] = 1
        bad_count[new_c] += delta
        
        total_bad = sum(bad_count)
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            colors[idx] = new_c
            if total_bad == 0:
                return True, colors.tolist()
        else:
            # Revert
            # Re-add to old_c
            delta = 0
            for v in range(n):
                if v == i or v == j: continue
                d = A[old_c][j, v]
                old_val = C[old_c][i, v]
                new_val = old_val + d
                if i < v:
                    if old_val > 1 and new_val <= 1: delta -= 1
                    elif old_val <= 1 and new_val > 1: delta += 1
                C[old_c][i, v] = new_val
                C[old_c][v, i] = new_val
                
                d = A[old_c][i, v]
                old_val = C[old_c][j, v]
                new_val = old_val + d
                if j < v:
                    if old_val > 1 and new_val <= 1: delta -= 1
                    elif old_val <= 1 and new_val > 1: delta += 1
                C[old_c][j, v] = new_val
                C[old_c][v, j] = new_val
            A[old_c][i, j] = 1
            A[old_c][j, i] = 1
            bad_count[old_c] += delta
            
            # Remove from new_c
            delta = 0
            for v in range(n):
                if v == i or v == j: continue
                d = A[new_c][j, v]
                old_val = C[new_c][i, v]
                new_val = old_val - d
                if i < v:
                    if old_val > 1 and new_val <= 1: delta -= 1
                    elif old_val <= 1 and new_val > 1: delta += 1
                C[new_c][i, v] = new_val
                C[new_c][v, i] = new_val
                
                d = A[new_c][i, v]
                old_val = C[new_c][j, v]
                new_val = old_val - d
                if j < v:
                    if old_val > 1 and new_val <= 1: delta -= 1
                    elif old_val <= 1 and new_val > 1: delta += 1
                C[new_c][j, v] = new_val
                C[new_c][v, j] = new_val
            A[new_c][i, j] = 0
            A[new_c][j, i] = 0
            bad_count[new_c] += delta
            
        T *= cool
        if T < T_min:
            T = T_min
            
    return False, colors.tolist()
