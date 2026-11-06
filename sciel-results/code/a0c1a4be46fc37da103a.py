import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Fallback: seed coloring for n=9
    def get_seed():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    
    # Target n=34 (known to be achievable)
    n = 34
    
    # 1. Precomputation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Build K4 list: each entry is a tuple of 4 triple indices
    k4_list = []
    for c4 in combinations(range(n), 4):
        a, b, c, d = c4
        t1 = (a, b, c)
        t2 = (a, b, d)
        t3 = (a, c, d)
        t4 = (b, c, d)
        k4_list.append((triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]))
    
    # Build K5 list: each entry is a tuple of 10 triple indices
    k5_list = []
    for c5 in combinations(range(n), 5):
        a, b, c, d, e = c5
        ts = [
            (a, b, c), (a, b, d), (a, b, e), (a, c, d), (a, c, e),
            (a, d, e), (b, c, d), (b, c, e), (b, d, e), (c, d, e)
        ]
        k5_list.append(tuple(triple_to_idx[t] for t in ts))
    
    # Build incidence structures for vectorized counting
    # For K4s: we need to know, for a given set of flips, how many red K4s are created/destroyed.
    # A K4 is red if all 4 edges are red (color 0).
    # Let R_i be 1 if edge i is red, 0 otherwise.
    # A K4 k is red if prod_{j in k} R_j == 1.
    # This is hard to vectorize directly for deltas without matrices.
    # Instead, we use a hybrid approach:
    # 1. Maintain current colors in a NumPy array.
    # 2. For each batch of proposed flips, calculate delta by checking the specific K4s/K5s involved.
    #    Since each edge is in O(n^2) K4s and O(n^3) K5s, pure vectorization of the *whole* graph is memory heavy.
    #    However, we can vectorize the *evaluation* of a batch of candidates against their local neighborhoods.
    
    # Let's optimize the Python loop version with better heuristics and a tight deadline, 
    # but change the structure to be distinct. 
    # Actually, let's try a pure NumPy vectorization for the *checking* part.
    
    # Incidence Matrix for K4s: M4[k4_idx, triple_idx] = 1 if triple is in k4
    # M4 shape: (num_k4s, num_triples)
    num_k4s = len(k4_list)
    M4 = np.zeros((num_k4s, num_triples), dtype=np.uint8)
    for i, k4 in enumerate(k4_list):
        for t_idx in k4:
            M4[i, t_idx] = 1
            
    # Incidence Matrix for K5s: M5[k5_idx, triple_idx] = 1 if triple is in k5
    num_k5s = len(k5_list)
    M5 = np.zeros((num_k5s, num_triples), dtype=np.uint8)
    for i, k5 in enumerate(k5_list):
        for t_idx in k5:
            M5[i, t_idx] = 1
            
    # Precompute "all_red" and "all_blue" conditions?
    # A K4 is a violation if all 4 are red.
    # Sum of red edges in K4 i = sum(M4[i, :] * colors_red) where colors_red is 1 for red.
    # Let C be the color array (0=red, 1=blue).
    # Red count in K4 i = sum(M4[i, :] * (1 - C))
    # Violation if sum == 4.
    # Blue count in K5 i = sum(M5[i, :] * C)
    # Violation if sum == 10.
    
    def get_violations(C):
        # C is numpy array of 0s and 1s
        red_counts = M4.dot(1 - C)
        red_viol = (red_counts == 4).sum()
        blue_counts = M5.dot(C)
        blue_viol = (blue_counts == 10).sum()
        return red_viol + blue_viol

    # Initial coloring: deterministic random
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 2, size=num_triples).astype(np.int8)
    
    total_viol = get_violations(colors)
    if total_viol == 0:
        return {'n': n, 'colors': colors.tolist()}
    
    # Simulated Annealing with Batched Flips
    # We will flip a batch of K edges at a time.
    # Calculating delta for a batch:
    # This is complex because flips interact (two flips in same K4).
    # To keep it simple and fast, we flip ONE edge at a time but use optimized C-extensions via NumPy for the check?
    # No, Python loop is the bottleneck.
    
    # Alternative: Use a lookup table for "current state of K4s/K5s"?
    # Let's stick to a highly optimized Python loop with pre-computed neighbor lists, 
    # but change the randomization and cooling to be distinct.
    
    # Re-initialize neighbor lists for fast access in Python loop
    triple_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        for t_idx in k4:
            triple_k4s[t_idx].append(i)
            
    triple_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        for t_idx in k5:
            triple_k5s[t_idx].append(i)
            
    # Convert to tuples for faster iteration? Or lists are fine.
    
    T = 3.0
    cooling = 0.99999
    min_T = 0.01
    best_viol = total_viol
    best_colors = colors.copy()
    
    # Use a while loop with time check
    # To make it distinct, we will add a "restart" mechanism that is triggered by stagnation, 
    # and we will try n=35 briefly if n=34 fails quickly (though 35 is likely impossible/too hard).
    # Actually, let's just optimize n=34.
    
    steps = 0
    max_steps = 500000 # High limit, time bounded
    
    # Use a different random seed for the SA process
    rand_seed = int(time.time()) % (2**32)
    random.seed(rand_seed)
    
    while time.time() < deadline - 5.0 and total_viol > 0 and steps < max_steps:
        # Pick a random edge
        i = random.randint(0, num_triples - 1)
        old_color = colors[i]
        new_color = 1 - old_color
        
        delta = 0
        
        # Check K4s
        # Optimization: Only check if the edge is part of a potential violation
        # But we don't know "potential" without counting.
        for k4_idx in triple_k4s[i]:
            k4 = k4_list[k4_idx]
            # Count reds among others
            # Unrolling for speed is hard in pure Python, so we rely on the small size (3 others)
            r1 = 1 - colors[k4[0]] if k4[0] != i else 1 - new_color # Wait, if we change i, the "others" are unchanged.
            # The "others" are the 3 edges in k4 that are NOT i.
            red_count_other = 0
            for j in k4:
                if j != i:
                    red_count_other += (1 - colors[j])
            
            # If old was red (0), and others are 3 red, it was a violation.
            if old_color == 0 and red_count_other == 3:
                delta -= 1
            # If new is red (0), and others are 3 red, it becomes a violation.
            if new_color == 0 and red_count_other == 3:
                delta += 1
                
        # Check K5s
        for k5_idx in triple_k5s[i]:
            k5 = k5_list[k5_idx]
            blue_count_other = 0
            for j in k5:
                if j != i:
                    blue_count_other += colors[j]
            
            if old_color == 1 and blue_count_other == 9:
                delta -= 1
            if new_color == 1 and blue_count_other == 9:
                delta += 1
                
        # Acceptance
        if delta < 0:
            colors[i] = new_color
            total_viol += delta
            if total_viol < best_viol:
                best_viol = total_viol
                best_colors = colors.copy()
        elif delta == 0:
            colors[i] = new_color
        else:
            # Metropolis criterion
            if random.random() < math.exp(-delta / T):
                colors[i] = new_color
                total_viol += delta
                
        T *= cooling
        if T < min_T:
            T = min_T
        steps += 1
        
        # Periodic restart if stuck at local minimum for a while
        if steps % 10000 == 0 and total_viol > 50:
             # Small perturbation
             for _ in range(10):
                 i = random.randint(0, num_triples - 1)
                 colors[i] = 1 - colors[i]
             total_viol = get_violations(colors)
             if total_viol < best_viol:
                 best_viol = total_viol
                 best_colors = colors.copy()
        
    if best_viol == 0:
        return {'n': n, 'colors': best_colors.tolist()}
        
    # If n=34 failed, try n=33 (less constraints)
    if time.time() < deadline - 15.0:
        n = 33
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = []
        for c4 in combinations(range(n), 4):
            a, b, c, d = c4
            k4_list.append((triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]))
            
        k5_list = []
        for c5 in combinations(range(n), 5):
            a, b, c, d, e = c5
            ts = [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]
            k5_list.append(tuple(triple_to_idx[t] for t in ts))
            
        triple_k4s = [[] for _ in range(num_triples)]
        for i, k4 in enumerate(k4_list):
            for t_idx in k4:
                triple_k4s[t_idx].append(i)
                
        triple_k5s = [[] for _ in range(num_triples)]
        for i, k5 in enumerate(k5_list):
            for t_idx in k5:
                triple_k5s[t_idx].append(i)
                
        # Reset M4, M5 for n=33 if we want to use get_violations, but we'll use a simpler inline check for speed here
        # Actually, let's just use the same SA logic but with the new lists
        
        rng = np.random.default_rng(123)
        colors = rng.integers(0, 2, size=num_triples).astype(np.int8)
        
        # Quick violation count
        def count_viol_simple(colors, k4s, k5s):
            v = 0
            for k4 in k4s:
                if all(colors[j] == 0 for j in k4): v += 1
            for k5 in k5s:
                if all(colors[j] == 1 for j in k5): v += 1
            return v
            
        total_viol = count_viol_simple(colors, k4_list, k5_list)
        if total_viol == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        T = 2.0
        cooling = 0.99999
        min_T = 0.01
        steps = 0
        
        while time.time() < deadline - 5.0 and total_viol > 0 and steps < 200000:
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            for k4_idx in triple_k4s[i]:
                k4 = k4_list[k4_idx]
                red_count_other = 0
                for j in k4:
                    if j != i: red_count_other += (1 - colors[j])
                if old_color == 0 and red_count_other == 3: delta -= 1
                if new_color == 0 and red_count_other == 3: delta += 1
                
            for k5_idx in triple_k5s[i]:
                k5 = k5_list[k5_idx]
                blue_count_other = 0
                for j in k5:
                    if j != i: blue_count_other += colors[j]
                if old_color == 1 and blue_count_other == 9: delta -= 1
                if new_color == 1 and blue_count_other == 9: delta += 1
                
            if delta < 0:
                colors[i] = new_color
                total_viol += delta
            elif delta == 0:
                colors[i] = new_color
            else:
                if random.random() < math.exp(-delta / T):
                    colors[i] = new_color
                    total_viol += delta
                    
            T *= cooling
            if T < min_T: T = min_T
            steps += 1
            
        if total_viol == 0:
            return {'n': n, 'colors': colors.tolist()}
            
    return get_seed()
