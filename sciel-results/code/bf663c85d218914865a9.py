import numpy as np
import itertools
import time
import math

def construct():
    n = 16
    start_wall = time.time()
    deadline = start_wall + 140.0  # Leave 10s margin for the 150s limit
    
    # Precompute all K4s
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    # Map each edge to the list of K4 indices containing it
    edge_to_k4s = [[[] for _ in range(n)] for _ in range(n)]
    for idx, quad in enumerate(k4s):
        edges = list(itertools.combinations(quad, 2))
        for u, v in edges:
            if u < v:
                edge_to_k4s[u][v].append(idx)
            else:
                edge_to_k4s[v][u].append(idx)

    def evaluate_coloring(colors, k):
        """
        colors: 16x16 matrix
        k: number of colors (0 to k-1)
        Returns total violations and distinct counts
        """
        violations = 0
        distinct_counts = np.zeros(num_k4s, dtype=np.int8)
        for idx, quad in enumerate(k4s):
            a, b, c, d = quad
            c1 = colors[a][b]
            c2 = colors[a][c]
            c3 = colors[a][d]
            c4 = colors[b][c]
            c5 = colors[b][d]
            c6 = colors[c][d]
            d_cnt = len({c1, c2, c3, c4, c5, c6})
            distinct_counts[idx] = d_cnt
            if d_cnt < 5:
                violations += 1
        return violations, distinct_counts

    def search(k, time_limit, seed=42):
        """
        Search for a valid coloring using k colors.
        Returns (valid, colors_matrix)
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # Initialize random coloring
        colors = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                c = np.random.randint(0, k)
                colors[i][j] = c
                colors[j][i] = c
        
        violations, distinct_counts = evaluate_coloring(colors, k)
        
        if violations == 0:
            return True, colors

        # Initialize histograms
        # hist[idx][c] = count of color c in K4 idx
        hist = np.zeros((num_k4s, k), dtype=np.int8)
        for idx, quad in enumerate(k4s):
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            for u, v in edges:
                col = colors[u][v]
                hist[idx][col] += 1
        
        best_violations = violations
        best_colors = colors.copy()
        
        # Simulated Annealing parameters
        T = 1.0
        decay = 0.9995
        min_T = 0.05
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            
            old_col = colors[u][v]
            new_col = np.random.randint(0, k)
            if old_col == new_col:
                continue
            
            # Calculate change in violations
            affected_k4s = edge_to_k4s[min(u,v)][max(u,v)]
            
            delta_violations = 0
            changes = [] # (k4_idx, old_distinct, new_distinct)
            
            for idx in affected_k4s:
                old_count = hist[idx][old_col]
                new_count = hist[idx][new_col]
                
                old_distinct = distinct_counts[idx]
                
                # Update hist
                hist[idx][old_col] -= 1
                if hist[idx][old_col] == 0:
                    old_distinct -= 1
                
                hist[idx][new_col] += 1
                if new_count == 0:
                    old_distinct += 1
                
                new_distinct = old_distinct
                
                old_is_violation = (distinct_counts[idx] < 5)
                new_is_violation = (new_distinct < 5)
                
                if old_is_violation and not new_is_violation:
                    delta_violations -= 1
                elif not old_is_violation and new_is_violation:
                    delta_violations += 1
                
                changes.append((idx, distinct_counts[idx], new_distinct))
                distinct_counts[idx] = new_distinct
            
            # Accept/Reject
            accept = False
            if delta_violations < 0:
                accept = True
            elif delta_violations == 0:
                accept = True
            else:
                if T > min_T:
                    # Use log to avoid underflow/overflow issues in exp if delta is large
                    # prob = exp(-delta / T)
                    # log_prob = -delta / T
                    # Compare with log(random)
                    # Or just check if delta/T is small enough
                    if delta_violations / T < 100: # Threshold to avoid exp underflow
                        prob = math.exp(-delta_violations / T)
                        if np.random.random() < prob:
                            accept = True
            
            if accept:
                colors[u][v] = new_col
                colors[v][u] = new_col
                violations += delta_violations
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors.copy()
                    if violations == 0:
                        return True, colors
            else:
                # Revert changes
                for idx, old_d, new_d in changes:
                    distinct_counts[idx] = old_d
                for idx in affected_k4s:
                    hist[idx][old_col] += 1
                    hist[idx][new_col] -= 1
            
            T *= decay
            if T < min_T:
                T = min_T
                
        return False, best_colors

    def finalize(colors):
        return [int(colors[i][j]) for i in range(16) for j in range(i+1, 16)]

    # Strategy:
    # 1. Try k=13 with a decent time budget.
    # 2. If fail, try k=14.
    # 3. If fail, try k=15.
    # 4. Fallback to seed formula.
    
    # Time allocation:
    # k=13: 45s
    # k=14: 45s
    # k=15: 45s
    # Total: 135s + overhead.
    
    # Try k=13
    t0 = time.time()
    valid, colors_13 = search(13, 45.0, seed=1)
    if valid:
        return finalize(colors_13)
    
    # Try k=14
    if time.time() < deadline - 50:
        t1 = time.time()
        valid, colors_14 = search(14, 45.0, seed=2)
        if valid:
            return finalize(colors_14)
    
    # Try k=15
    if time.time() < deadline - 10:
        t2 = time.time()
        valid, colors_15 = search(15, 45.0, seed=3)
        if valid:
            return finalize(colors_15)
            
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

