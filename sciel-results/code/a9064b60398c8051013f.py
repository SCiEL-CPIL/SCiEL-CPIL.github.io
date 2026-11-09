import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions as index arrays
    prog_a = []
    prog_d = []
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 < n:
                prog_a.append(a)
                prog_d.append(d)
    
    num_progs = len(prog_a)
    if num_progs == 0:
        return list(range(n))
    
    prog_a = np.array(prog_a, dtype=np.int32)
    prog_d = np.array(prog_d, dtype=np.int32)
    
    # Index arrays for vectorized access
    idx0 = prog_a
    idx1 = prog_a + prog_d
    idx2 = prog_a + 2 * prog_d
    idx3 = prog_a + 3 * prog_d
    
    def count_violations_vec(colors):
        """Vectorized violation count using NumPy"""
        c0 = colors[idx0]
        c1 = colors[idx1]
        c2 = colors[idx2]
        c3 = colors[idx3]
        
        # Violation if < 3 distinct colors, i.e., <= 2 distinct
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        three_same = ((c0 == c1) & (c1 == c2) & (c2 != c3)) | \
                     ((c0 == c1) & (c1 != c2) & (c2 == c3)) | \
                     ((c0 != c1) & (c1 == c2) & (c2 == c3)) | \
                     ((c0 == c1) & (c1 != c2) & (c2 != c3) & (c1 == c3))
        two_pairs_1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        two_pairs_2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        two_pairs_3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        
        violation = all_same | three_same | two_pairs_1 | two_pairs_2 | two_pairs_3
        return np.sum(violation)
    
    def count_distinct_colors(colors):
        """Count number of distinct color labels used"""
        return len(set(colors.tolist()))
    
    # Group-theoretic approach: map positions to F_2^7
    # Position i maps to its 7-bit representation
    # Color = linear functional L(i) = dot(v, bits(i)) mod 2
    # This gives 2 colors, but we can use multiple functionals
    
    # Strategy: Try combinations of linear functionals to create multi-color schemes
    # Each functional gives a binary component; combine k functionals to get 2^k colors
    
    best_result = None
    best_score = float('inf')
    
    # Try 1-3 linear functionals (2, 4, 8 colors)
    for num_funcs in range(1, 4):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate functionals (non-zero vectors in F_2^7)
        # There are 127 non-zero vectors; we'll sample strategically
        candidates = []
        
        # Include basis vectors
        for i in range(7):
            v = [0]*7
            v[i] = 1
            candidates.append(tuple(v))
        
        # Include some random vectors
        np.random.seed(42)
        for _ in range(50):
            v = np.random.randint(0, 2, 7)
            if np.any(v):
                candidates.append(tuple(v))
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        # Try combinations of num_funcs functionals
        from itertools import combinations
        for combo in combinations(range(len(candidates)), num_funcs):
            if time.time() > deadline - 10:
                break
            
            # Build color array using these functionals
            colors = np.zeros(n, dtype=np.int32)
            for j, func_idx in enumerate(combo):
                v = np.array(candidates[func_idx])
                # For each position, compute dot product mod 2
                for i in range(n):
                    bits = [(i >> b) & 1 for b in range(7)]
                    if sum(b*v[b] for b in range(7)) % 2 == 1:
                        colors[i] |= (1 << j)
            
            viol = count_violations_vec(colors)
            if viol == 0:
                num_colors = count_distinct_colors(colors)
                if num_colors < best_score:
                    best_score = num_colors
                    best_result = colors.copy()
            
            if best_result is not None and best_score <= 2**num_funcs:
                break
    
    # If we found a valid solution with group-theoretic approach, use it
    if best_result is not None:
        return best_result.tolist()
    
    # Fallback: try periodic constructions
    for period in range(2, 33):
        if time.time() > deadline - 5:
            break
        
        # Try all possible colorings with this period
        # Use a greedy/local approach
        base = np.zeros(period, dtype=np.int32)
        
        # Try to assign colors to minimize violations
        np.random.seed(period)
        for attempt in range(1000):
            base = np.random.randint(0, 4, period)
            
            # Expand to full length
            colors = np.tile(base, n // period + 1)[:n]
            
            viol = count_violations_vec(colors)
            if viol == 0:
                num_colors = count_distinct_colors(colors)
                if num_colors < best_score:
                    best_score = num_colors
                    best_result = colors.copy()
                break
    
    if best_result is not None:
        return best_result.tolist()
    
    # Final fallback: use as few colors as possible with direct search
    # Try 2 colors
    for c0 in range(2):
        for c1 in range(2):
            if c0 == c1:
                continue
            colors = np.zeros(n, dtype=np.int32)
            colors[0] = c0
            colors[1] = c1
            # Simple alternating pattern
            for i in range(2, n):
                colors[i] = colors[i-1] if colors[i-1] != colors[i-2] else colors[i-2]
            
            viol = count_violations_vec(colors)
            if viol == 0:
                return colors.tolist()
    
    # Use 4 colors with structured pattern
    colors4 = np.array([(i % 2) * 2 + ((i // 2) % 2) for i in range(n)], dtype=np.int32)
    v4 = count_violations_vec(colors4)
    if v4 == 0:
        return colors4.tolist()
    
    # Trivially valid: all distinct colors
    return list(range(n))
