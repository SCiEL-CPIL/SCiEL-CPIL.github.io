import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    
    # Precompute 4-term progressions: (a, a+d, a+2d, a+3d) with d > 0, all < n
    progressions = []
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e < n:
                progressions.append((a, b, c, e))
    
    num_progs = len(progressions)
    prog_array = np.array(progressions, dtype=np.int32)
    
    # For each position, list of (prog_idx, pos_in_prog)
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(progressions):
        prog_at[a].append((idx, 0))
        prog_at[b].append((idx, 1))
        prog_at[c].append((idx, 2))
        prog_at[e].append((idx, 3))
        
    # Convert prog_at to lists of indices for faster access
    prog_indices_at = [[] for _ in range(n)]
    for i in range(n):
        prog_indices_at[i] = [p[0] for p in prog_at[i]]
        
    def count_violations(colors):
        """
        Vectorized count of violations.
        A progression is violated if it has < 3 distinct colors.
        This means it has exactly 1 or exactly 2 distinct colors.
        
        For a progression (a,b,c,e):
        1 distinct color: c[a]==c[b]==c[c]==c[e]
        2 distinct colors: 
           - c[a]==c[b]==c[c] != c[e]
           - c[a]==c[b] != c[c] == c[e]
           - c[a]==c[c]==c[e] != c[b]
           - c[b]==c[c]==c[e] != c[a]
           - c[a]==c[e] != c[b] != c[c]
           - c[a]==c[b] != c[c] != c[e]
           - c[a]==c[c] != c[b] != c[e]
           - c[a]==c[e] != c[b] == c[c] -- wait, this is covered by pairs.
           
        Actually, it's easier to count distinct colors directly or use a bitwise trick if colors are small.
        Since k is small (3 or 4), we can just check the conditions.
        
        Let's use a simpler logic for vectorization:
        distinct_count < 3 is equivalent to:
        (c0==c1 and c1==c2 and c2==c3) OR 
        (c0==c1 and c1==c2 and c2!=c3) OR
        (c0==c1 and c1!=c2 and c2==c3) OR
        (c0!=c1 and c1==c2 and c2==c3) OR
        (c0==c2 and c2==c3 and c0!=c1) OR
        (c0==c1 and c1!=c2 and c2!=c3 and c1!=c3) -- this is getting complicated.
        
        Alternative: 
        Number of distinct colors is 1 if all equal.
        Number of distinct colors is 2 if not all equal, but some pairs are equal.
        
        Let's just compute the number of distinct colors for each progression using a loop over the small number of progressions per element change? 
        No, we need fast evaluation.
        
        Let's try:
        For each progression, let c0, c1, c2, c3 be the colors.
        It is valid if:
        (c0 != c1) or (c0 != c2) or (c0 != c3) ... no.
        
        Valid if:
        Not (all 4 same)
        And Not (exactly 2 colors used in specific patterns that result in <3 distinct)
        
        Actually, "at least 3 distinct colors" means:
        It is NOT the case that:
        1. All 4 are the same.
        2. Three are the same, one is different.
        3. Two pairs are the same (e.g., c0==c1, c2==c3, c0!=c2).
        4. One pair is the same, others distinct? No, that's 3 distinct.
        
        So Violation occurs if:
        - All 4 equal: c0==c1==c2==c3
        - 3 equal: (c0==c1==c2 != c3) OR (c0==c1==c3 != c2) OR (c0==c2==c3 != c1) OR (c1==c2==c3 != c0)
        - 2 pairs: (c0==c1 != c2==c3) OR (c0==c2 != c1==c3) OR (c0==c3 != c1==c2)
        
        Let's implement this logic in NumPy.
        """
        c0 = colors[prog_array[:, 0]]
        c1 = colors[prog_array[:, 1]]
        c2 = colors[prog_array[:, 2]]
        c3 = colors[prog_array[:, 3]]
        
        # All 4 same
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # 3 same
        three_same = (
            ((c0 == c1) & (c1 == c2) & (c2 != c3)) |
            ((c0 == c1) & (c1 == c3) & (c2 != c3)) |
            ((c0 == c2) & (c2 == c3) & (c0 != c1)) |
            ((c1 == c2) & (c2 == c3) & (c1 != c0))
        )
        
        # 2 pairs
        two_pairs = (
            ((c0 == c1) & (c2 == c3) & (c0 != c2)) |
            ((c0 == c2) & (c1 == c3) & (c0 != c1)) |
            ((c0 == c3) & (c1 == c2) & (c0 != c1))
        )
        
        violations = all_same | three_same | two_pairs
        return np.sum(violations)

    def solve_sa(k, time_limit):
        """Simulated Annealing to find valid k-coloring."""
        local_deadline = start_time + time_limit
        
        best_colors = None
        best_violations = float('inf')
        
        # Try multiple restarts with different initializations
        for restart in range(5):
            if time.time() > local_deadline:
                break
            
            # Initialization: Modular pattern
            # Using modulo k often works well for APs
            if restart == 0:
                colors = np.array([i % k for i in range(n)], dtype=np.int32)
            elif restart == 1:
                colors = np.array([i % (k+1) % k for i in range(n)], dtype=np.int32)
            else:
                # Random
                colors = np.array([random.randint(0, k-1) for _ in range(n)], dtype=np.int32)
            
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                return colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # SA Parameters
            T = 10.0
            cooling = 0.995
            min_T = 0.01
            
            while time.time() < local_deadline and T > min_T:
                # Pick a random position
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                
                # Pick a new color
                if k > 1:
                    new_color = random.randint(0, k-1)
                    while new_color == old_color:
                        new_color = random.randint(0, k-1)
                else:
                    continue
                
                colors[pos] = new_color
                
                # Calculate delta in violations
                # We need to re-evaluate only progressions involving pos
                # But count_violations is vectorized and fast enough for n=128?
                # num_progs is ~1000-2000. Numpy ops on 2000 elements are very fast (<1ms).
                # So we can just call count_violations every step.
                
                new_violations = count_violations(colors)
                delta = new_violations - current_violations
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_violations = new_violations
                    if current_violations == 0:
                        return colors
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors.copy()
                else:
                    colors[pos] = old_color
                
                T *= cooling
                
        return best_colors if best_violations == 0 else None

    # Try k=3 first with SA
    result = solve_sa(3, 45.0)
    if result is not None:
        # Convert numpy array to list of ints
        return [int(c) for c in result]
        
    if time.time() < deadline - 5.0:
        # Try k=4
        result = solve_sa(4, 10.0)
        if result is not None:
            return [int(c) for c in result]
            
    # Fallback: Return a simple pattern that might be close or just valid if k is large
    # If k=3 fails, try a deterministic construction if we have time
    # For now, return best effort from k=4 or a dummy
    if time.time() < deadline - 1.0:
         # Greedy fallback
         colors = [0] * n
         # This is just a placeholder to avoid crash, SA should have found something
         # A valid 3-coloring for n=128 likely exists.
         # Let's try one more quick SA run with k=3 if we didn't find one
         pass
         
    # Final fallback
    return list(range(n))
