import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 128
    
    # Precompute 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0
    # Indices are 0-based in our array.
    progressions = []
    for i in range(n):
        for d in range(1, n):
            j = i + d
            k = i + 2*d
            l = i + 3*d
            if l < n:
                progressions.append((i, j, k, l))
                
    def is_valid(colors):
        """Check if the coloring satisfies the constraint:
        Every nonconstant four-term progression uses at least 3 distinct colors.
        """
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
        
    def count_colors(colors):
        return len(set(colors))
        
    # Strategy 1: Deterministic Number-Theoretic Construction
    # Map each position i to its Hamming weight (number of 1s in binary representation).
    # For n=128 (2^7), positions are 0..127.
    # Hamming weights range from 0 to 7.
    # This creates 8 color classes.
    
    # The hypothesis is that for any 4-term AP (a, a+d, a+2d, a+3d),
    # the Hamming weights of these four numbers will contain at least 3 distinct values.
    
    # Let's test this hypothesis.
    colors_popcount = [bin(i).count('1') for i in range(n)]
    
    if is_valid(colors_popcount):
        # This is a valid solution with at most 8 colors.
        # We can try to see if we can reduce the number of colors used,
        # but 8 is already much better than 128.
        # The task says "Minimize the number of labels actually used".
        # The labels are 0..7. So 8 labels are used.
        # Is it possible to use fewer?
        # Let's check if we can map some popcount values to the same color.
        # But first, let's return this valid solution.
        # We can relabel to 0..k-1 if needed, but 0..7 are already distinct integers.
        return colors_popcount
        
    # If the popcount strategy fails (which might happen if there's a 4-AP with only 2 distinct popcounts),
    # we need a fallback.
    
    # Let's quickly check why it might fail.
    # Example: 0 (0), 1 (1), 2 (1), 3 (2) -> d=1. Weights: 0, 1, 1, 2. Distinct: {0,1,2}. OK.
    # Example: 0 (0), 2 (1), 4 (1), 6 (2) -> d=2. Weights: 0, 1, 1, 2. Distinct: {0,1,2}. OK.
    # Example: 1 (1), 3 (2), 5 (2), 7 (3) -> d=2. Weights: 1, 2, 2, 3. Distinct: {1,2,3}. OK.
    # Example: 0 (0), 4 (1), 8 (1), 12 (2) -> d=4. Weights: 0, 1, 1, 2. Distinct: {0,1,2}. OK.
    # Example: 2 (1), 6 (2), 10 (2), 14 (3) -> d=4. Weights: 1, 2, 2, 3. Distinct: {1,2,3}. OK.
    
    # It seems popcount is a very strong candidate.
    
    # Strategy 2: Fallback to a simple greedy or local search if popcount fails.
    # But given the time limit, we should prioritize the deterministic check.
    
    # If popcount is not valid, we can try a variant:
    # Maybe map based on modulo something?
    # Or use a different polynomial?
    
    # For now, if popcount is invalid, we fall back to a robust local search
    # but with a much tighter time limit to avoid timeouts.
    
    if time.time() > deadline:
        # Absolute fallback: unique colors
        return list(range(n))
        
    # Local Search Fallback
    import random
    random.seed(42)
    
    prog_at = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(progressions):
        prog_at[a].append(idx)
        prog_at[b].append(idx)
        prog_at[c].append(idx)
        prog_at[d].append(idx)
        
    def minimize_colors(k):
        colors = [random.randint(0, k-1) for _ in range(n)]
        
        is_violated = [False] * len(progressions)
        total_violations = 0
        for idx, (a, b, c, d) in enumerate(progressions):
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                is_violated[idx] = True
                total_violations += 1
        
        if total_violations == 0:
            return colors
            
        best_colors = colors[:]
        best_violations = total_violations
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        # Limit iterations to fit in remaining time
        max_iters = 50000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n-1)
            old_c = colors[i]
            
            best_new_c = old_c
            best_delta = 0
            
            affected_progs = prog_at[i]
            current_violations = sum(1 for idx in affected_progs if is_violated[idx])
            
            for c in range(k):
                if c == old_c:
                    continue
                colors[i] = c
                
                new_violations = 0
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    if len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3:
                        new_violations += 1
                        
                delta = new_violations - current_violations
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            colors[i] = old_c
            
            if best_new_c == old_c:
                if best_delta > 0 and T > min_T:
                    prob = pow(2.718281828, -best_delta / T)
                    if random.random() < prob:
                        colors[i] = best_new_c
                        for idx in affected_progs:
                            a, b, c_idx, d = progressions[idx]
                            old_violated = is_violated[idx]
                            new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                            if old_violated != new_violated:
                                if new_violated:
                                    total_violations += 1
                                else:
                                    total_violations -= 1
                                is_violated[idx] = new_violated
                        if total_violations < best_violations:
                            best_violations = total_violations
                            best_colors = colors[:]
                            if best_violations == 0:
                                return best_colors
                        T *= cooling
            else:
                colors[i] = best_new_c
                for idx in affected_progs:
                    a, b, c_idx, d = progressions[idx]
                    old_violated = is_violated[idx]
                    new_violated = len({colors[a], colors[b], colors[c_idx], colors[d]}) < 3
                    if old_violated != new_violated:
                        if new_violated:
                            total_violations += 1
                        else:
                            total_violations -= 1
                        is_violated[idx] = new_violated
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
                T *= cooling
        
        return best_colors if best_violations == 0 else None

    # Try to find a solution with fewer colors than popcount (8)
    # If popcount was valid, we already returned it.
    # If it wasn't, we try local search with k=3, 4, 5, 6, 7, 8.
    
    for k in range(3, 9):
        if time.time() > deadline:
            break
        result = minimize_colors(k)
        if result:
            return result
            
    # If all else fails, return unique colors
    return list(range(n))
