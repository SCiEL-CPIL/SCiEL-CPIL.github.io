def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    target_n = 34
    # We will try to find a valid coloring for target_n.
    # If we fail, we decrease n.
    
    # Precomputation for a given n
    def precompute(n):
        # Map triple (a,b,c) to index
        # Index in lex order
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # For each edge (triple), find which K4s and K5s it is part of
        # K4s: combinations of 4 vertices
        k4s = list(combinations(range(n), 4))
        # K5s: combinations of 5 vertices
        k5s = list(combinations(range(n), 5))
        
        # For each edge, list of k4 indices and k5 indices
        edge_to_k4s = [[] for _ in range(len(triples))]
        edge_to_k5s = [[] for _ in range(len(triples))]
        
        # Iterate through K4s
        for k4_idx, k4 in enumerate(k4s):
            # Get the 4 triples in this K4
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            
            idxs = [triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]]
            for idx in idxs:
                edge_to_k4s[idx].append(k4_idx)
                
        # Iterate through K5s
        for k5_idx, k5 in enumerate(k5s):
            # Get the 10 triples in this K5
            ts = list(combinations(k5, 3))
            for t in ts:
                idx = triple_to_idx[t]
                edge_to_k5s[idx].append(k5_idx)
                
        return triples, edge_to_k4s, edge_to_k5s, len(k4s), len(k5s)

    def solve(n, time_limit):
        triples, edge_to_k4s, edge_to_k5s, num_k4, num_k5 = precompute(n)
        num_edges = len(triples)
        
        # Initialize colors randomly
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Initialize counts
        # red_count_k4[i] = number of red edges in k4[i]
        # blue_count_k5[i] = number of blue edges in k5[i]
        red_count_k4 = [0] * num_k4
        blue_count_k5 = [0] * num_k5
        
        for i, c in enumerate(colors):
            if c == 0: # Red
                for k4_idx in edge_to_k4s[i]:
                    red_count_k4[k4_idx] += 1
            else: # Blue
                for k5_idx in edge_to_k5s[i]:
                    blue_count_k5[k5_idx] += 1
        
        # Calculate initial violations
        violations = sum(1 for c in red_count_k4 if c == 4) + sum(1 for c in blue_count_k5 if c == 10)
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        best_violations = violations
        best_colors = colors.copy()
        
        start = time.time()
        steps = 0
        while time.time() - start < time_limit:
            # Pick random edge
            i = random.randint(0, num_edges - 1)
            old_c = colors[i]
            new_c = 1 - old_c
            
            # Calculate delta in violations
            delta = 0
            
            # Update K4s
            if old_c == 0: # Was Red, becomes Blue
                # Red count decreases by 1 for each K4
                for k4_idx in edge_to_k4s[i]:
                    old_rc = red_count_k4[k4_idx]
                    new_rc = old_rc - 1
                    # Violation if old_rc == 4 (was viol, now not) -> -1
                    # Violation if new_rc == 4 (was not, now viol) -> +1
                    if old_rc == 4:
                        delta -= 1
                    if new_rc == 4:
                        delta += 1
                    red_count_k4[k4_idx] = new_rc
            else: # Was Blue, becomes Red
                for k4_idx in edge_to_k4s[i]:
                    old_rc = red_count_k4[k4_idx]
                    new_rc = old_rc + 1
                    if old_rc == 4:
                        delta -= 1
                    if new_rc == 4:
                        delta += 1
                    red_count_k4[k4_idx] = new_rc
            
            # Update K5s
            if old_c == 1: # Was Blue, becomes Red
                # Blue count decreases by 1
                for k5_idx in edge_to_k5s[i]:
                    old_bc = blue_count_k5[k5_idx]
                    new_bc = old_bc - 1
                    if old_bc == 10:
                        delta -= 1
                    if new_bc == 10:
                        delta += 1
                    blue_count_k5[k5_idx] = new_bc
            else: # Was Red, becomes Blue
                for k5_idx in edge_to_k5s[i]:
                    old_bc = blue_count_k5[k5_idx]
                    new_bc = old_bc + 1
                    if old_bc == 10:
                        delta -= 1
                    if new_bc == 10:
                        delta += 1
                    blue_count_k5[k5_idx] = new_bc
            
            violations += delta
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
                if best_violations == 0:
                    return best_colors
            
            # Acceptance
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                colors[i] = new_c
            else:
                # Revert
                if new_c == 0: # Became Red
                    for k4_idx in edge_to_k4s[i]:
                        red_count_k4[k4_idx] -= 1
                else:
                    for k4_idx in edge_to_k4s[i]:
                        red_count_k4[k4_idx] += 1
                        
                if new_c == 1: # Became Blue
                    for k5_idx in edge_to_k5s[i]:
                        blue_count_k5[k5_idx] -= 1
                else:
                    for k5_idx in edge_to_k5s[i]:
                        blue_count_k5[k5_idx] += 1
                
                violations -= delta
                
            T *= cooling
            if T < T_min:
                T = T_min
            steps += 1
            if steps % 10000 == 0 and time.time() - start > time_limit:
                break
                
        return best_colors if best_violations == 0 else None

    # Try to solve for n=34, then 33, etc.
    # Allocate time wisely.
    current_time = time.time()
    deadline = current_time + 110 # Leave 10s margin
    
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        # Allocate remaining time, capped at 30s per n to allow fallback
        remaining = deadline - time.time()
        t_alloc = min(30, remaining)
        if t_alloc < 1:
            break
            
        colors = solve(n, t_alloc)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback to seed if all fail
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
