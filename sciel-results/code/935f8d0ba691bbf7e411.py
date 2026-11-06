"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n = 16
    num_quads = list(itertools.combinations(range(n), 4))
    
    def count_colors(coloring):
        return max(coloring) + 1
    
    def build_ec(coloring, n=16):
        """Build edge color lookup as a 2D array."""
        ec = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                ec[i, j] = coloring[idx]
                ec[j, i] = coloring[idx]
                idx += 1
        return ec
    
    def is_valid_fast(coloring, n=16):
        """Fast validation using a 2D array and precomputed K4 edge indices."""
        ec = build_ec(coloring, n)
        for a, b, c, d in num_quads:
            colors = [ec[a,b], ec[a,c], ec[a,d], ec[b,c], ec[b,d], ec[c,d]]
            if len(set(colors)) < 5:
                return False
        return True
    
    def count_violations(coloring, n=16):
        """Count how many K4s have fewer than 5 distinct colors."""
        ec = build_ec(coloring, n)
        violations = 0
        for a, b, c, d in num_quads:
            colors = [ec[a,b], ec[a,c], ec[a,d], ec[b,c], ec[b,d], ec[c,d]]
            if len(set(colors)) < 5:
                violations += 1
        return violations
    
    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    def local_search(coloring, max_iter=5000, n=16):
        """Hill-climbing: try changing one edge color at a time to reduce violations."""
        coloring = list(coloring)
        ec = build_ec(coloring, n)
        current_violations = count_violations(coloring, n)
        if current_violations == 0:
            return coloring, 0
        
        best_violations = current_violations
        best_coloring = coloring[:]
        no_improve = 0
        max_no_improve = 50
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try each edge
            for idx in range(n*(n-1)//2):
                if time.time() > deadline:
                    break
                # Find i, j for this edge index
                # Edge idx corresponds to pair (i,j) where idx = i*n - i*(i+1)//2 + (j-i-1)
                # Actually let's just iterate properly
                pass
            
            # More efficient: iterate over all edges directly
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    old_color = coloring[i*(n-1) - i*(i+1)//2 + (j-i-1)]
                    for new_color in range(max(coloring)+1):
                        if new_color == old_color:
                            continue
                        # Update
                        edge_idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
                        coloring[edge_idx] = new_color
                        v = count_violations(coloring, n)
                        if v < best_violations:
                            best_violations = v
                            best_coloring = coloring[:]
                            improved = True
                            no_improve = 0
                        else:
                            coloring[edge_idx] = old_color
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
            else:
                coloring = best_coloring[:]
        
        return best_coloring, best_violations
    
    # Try various constructions
    candidates = []
    
    # Linear forms: (a*i + b*j) mod m
    for mod in range(13, 20):
        for a in range(1, 9):
            for b in range(1, 9):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
                candidates.append(cand)
    
    # Non-linear: (a*i*i + b*i*j + c*j*j) mod m
    for mod in range(13, 19):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    if a == b == c:
                        continue
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: (a*i*i + b*i*j + c*j*j) % mod
                    )
                    candidates.append(cand)
    
    # XOR-based: (i XOR j) * k mod m
    for mod in range(13, 19):
        for k in range(1, 6):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i ^ j) * k) % mod
            )
            candidates.append(cand)
    
    # Product-based: (i * j) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j) % mod)
        candidates.append(cand)
    
    # Mixed: (a*i + b*j + c*i*j) mod m
    for mod in range(13, 18):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 4):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: (a*i + b*j + c*i*j) % mod
                    )
                    candidates.append(cand)
    
    # Difference-based: (i - j)^2 mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i - j)**2) % mod)
        candidates.append(cand)
    
    # Sum of squares: (i^2 + j^2) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i + j*j) % mod)
        candidates.append(cand)
    
    # NEW: Polynomial in (i-j): (a*(i-j) + b*(i-j)^2 + c*(i-j)^3) mod m
    for mod in range(13, 19):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: 
                        (a*(i-j) + b*(i-j)**2 + c*(i-j)**3) % mod
                    )
                    candidates.append(cand)
    
    # NEW: (a*i + b*j + c*i*j + d*i*i + e*j*j) mod m
    for mod in range(13, 17):
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 3):
                    for d in range(1, 3):
                        for e in range(1, 3):
                            cand = make_coloring_formula(
                                lambda i, j, a=a, b=b, c=c, d=d, e=e, mod=mod: 
                                (a*i + b*j + c*i*j + d*i*i + e*j*j) % mod
                            )
                            candidates.append(cand)
    
    # NEW: Bitwise operations
    for mod in range(13, 19):
        for k in range(1, 5):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i & j) * k + (i | j)) % mod
            )
            candidates.append(cand)
        for k in range(1, 5):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i ^ j) + (i & j) * k) % mod
            )
            candidates.append(cand)
    
    # NEW: (i + j)^2 mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i + j)**2 % mod)
        candidates.append(cand)
    
    # NEW: (i * j + i + j) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j + i + j) % mod)
        candidates.append(cand)
    
    # NEW: (i * (j - i)) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * (j - i)) % mod)
        candidates.append(cand)
    
    # NEW: (j * (j - i)) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (j * (j - i)) % mod)
        candidates.append(cand)
    
    # NEW: (i * j * (i + j)) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j * (i + j)) % mod)
        candidates.append(cand)
    
    # NEW: (i^3 + j^3) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i**3 + j**3) % mod)
        candidates.append(cand)
    
    # NEW: (i*j + i^2 + j^2) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j + i*i + j*j) % mod)
        candidates.append(cand)
    
    # Find the best valid candidate
    best = None
    best_num_colors = float('inf')
    
    # Sort candidates by number of colors used (ascending) to find minimum faster
    candidates.sort(key=lambda c: count_colors(c))
    
    # First pass: check candidates with <= 15 colors
    for cand in candidates:
        if time.time() > deadline:
            break
        nc = count_colors(cand)
        if nc >= best_num_colors:
            break  # No need to check candidates with more colors
        if nc > 15:
            break  # We want to find <= 15
        if is_valid_fast(cand):
            best_num_colors = nc
            best = cand
    
    # If we found a valid one with <= 15 colors, try to improve with local search
    if best is not None and best_num_colors <= 15:
        # Try local search to see if we can reduce further
        improved, violations = local_search(best, max_iter=2000, n=16)
        if violations == 0:
            inc = count_colors(improved)
            if inc < best_num_colors:
                best = improved
                best_num_colors = inc
    
    # If no valid candidate with <= 15 found, try local search on near-miss candidates
    if best is None:
        # Find candidates with fewest violations and try to repair them
        best_violation_count = float('inf')
        best_near_miss = None
        for cand in candidates:
            if time.time() > deadline:
                break
            nc = count_colors(cand)
            if nc > 16:
                break
            v = count_violations(cand, n)
            if v < best_violation_count:
                best_violation_count = v
                best_near_miss = cand
            if best_violation_count == 0:
                break
        
        if best_near_miss is not None:
            repaired, violations = local_search(best_near_miss, max_iter=5000, n=16)
            if violations == 0:
                best = repaired
                best_num_colors = count_colors(repaired)
    
    if best is not None:
        return best
    
    # Fallback: return a known valid coloring
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
