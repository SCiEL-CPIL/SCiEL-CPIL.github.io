"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    
    # Precompute conflict matrix
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    evens = [i for i in range(2, N + 1) if i % 2 == 0]
    
    # All odds are mutually compatible (odd*odd=odd, odd+odd=even, even∤odd)
    base_set = set(odds)
    
    # Find evens compatible with all odds
    compatible_evens = []
    for e in evens:
        if all(not conflict[e][o] for o in odds):
            compatible_evens.append(e)
    
    # Build conflict graph among compatible_evens
    # Find maximum independent set using greedy + local search
    
    best_size = len(base_set)
    best_evens = set()
    
    # Multiple greedy strategies
    strategies = []
    
    # Strategy 1: Minimum degree greedy
    def greedy_min_degree(cands):
        selected = set()
        remaining = list(cands)
        while remaining:
            if time.time() > DEADLINE - 15.0:
                break
            best_e = min(remaining, key=lambda e: sum(1 for x in remaining if x != e and conflict[e][x]))
            selected.add(best_e)
            remaining = [x for x in remaining if x != best_e and not conflict[best_e][x]]
        return selected
    
    # Strategy 2: Maximum degree greedy (remove high-degree nodes first)
    def greedy_max_degree(cands):
        selected = set()
        remaining = list(cands)
        while remaining:
            if time.time() > DEADLINE - 15.0:
                break
            best_e = max(remaining, key=lambda e: sum(1 for x in remaining if x != e and conflict[e][x]))
            selected.add(best_e)
            remaining = [x for x in remaining if x != best_e and not conflict[best_e][x]]
        return selected
    
    # Strategy 3: Random greedy
    def greedy_random(cands, seed):
        rng = random.Random(seed)
        selected = set()
        remaining = list(cands)
        rng.shuffle(remaining)
        for e in remaining:
            if time.time() > DEADLINE - 15.0:
                break
            if not any(conflict[e][x] for x in selected):
                selected.add(e)
        return selected
    
    # Try all strategies
    sel1 = greedy_min_degree(compatible_evens)
    if len(base_set) + len(sel1) > best_size:
        best_size = len(base_set) + len(sel1)
        best_evens = set(sel1)
    
    sel2 = greedy_max_degree(compatible_evens)
    if len(base_set) + len(sel2) > best_size:
        best_size = len(base_set) + len(sel2)
        best_evens = set(sel2)
    
    for seed in range(50):
        if time.time() > DEADLINE - 10.0:
            break
        sel_r = greedy_random(compatible_evens, seed)
        if len(base_set) + len(sel_r) > best_size:
            best_size = len(base_set) + len(sel_r)
            best_evens = set(sel_r)
    
    # Local search on best solution
    current_evens = set(best_evens)
    
    for iteration in range(1000):
        if time.time() > DEADLINE - 5.0:
            break
        
        improved = False
        in_set = list(current_evens)
        out_set = [e for e in compatible_evens if e not in current_evens]
        
        # Try adding
        for e in out_set:
            if time.time() > DEADLINE - 5.0:
                break
            if not any(conflict[e][x] for x in current_evens):
                current_evens.add(e)
                improved = True
        
        if improved:
            continue
        
        # Try 1-swap
        for a in in_set:
            if time.time() > DEADLINE - 5.0:
                break
            for b in out_set:
                if time.time() > DEADLINE - 5.0:
                    break
                # Check b is compatible with (current_evens - {a})
                if all(not conflict[b][x] for x in current_evens if x != a):
                    # Check if swap improves (b has fewer conflicts than a)
                    a_deg = sum(1 for x in current_evens if x != a and conflict[a][x])
                    b_deg = sum(1 for x in current_evens if x != a and conflict[b][x])
                    if b_deg < a_deg:
                        current_evens.remove(a)
                        current_evens.add(b)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    # Try 2-swap: remove two, add two
    for iteration in range(200):
        if time.time() > DEADLINE - 3.0:
            break
        
        improved = False
        in_set = list(current_evens)
        out_set = [e for e in compatible_evens if e not in current_evens]
        
        for i in range(len(in_set)):
            if time.time() > DEADLINE - 3.0:
                break
            for j in range(i + 1, len(in_set)):
                if time.time() > DEADLINE - 3.0:
                    break
                a, b = in_set[i], in_set[j]
                # Try to find two evens c, d from out_set that are compatible with (current - {a,b})
                # and with each other
                temp_set = current_evens - {a, b}
                candidates = [x for x in out_set if all(not conflict[x][y] for y in temp_set)]
                # Find independent set of size 2 in candidates
                found = False
                for c in candidates:
                    if time.time() > DEADLINE - 3.0:
                        break
                    for d in candidates:
                        if d == c:
                            continue
                        if not conflict[c][d]:
                            # Check both are better than a,b
                            a_deg = sum(1 for x in current_evens if x != a and x != b and conflict[a][x])
                            b_deg = sum(1 for x in current_evens if x != a and x != b and conflict[b][x])
                            c_deg = sum(1 for x in temp_set if conflict[c][x])
                            d_deg = sum(1 for x in temp_set if conflict[d][x])
                            if c_deg + d_deg < a_deg + b_deg:
                                current_evens.remove(a)
                                current_evens.remove(b)
                                current_evens.add(c)
                                current_evens.add(d)
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    result = sorted(base_set | current_evens)
    
    # Final verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                return sorted(base_set)
    
    return result
# EVOLVE-BLOCK-END
