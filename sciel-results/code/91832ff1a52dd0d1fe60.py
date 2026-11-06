def construct():
    import time
    from math import gcd
    from itertools import combinations
    import random
    
    deadline = time.time() + 80  # safety margin
    
    N = 128
    # Precompute GCD matrix
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    # Precompute bad triples: for each element x, which pairs (a,b) would make {a,b,x} bad?
    # A triple (a,b,x) is bad iff g[a][b] == g[a][x] == g[b][x]
    # For local search, we need: can we add x to set S?
    # x can be added iff for all pairs (a,b) in S, NOT (g[a][b] == g[a][x] == g[b][x])
    
    # Greedy construction
    selected = []
    selected_set = set()
    for x in range(1, N+1):
        ok = True
        for a in selected:
            for b in selected:
                if a == b:
                    continue
                if g[a][b] == g[a][x] == g[b][x]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
            selected_set.add(x)
    
    best = list(selected)
    best_size = len(selected)
    
    # Local search with swaps
    def can_add(x, sel_list, sel_set):
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i+1, len(sel_list)):
                b = sel_list[j]
                if g[a][b] == g[a][x] == g[b][x]:
                    return False
        return True
    
    def is_valid(sel_list):
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i+1, len(sel_list)):
                b = sel_list[j]
                for k in range(j+1, len(sel_list)):
                    c = sel_list[k]
                    if g[a][b] == g[a][c] == g[b][c]:
                        return False
        return True
    
    # Try to add elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N+1):
            if x in selected_set:
                continue
            if can_add(x, selected, selected_set):
                selected.append(x)
                selected_set.add(x)
                improved = True
                if len(selected) > best_size:
                    best = list(selected)
                    best_size = len(selected)
    
    # Swap-based local search
    for x in range(1, N+1):
        if x in selected_set:
            continue
        # Try removing one element and adding x
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            removed = selected[i]
            # Try removing selected[i] and adding x
            new_sel = [s for s in selected if s != removed]
            if can_add(x, new_sel, set(new_sel)):
                # Check full validity
                if is_valid(new_sel + [x]):
                    selected = new_sel + [x]
                    selected_set = set(selected)
                    if len(selected) > best_size:
                        best = list(selected)
                        best_size = len(selected)
                    break
        if time.time() > deadline:
            break
    
    # More aggressive: try multiple random restarts of local search
    if time.time() < deadline - 10:
        random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            # Random greedy with different orderings
            order = list(range(1, N+1))
            random.shuffle(order)
            sel = []
            sel_set = set()
            for x in order:
                if can_add(x, sel, sel_set):
                    sel.append(x)
                    sel_set.add(x)
            # Local search on this
            improved2 = True
            while improved2 and time.time() < deadline - 2:
                improved2 = False
                for x in range(1, N+1):
                    if x in sel_set:
                        continue
                    if can_add(x, sel, sel_set):
                        sel.append(x)
                        sel_set.add(x)
                        improved2 = True
            if len(sel) > best_size:
                best = list(sel)
                best_size = len(sel)
    
    return best if best else [1]
