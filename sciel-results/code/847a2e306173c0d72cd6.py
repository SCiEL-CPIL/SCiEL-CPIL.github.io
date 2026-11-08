"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 120s limit

    def generate_weight_k(k):
        """Generate all integer vectors with exactly k nonzero entries, each ±1."""
        vectors = []
        for coords in itertools.combinations(range(D), k):
            for mask in range(1 << k):
                v = [0] * D
                for i, c in enumerate(coords):
                    v[c] = 1 if (mask >> i) & 1 else -1
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    # Baseline: all weight-2 vectors (220 of them, all mutually compatible)
    w2 = generate_weight_k(2)
    best_result = w2
    best_count = len(w2)  # 220

    # Try weight-3 vectors
    w3 = generate_weight_k(3)
    n3 = len(w3)  # 1320
    
    # For weight-3: norm²=3, need dist²≥3 => 6-2*dot≥3 => dot≤1.5 => dot≤1
    # Conflict if dot ≥ 2
    dots3 = w3 @ w3.T
    conflict3 = dots3 >= 2
    np.fill_diagonal(conflict3, False)
    
    # Build adjacency as list of sets for efficient MIS
    adj3 = [set() for _ in range(n3)]
    conflict_idx3 = np.argwhere(conflict3)
    for i, j in conflict_idx3:
        i, j = int(i), int(j)
        adj3[i].add(j)
        adj3[j].add(i)
    
    # Multiple greedy strategies on weight-3
    np.random.seed(42)
    
    # Strategy A: Minimum degree greedy (deterministic)
    remaining = set(range(n3))
    deg = [len(adj3[i]) for i in range(n3)]
    selected_a = []
    while remaining:
        min_d = min(deg[i] for i in remaining)
        cands = [i for i in remaining if deg[i] == min_d]
        v = cands[0]
        selected_a.append(v)
        remaining.discard(v)
        for u in list(adj3[v]):
            if u in remaining:
                remaining.discard(u)
                for w in adj3[u]:
                    if w in remaining:
                        deg[w] -= 1
    
    if len(selected_a) > best_count:
        best_count = len(selected_a)
        best_result = w3[selected_a]
    
    # Strategy B: Randomized minimum degree (multiple restarts)
    best_sel_b = []
    for r in range(50):
        if time.time() > deadline - 30:
            break
        remaining = set(range(n3))
        deg = [len(adj3[i]) for i in range(n3)]
        selected_b = []
        while remaining:
            min_d = min(deg[i] for i in remaining)
            cands = [i for i in remaining if deg[i] == min_d]
            v = np.random.choice(cands)
            selected_b.append(v)
            remaining.discard(v)
            for u in list(adj3[v]):
                if u in remaining:
                    remaining.discard(u)
                    for w in adj3[u]:
                        if w in remaining:
                            deg[w] -= 1
        if len(selected_b) > len(best_sel_b):
            best_sel_b = selected_b[:]
    
    if len(best_sel_b) > best_count:
        best_count = len(best_sel_b)
        best_result = w3[best_sel_b]
    
    # Strategy C: Random order greedy
    best_sel_c = []
    for r in range(100):
        if time.time() > deadline - 25:
            break
        perm = np.random.permutation(n3)
        in_set = np.zeros(n3, dtype=bool)
        selected_c = []
        for idx in perm:
            if in_set[idx]:
                continue
            # Check if conflicts with any already selected
            if not any(in_set[nb] for nb in adj3[int(idx)]):
                in_set[idx] = True
                selected_c.append(idx)
        if len(selected_c) > len(best_sel_c):
            best_sel_c = selected_c[:]
    
    if len(best_sel_c) > best_count:
        best_count = len(best_sel_c)
        best_result = w3[best_sel_c]
    
    # Local search improvement on best weight-3 solution
    if time.time() < deadline - 10 and best_count > 220:
        in_set = np.zeros(n3, dtype=bool)
        for idx in np.where(in_set == 0)[0]:
            pass
        # Reconstruct in_set from best_result
        # Actually, let's work with the selected list
        if best_count > 220:
            selected = list(best_sel_b) if len(best_sel_b) == best_count else list(selected_a)
            if len(selected) != best_count:
                selected = list(best_sel_c)
            
            in_set = np.zeros(n3, dtype=bool)
            for idx in selected:
                in_set[idx] = True
            
            # Try to add vertices that don't conflict
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                added = 0
                for v in range(n3):
                    if in_set[v]:
                        continue
                    if not any(in_set[nb] for nb in adj3[v]):
                        in_set[v] = True
                        added += 1
                if added == 0:
                    break
            
            if in_set.sum() > best_count:
                best_count = int(in_set.sum())
                best_result = w3[in_set]
    
    # Try weight-4 vectors if time permits
    if time.time() < deadline - 40:
        w4 = generate_weight_k(4)
        n4 = len(w4)  # 5280
        
        # For weight-4: norm²=4, need dist²≥4 => 8-2*dot≥4 => dot≤2
        # Conflict if dot ≥ 3
        dots4 = w4 @ w4.T
        conflict4 = dots4 >= 3
        np.fill_diagonal(conflict4, False)
        
        # Build adjacency
        adj4 = [set() for _ in range(n4)]
        conflict_idx4 = np.argwhere(conflict4)
        for i, j in conflict_idx4:
            i, j = int(i), int(j)
            adj4[i].add(j)
            adj4[j].add(i)
        
        # Randomized greedy on weight-4
        best_sel_d = []
        for r in range(30):
            if time.time() > deadline - 15:
                break
            perm = np.random.permutation(n4)
            in_set = np.zeros(n4, dtype=bool)
            selected_d = []
            for idx in perm:
                if in_set[idx]:
                    continue
                if not any(in_set[nb] for nb in adj4[int(idx)]):
                    in_set[idx] = True
                    selected_d.append(idx)
            if len(selected_d) > len(best_sel_d):
                best_sel_d = selected_d[:]
        
        if len(best_sel_d) > best_count:
            best_count = len(best_sel_d)
            best_result = w4[best_sel_d]
        
        # Local search on weight-4
        if time.time() < deadline - 5 and best_count > 220:
            in_set = np.zeros(n4, dtype=bool)
            for idx in best_sel_d:
                in_set[idx] = True
            
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                added = 0
                for v in range(n4):
                    if in_set[v]:
                        continue
                    if not any(in_set[nb] for nb in adj4[v]):
                        in_set[v] = True
                        added += 1
                if added == 0:
                    break
            
            if in_set.sum() > best_count:
                best_count = int(in_set.sum())
                best_result = w4[in_set]
    
    return best_result
# EVOLVE-BLOCK-END
