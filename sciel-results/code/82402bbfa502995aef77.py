import numpy as np
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # Precompute conflict matrix for speed
    # conflict[i][j] = True if (i-j) mod m or (j-i) mod m is in forbidden
    # i.e., if (i-j) mod m in forbidden
    conflict = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            d = (i - j) % m
            if d in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True
    
    best_size = 0
    best_set = []
    
    # Multiple random restarts with greedy + local improvement
    random.seed(42)
    num_restarts = 500
    
    for restart in range(num_restarts):
        # Random permutation
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy: add vertex if no conflict with current set
        selected = set()
        for v in order:
            ok = True
            for u in selected:
                if conflict[v][u]:
                    ok = False
                    break
            if ok:
                selected.add(v)
        
        # Local improvement: try to add any vertex not in selected
        improved = True
        while improved:
            improved = False
            for v in range(m):
                if v in selected:
                    continue
                ok = True
                for u in selected:
                    if conflict[v][u]:
                        ok = False
                        break
                if ok:
                    selected.add(v)
                    improved = True
        
        # Try swaps: remove one, add two? Or just try adding after removing one
        # Simple local search: try removing each vertex and see if we can add more
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                trial = selected - {v}
                # Try to add any vertex not in trial
                added = 0
                for w in range(m):
                    if w in trial:
                        continue
                    ok = True
                    for u in trial:
                        if conflict[w][u]:
                            ok = False
                            break
                    if ok:
                        trial.add(w)
                        added += 1
                if len(trial) > len(selected):
                    selected = trial
                    improved = True
                    break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    return np.array(sorted(best_set), dtype=np.int64)
