# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total = q * q
    deadline = time.time() + 580
    
    k_const = 3
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    directions.append((0, 1, 0, 0))
    n_dirs = len(directions)
    
    # Precompute affected_by
    affected_by = [[] for _ in range(total)]
    for d in range(n_dirs):
        vxa, vxb, vya, vyb = directions[d]
        for y_idx in range(total):
            ya_a = y_idx % p
            ya_b = (y_idx // p) % p
            yb_a = (y_idx // (p*p)) % p
            yb_b = (y_idx // (p*p*p)) % p
            
            for t_a in range(p):
                for t_b in range(p):
                    if t_a == 0 and t_b == 0:
                        continue
                    tvxa = (t_a * vxa + t_b * vxb * k_const) % p
                    tvxb = (t_a * vxb + t_b * vxa) % p
                    tvya = (t_a * vya + t_b * vyb * k_const) % p
                    tvyb = (t_a * vyb + t_b * vya) % p
                    
                    xa = (ya_a - tvxa) % p
                    xb = (ya_b - tvxb) % p
                    yx_a = (yb_a - tvya) % p
                    yx_b = (yb_b - tvyb) % p
                    
                    x_idx = xa + xb * p + yx_a * p * p + yx_b * p * p * p
                    # x is on line through y in dir d
                    # So adding x affects C[y] in dir d
                    affected_by[x_idx].append((y_idx, d))
                    
    def greedy_with_repair(seed):
        random.seed(seed)
        C = [set(range(n_dirs)) for _ in range(total)]
        S = [False] * total
        
        # Phase 1: Unsafe Greedy
        # Add points if C[x] is not empty.
        active = [True] * total
        while True:
            if time.time() > deadline - 10:
                break
            candidates = [i for i in range(total) if active[i] and C[i]]
            if not candidates:
                break
            x = random.choice(candidates)
            S[x] = True
            active[x] = False
            for (y, d) in affected_by[x]:
                if d in C[y]:
                    C[y].remove(d)
        
        # Phase 2: Repair
        # Find points in S with C[x] empty.
        # Remove them.
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            for x in range(total):
                if S[x] and not C[x]:
                    S[x] = False
                    active[x] = True # Can be added again?
                    # Removing x from S restores C[y] for neighbors.
                    for (y, d) in affected_by[x]:
                        if d not in C[y]:
                            C[y].add(d)
                    changed = True
                    # Note: We might need to re-add points?
                    # Let's just do one pass of removal.
        
        # Phase 3: Try to add more points?
        # Now that some points are removed, new candidates might appear.
        # But we already used up the "unsafe" potential.
        # Let's just return S.
        
        return S

    best_S = None
    best_size = 0
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 500:
        S = greedy_with_repair(seed)
        size = sum(S)
        if size > best_size:
            best_size = size
            best_S = S
        seed += 1
        
    if best_S is None:
        best_S = [False] * total
        best_size = 0
        
    N_indices = [i for i in range(total) if not best_S[i]]
    points = np.array([[[i % p, (i // p) % p],
                        [(i // (p*p)) % p, (i // (p*p*p)) % p]] for i in N_indices],
                      dtype=np.int64)
    
    return points
# EVOLVE-BLOCK-END
