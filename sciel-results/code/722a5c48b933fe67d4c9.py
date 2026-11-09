# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580
    
    # Forbidden differences: nonzero squares mod 205
    forbidden_set = set()
    for x in range(1, m):
        v = (x * x) % m
        if v != 0:
            forbidden_set.add(v)
    
    # Conflict: i and j conflict if (i-j) % m in forbidden_set
    n = m
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = (i - j) % m
            if d in forbidden_set:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy with multiple orderings
    def greedy(order):
        sel = []
        for c in order:
            if all(c not in conflict[s] for s in sel):
                sel.append(c)
        return sel
    
    best_sol = []
    for order in [list(range(n)), list(range(n-1, -1, -1)),
                  sorted(range(n), key=lambda x: len(conflict[x])),
                  sorted(range(n), key=lambda x: -len(conflict[x]))]:
        s = greedy(order)
        if len(s) > len(best_sol):
            best_sol = s
    
    best_size = len(best_sol)
    
    # Branch and bound
    def ub_matching(cands):
        """Upper bound on independence number via maximal matching."""
        cset = set(cands)
        matched = set()
        msz = 0
        for v in cands:
            if v in matched:
                continue
            for u in cands:
                if u > v and u not in matched and u in conflict[v]:
                    matched.add(v)
                    matched.add(u)
                    msz += 1
                    break
        return len(cands) - msz
    
    def bb(cands, sel, sz):
        nonlocal best_size, best_sol
        if time.time() > deadline:
            return
        if not cands:
            if sz > best_size:
                best_size = sz
                best_sol = sel[:]
            return
        if sz + ub_matching(cands) <= best_size:
            return
        # Pick vertex with max degree in remaining
        cset = set(cands)
        v = max(cands, key=lambda x: len(conflict[x] & cset))
        # Include v
        new_cands = [c for c in cands if c != v and c not in conflict[v]]
        bb(new_cands, sel + [v], sz + 1)
        if time.time() > deadline:
            return
        # Exclude v
        bb([c for c in cands if c != v], sel, sz)
    
    sel_set = set(best_sol)
    rem = [i for i in range(n) if i not in sel_set]
    bb(rem, best_sol[:], best_size)
    
    return np.array(best_sol, dtype=np.int64)

# EVOLVE-BLOCK-END
