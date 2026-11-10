# EVOLVE-BLOCK-START
import numpy as np
import time
import random

M = 205

def _compute_forbidden():
    forbidden = set()
    for x in range(1, M):
        val = pow(x, 2, M)
        if val != 0:
            forbidden.add(val)
    return forbidden

FORBIDDEN = _compute_forbidden()

# Build conflict bitsets
CONFLICT_BITS = [0] * M
for a in range(M):
    mask = 0
    for b in range(M):
        if a != b and (a - b) % M in FORBIDDEN:
            mask |= (1 << b)
    CONFLICT_BITS[a] = mask

def _can_add(cand, selected_bits):
    return (selected_bits & CONFLICT_BITS[cand]) == 0

def _popcount(x):
    return bin(x).count('1')

def _greedy(order, selected_bits=0):
    sel = selected_bits
    for cand in order:
        if (sel >> cand) & 1:
            continue
        if _can_add(cand, sel):
            sel |= (1 << cand)
    return sel

def _local_search(sel_bits, deadline, max_iter=100000):
    sel = sel_bits
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try pure add
        for cand in range(M):
            if (sel >> cand) & 1:
                continue
            if _can_add(cand, sel):
                sel |= (1 << cand)
                improved = True
                break
        if improved:
            continue
        
        # Try 1-1 swap
        found = False
        for s in range(M):
            if not ((sel >> s) & 1):
                continue
            new_sel = sel & ~(1 << s)
            for cand in range(M):
                if (new_sel >> cand) & 1:
                    continue
                if _can_add(cand, new_sel):
                    sel = new_sel | (1 << cand)
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # Try 1-2 move
        found = False
        for s in range(M):
            if not ((sel >> s) & 1):
                continue
            new_sel = sel & ~(1 << s)
            adds = 0
            for cand in range(M):
                if adds == 2:
                    break
                if (new_sel >> cand) & 1:
                    continue
                if _can_add(cand, new_sel):
                    new_sel |= (1 << cand)
                    adds += 1
            if adds == 2:
                sel = new_sel
                found = True
                break
        if found:
            continue
        
        # Try 2-2 move
        found = False
        sel_list = [i for i in range(M) if (sel >> i) & 1]
        n = len(sel_list)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                new_sel = sel & ~(1 << sel_list[i]) & ~(1 << sel_list[j])
                adds = 0
                for cand in range(M):
                    if adds == 2:
                        break
                    if (new_sel >> cand) & 1:
                        continue
                    if _can_add(cand, new_sel):
                        new_sel |= (1 << cand)
                        adds += 1
                if adds == 2:
                    sel = new_sel
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # Try 2-3 move
        found = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                new_sel = sel & ~(1 << sel_list[i]) & ~(1 << sel_list[j])
                adds = 0
                for cand in range(M):
                    if adds == 3:
                        break
                    if (new_sel >> cand) & 1:
                        continue
                    if _can_add(cand, new_sel):
                        new_sel |= (1 << cand)
                        adds += 1
                if adds == 3:
                    sel = new_sel
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return sel

def furstenberg_sarkozy_205() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0
    best = 0
    best_count = 0
    
    random.seed(42)
    
    # Phase 1: many greedy starts
    for i in range(10000):
        if time.time() > deadline - 200:
            break
        order = list(range(M))
        random.shuffle(order)
        sel = _greedy(order, 0)
        cnt = _popcount(sel)
        if cnt > best_count:
            best = sel
            best_count = cnt
    
    # Phase 2: local search on best
    sel = _local_search(best, deadline - 100, max_iter=500000)
    cnt = _popcount(sel)
    if cnt > best_count:
        best = sel
        best_count = cnt
    
    # Phase 3: local search on random greeds
    for i in range(800):
        if time.time() > deadline - 100:
            break
        order = list(range(M))
        random.shuffle(order)
        sel = _greedy(order, 0)
        sel = _local_search(sel, deadline - 100, max_iter=100000)
        cnt = _popcount(sel)
        if cnt > best_count:
            best = sel
            best_count = cnt
    
    # Phase 4: final polish
    sel = _local_search(best, deadline - 10, max_iter=500000)
    cnt = _popcount(sel)
    if cnt > best_count:
        best = sel
        best_count = cnt
    
    result = [i for i in range(M) if (best >> i) & 1]
    return np.array(sorted(result), dtype=np.int64)

# EVOLVE-BLOCK-END
