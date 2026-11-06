"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

MAX_MOD = 128
DEADLINE = 85

def _compatible(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return True
    return (r1 - r2) % g != 0

def construct():
    t0 = time.time()
    
    # Greedy: select moduli and assign residues
    # Key: for modulus m, residue r. Two classes (r1,m1),(r2,m2) disjoint iff
    # gcd(m1,m2) does not divide (r1-r2)
    
    # Strategy: process moduli in order of increasing "conflict"
    # For each modulus, pick a residue that's compatible with all previously chosen
    
    moduli = list(range(2, MAX_MOD + 1))
    
    # Count for each modulus how many others share gcd>1
    conflict = {}
    for m in moduli:
        c = 0
        for m2 in moduli:
            if m != m2 and gcd(m, m2) > 1:
                c += 1
        conflict[m] = c
    
    # Order by increasing conflict (least conflicting first)
    order = sorted(moduli, key=lambda m: conflict[m])
    
    selected = []  # (residue, modulus)
    
    for m in order:
        if time.time() - t0 > DEADLINE - 1:
            break
        # Try all residues, pick first compatible
        for r in range(m):
            ok = True
            for (rj, mj) in selected:
                if not _compatible(r, m, rj, mj):
                    ok = False
                    break
            if ok:
                selected.append((r, m))
                break
    
    if not selected:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    best = len(selected)
    best_result = [[r, m] for r, m in selected]
    
    # Try to improve with random restarts
    if time.time() - t0 < DEADLINE - 5:
        for _ in range(30):
            if time.time() - t0 > DEADLINE - 2:
                break
            rand_order = order[:]
            random.shuffle(rand_order)
            sel = []
            for m in rand_order:
                if time.time() - t0 > DEADLINE - 1:
                    break
                for r in range(m):
                    ok = True
                    for (rj, mj) in sel:
                        if not _compatible(r, m, rj, mj):
                            ok = False
                            break
                    if ok:
                        sel.append((r, m))
                        break
            if len(sel) > best:
                best = len(sel)
                best_result = [[r, m] for r, m in sel]
    
    return best_result
# EVOLVE-BLOCK-END
