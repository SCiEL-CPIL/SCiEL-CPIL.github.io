"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

MAX_MOD = 128
TIME_LIMIT = 85.0

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    
    moduli = list(range(2, MAX_MOD + 1))
    n = len(moduli)
    
    # Precompute gcd
    gcd_mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Greedy: pick moduli with many non-coprime neighbors
    neighbor_count = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if gcd_mat[i][j] > 1:
                neighbor_count[i] += 1
                neighbor_count[j] += 1
    
    order = sorted(range(n), key=lambda i: -neighbor_count[i])
    
    selected = []
    for idx in order:
        if all(gcd_mat[idx][sel] > 1 for sel in selected):
            selected.append(idx)
    
    # Assign residues via greedy
    sel_moduli = [moduli[i] for i in selected]
    k = len(sel_moduli)
    residues = [0]*k
    
    # For each pair, we need r_i % g != r_j % g where g = gcd(m_i, m_j)
    # Greedy: assign in order of most constraints
    degree = [0]*k
    for i in range(k):
        for j in range(i+1, k):
            if gcd_mat[selected[i]][selected[j]] > 1:
                degree[i] += 1
                degree[j] += 1
    
    var_order = sorted(range(k), key=lambda i: -degree[i])
    assigned = [False]*k
    residues = [0]*k
    
    for vi in var_order:
        mi = sel_moduli[vi]
        # Try all residues, pick one that doesn't conflict with already assigned
        for r in range(mi):
            ok = True
            for vj in range(k):
                if vj == vi or not assigned[vj]:
                    continue
                mj = sel_moduli[vj]
                rj = residues[vj]
                g = gcd_mat[selected[vi]][selected[vj]]
                if g > 1 and r % g == rj % g:
                    ok = False
                    break
            if ok:
                residues[vi] = r
                assigned[vi] = True
                break
        else:
            # Couldn't assign, remove this modulus
            selected.remove(selected[vi])
            sel_moduli = [moduli[i] for i in selected]
            k = len(sel_moduli)
            break
    
    if time.time() - start_time > TIME_LIMIT - 5:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    result = [[residues[i], sel_moduli[i]] for i in range(len(sel_moduli))]
    
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            g = _gcd(m1, m2)
            if g > 1 and r1 % g == r2 % g:
                # Invalid, fall back
                return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return result
# EVOLVE-BLOCK-END
