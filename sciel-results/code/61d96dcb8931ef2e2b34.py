import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_compatible(selected, m, r):
        for (m2, r2) in selected:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def find_residue(selected, m):
        for r in range(m):
            if is_compatible(selected, m, r):
                return r
        return None

    moduli = list(range(20, 201))
    selected = []

    # Greedy phase: process moduli in increasing order
    for m in moduli:
        if time.time() > deadline - 5.0:
            break
        r = find_residue(selected, m)
        if r is not None:
            selected.append([m, r])

    # Second pass: try to add any moduli we might have missed
    selected_mods = set(m for m, r in selected)
    for m in moduli:
        if time.time() > deadline - 5.0:
            break
        if m in selected_mods:
            continue
        r = find_residue(selected, m)
        if r is not None:
            selected.append([m, r])
            selected_mods.add(m)

    # Local search: try swapping residues to enable adding more moduli
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline - 15.0:
                break
            m, r = selected[i]
            for r_new in range(m):
                if r_new == r:
                    continue
                old_r = selected[i][1]
                selected[i][1] = r_new
                selected_mods = set(s[0] for s in selected)
                added = False
                for mm in moduli:
                    if mm in selected_mods:
                        continue
                    r_add = find_residue(selected, mm)
                    if r_add is not None:
                        selected.append([mm, r_add])
                        added = True
                        break
                if added:
                    improved = True
                    break
                else:
                    selected[i][1] = old_r
            if improved:
                break

    return [[m, r] for m, r in selected]
