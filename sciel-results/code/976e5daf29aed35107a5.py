"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 115.0  # safety margin before 120s kill
    
    # All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Greedy construction: add moduli in increasing order
    covered = np.zeros(base, dtype=np.uint8)
    classes = []  # list of [m, r]
    
    for m in divisors:
        if time.time() > deadline - 2.0:
            break
        # Find residue that covers most uncovered residues
        # covered[r::m] gives the coverage status of residues congruent to r mod m
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = np.count_nonzero(covered[r::m] == 0)
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    # Local search polish: try to improve by changing residues of existing moduli
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 3.0:
                break
            m, r = classes[i]
            # Temporarily remove this class's coverage
            idx = np.arange(r, base, m)
            old_vals = covered[idx].copy()
            covered[idx] = 0
            
            # Current uncovered count without this class
            current_uncovered = np.count_nonzero(covered)
            best_r = r
            best_unc = current_uncovered  # if we keep r, uncovered stays same as current (since we'll re-add r)
            
            # Actually, if we keep r, uncovered = current_uncovered (since we removed it, re-adding gives same as before)
            # If we switch to new_r, uncovered = current_uncovered - (newly covered by new_r)
            # where newly covered = count of positions in new_r::m that are 0 in current covered
            for new_r in range(m):
                if new_r == r:
                    continue
                new_idx = np.arange(new_r, base, m)
                newly = np.count_nonzero(covered[new_idx] == 0)
                unc_after = current_uncovered - newly
                if unc_after < best_unc:
                    best_unc = unc_after
                    best_r = new_r
            
            # Restore original coverage
            covered[idx] = old_vals
            
            if best_r != r:
                classes[i][1] = best_r
                covered[best_r::m] = 1
                improved = True
    
    return classes
# EVOLVE-BLOCK-END
