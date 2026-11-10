import time
import math
import random

MAXM = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _gcd_table():
    g = [[0] * (MAXM + 1) for _ in range(MAXM + 1)]
    for a in range(2, MAXM + 1):
        for b in range(2, MAXM + 1):
            g[a][b] = math.gcd(a, b)
    return g

def _is_disjoint(r1, m1, r2, m2, gt):
    return (r1 - r2) % gt[m1][m2] != 0

def _can_add(r, m, selected, gt):
    for (r2, m2) in selected:
        if not _is_disjoint(r, m, r2, m2, gt):
            return False
    return True

def _solve_randomized(gt, time_limit, seed):
    """
    Attempt to construct a large set using randomized greedy.
    """
    rnd = random.Random(seed)
    mods = list(range(2, MAXM + 1))
    rnd.shuffle(mods)
    
    selected = []
    t0 = time.time()
    
    for m in mods:
        if time.time() - t0 > time_limit:
            break
        
        # Try residues in random order
        residues = list(range(m))
        rnd.shuffle(residues)
        
        added = False
        for r in residues:
            if _can_add(r, m, selected, gt):
                selected.append((r, m))
                added = True
                break
        
        # If we couldn't add this modulus, we just skip it in this trial.
        # A more advanced approach would backtrack, but skipping + restart is often effective.
        
    return selected

def construct():
    start = time.time()
    deadline = start + DEADLINE
    gt = _gcd_table()
    
    best = []
    best_size = 0
    
    # Initial seed: powers of 2
    seed_sol = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed_sol
    best_size = len(seed_sol)
    
    # Try many random restarts
    seed_val = 0
    while time.time() < deadline:
        # Each trial should be fast. Limit time per trial to avoid one long hang.
        # 127 moduli * 128 residues * 127 checks ~ 2M ops max, which is < 0.1s in Python.
        # So we can run many trials.
        trial_time = min(1.0, (deadline - time.time()) * 0.1)
        
        sol = _solve_randomized(gt, trial_time, seed_val)
        if len(sol) > best_size:
            best_size = len(sol)
            best = [list(x) for x in sol]
            
            # If we found all 127, we can stop
            if best_size == 127:
                break
                
        seed_val += 1
        
    # Final polish: try to add any missing moduli to the best solution found
    if best and time.time() < deadline:
        # Convert to tuples for internal processing
        current = [tuple(x) for x in best]
        used_mods = set(m for _, m in current)
        missing = [m for m in range(2, MAXM + 1) if m not in used_mods]
        
        # Try to insert missing moduli
        for m in missing:
            if time.time() > deadline:
                break
            # Try random residues
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if _can_add(r, m, current, gt):
                    current.append((r, m))
                    used_mods.add(m)
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best = [list(x) for x in current]

    if not best:
        best = [[1, 2]]
    return best

if __name__ == "__main__":
    sol = construct()
    print("size:", len(sol))
    # Verify
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            r1, m1 = sol[i]
            r2, m2 = sol[j]
            g = math.gcd(m1, m2)
            assert (r1 - r2) % g != 0, (i, j, r1, m1, r2, m2)
    mods = [m for (_, m) in sol]
    assert len(set(mods)) == len(mods)
    for r, m in sol:
        assert 0 <= r < m and 2 <= m <= 128
    print("OK")
