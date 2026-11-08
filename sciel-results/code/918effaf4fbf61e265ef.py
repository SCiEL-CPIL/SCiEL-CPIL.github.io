import time
from math import log
import random

def score_set(U):
    """Compute the score for a set U. Fast version."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return -1.0
    
    # Compute sumset and diffset
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sc = len(sums)
    dc = len(diffs)
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1.0:
        return -1.0
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 150s
    
    best_set = [0, 1, 4, 9, 16, 25]
    best_score = score_set(best_set)
    
    # Generate many candidate structures
    candidates = []
    
    # Geometric-like sets with various bases
    for b in range(2, 30):
        for m in range(3, 12):
            U = [0]
            val = 1
            for k in range(1, m):
                val *= b
                U.append(val)
            if max(U) < 200000:
                candidates.append(U)
    
    # Squares
    for m in range(3, 20):
        U = [0] + [i*i for i in range(1, m+1)]
        if max(U) < 200000:
            candidates.append(U)
    
    # Triangular numbers
    for m in range(3, 15):
        U = [0] + [i*(i+1)//2 for i in range(1, m+1)]
        if max(U) < 200000:
            candidates.append(U)
    
    # Fibonacci-like
    for m in range(3, 20):
        U = [0, 1]
        a, b = 1, 1
        for _ in range(m-2):
            a, b = b, a+b
            U.append(b)
        if max(U) < 200000:
            candidates.append(U)
    
    # Mixed: small dense + sparse tail
    for small_n in range(3, 8):
        for tail_base in [3, 4, 5, 7]:
            U = list(range(small_n))
            val = small_n
            for _ in range(3, 8):
                val *= tail_base
                U.append(val)
            if max(U) < 200000:
                candidates.append(sorted(set(U)))
    
    # Evaluate all candidates
    for U in candidates:
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Local search from best
    def local_search(U, deadline):
        U = sorted(set(U))
        cur_score = score_set(U)
        best_local = list(U)
        best_local_score = cur_score
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            improved = False
            n = len(U)
            max_val = max(U)
            
            # Try adding elements
            if n < 25:
                # Try values near existing ones and some random
                trial_vals = set()
                for x in U:
                    for d in range(1, 6):
                        trial_vals.add(x + d)
                        if x - d > 0:
                            trial_vals.add(x - d)
                # Add some random values
                for _ in range(20):
                    trial_vals.add(random.randint(1, max_val + 50))
                
                for v in trial_vals:
                    if v in U or v <= 0:
                        continue
                    new_U = sorted(U + [v])
                    new_s = score_set(new_U)
                    if new_s > cur_score:
                        U = new_U
                        cur_score = new_s
                        improved = True
                        break
            
            # Try removing elements
            if not improved and n > 3:
                for i in range(1, n):
                    new_U = [U[j] for j in range(n) if j != i]
                    new_s = score_set(new_U)
                    if new_s > cur_score:
                        U = new_U
                        cur_score = new_s
                        improved = True
                        break
            
            # Try replacing elements
            if not improved and n > 2:
                for i in range(1, n):
                    old = U[i]
                    for delta in range(-5, 6):
                        if delta == 0:
                            continue
                        new_val = old + delta
                        if new_val <= 0:
                            continue
                        if new_val in U:
                            continue
                        new_U = [U[j] if j != i else new_val for j in range(n)]
                        new_U = sorted(set(new_U))
                        if len(new_U) != n:
                            continue
                        new_s = score_set(new_U)
                        if new_s > cur_score:
                            U = new_U
                            cur_score = new_s
                            improved = True
                            break
                    if improved:
                        break
            
            if improved:
                no_improve = 0
                if cur_score > best_local_score:
                    best_local_score = cur_score
                    best_local = list(U)
            else:
                no_improve += 1
        
        return best_local_score, best_local
    
    # Run local search from multiple seeds
    seeds = [best_set]
    # Add some diverse seeds
    for U in candidates[:5]:
        if U != best_set:
            seeds.append(U)
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        ls_score, ls_set = local_search(list(seed), deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Final fine-tuning
    if time.time() < deadline - 2:
        U = list(best_set)
        cur_score = best_score
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            improved = False
            n = len(U)
            for i in range(1, n):
                old = U[i]
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    nv = old + delta
                    if nv <= 0 or nv in U:
                        continue
                    new_U = [U[j] if j != i else nv for j in range(n)]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    ns = score_set(new_U)
                    if ns > cur_score:
                        U = new_U
                        cur_score = ns
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if cur_score > best_score:
            best_score = cur_score
            best_set = U
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
