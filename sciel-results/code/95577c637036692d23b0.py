import time
from math import log
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    # Fast sumset/diffset computation using numpy
    arr = np.array(U, dtype=np.int64)
    # Pairwise sums (including i=j)
    sums = arr[:, None] + arr[None, :]
    unique_sums = np.unique(sums)
    # Pairwise differences (all ordered pairs)
    diffs = arr[:, None] - arr[None, :]
    unique_diffs = np.unique(diffs)
    
    sum_count = len(unique_sums)
    diff_count = len(unique_diffs)
    
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1.0 + log(ratio) / log(D)

def construct_interval_set(alphas, base=1):
    """
    Construct a set where the gaps between consecutive elements grow geometrically.
    U = {0, g1, g1+g2, g1+g2+g3, ...}
    where g_{k+1} = g_k * alpha (with integer rounding)
    """
    if len(alphas) == 0:
        return [0]
    U = [0]
    current = 0
    gap = base
    for alpha in alphas:
        current += gap
        U.append(current)
        gap = max(1, int(gap * alpha))
    return U

def construct_geometric_alpha(alpha, n):
    """
    Construct a set with n-1 non-zero elements where gaps grow by factor alpha.
    U_k = sum_{i=0}^{k-1} floor(alpha^i) for k=1..n-1
    """
    if n <= 1:
        return [0]
    U = [0]
    current = 0
    for i in range(n - 1):
        gap = max(1, int(round(alpha ** i)))
        current += gap
        U.append(current)
    return U

def local_search(U, deadline, max_n=25, max_val=5000):
    """
    Local search with numpy-based scoring for speed.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    no_improve = 0
    max_no_improve = 40
    
    while time.time() < deadline - 3 and no_improve < max_no_improve:
        improved = False
        move_type = random.random()
        n = len(U)
        
        if move_type < 0.4:
            # ADD
            if n < max_n:
                # Generate candidates from structure
                U_set = set(U)
                candidates = set()
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val and s not in U_set:
                            candidates.add(s)
                        if 0 < d <= max_val and d not in U_set:
                            candidates.add(d)
                # Also try geometric extensions
                if n >= 2:
                    last_gap = U[-1] - U[-2]
                    for mult in [1.5, 2.0, 2.5, 3.0]:
                        new_val = U[-1] + int(last_gap * mult)
                        if 0 < new_val <= max_val and new_val not in U_set:
                            candidates.add(new_val)
                
                candidates = list(candidates)
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = sorted(U + [c])
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        elif move_type < 0.65:
            # REMOVE
            if n > 2:
                idx = random.randint(1, n - 1)
                new_U = [U[j] for j in range(n) if j != idx]
                new_score = score_set(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
        elif move_type < 0.85:
            # REPLACE
            if n > 1:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                # Try nearby values
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in U:
                        continue
                    new_U = sorted([U[j] if j != idx else new_val for j in range(n)])
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        else:
            # SWAP two elements
            if n > 2:
                i, j = random.sample(range(1, n), 2)
                U_copy = U[:]
                U_copy[i], U_copy[j] = U_copy[j], U_copy[i]
                U_copy = sorted(set(U_copy))
                if len(U_copy) == n:
                    new_score = score_set(U_copy)
                    if new_score > current_score:
                        U = U_copy
                        current_score = new_score
                        improved = True
        
        if improved:
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improve += 1
    
    return best_score, best_set

def ternary_search_alpha(n, deadline):
    """
    Optimize alpha for a fixed number of elements n using ternary search.
    """
    lo = 1.1
    hi = 5.0
    
    for _ in range(60):
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        U1 = construct_geometric_alpha(m1, n)
        U2 = construct_geometric_alpha(m2, n)
        s1 = score_set(U1)
        s2 = score_set(U2)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    opt_alpha = (lo + hi) / 2
    U = construct_geometric_alpha(opt_alpha, n)
    return score_set(U), U, opt_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_set = [0, 1, 4, 9, 16]
    best_score = score_set(best_set)
    
    # Phase 1: Scan geometric alpha sets for various n
    for n in range(3, 20):
        if time.time() > deadline - 60:
            break
        s, U, alpha = ternary_search_alpha(n, deadline)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Phase 2: Structured seeds
    structured_seeds = [
        [0, 1, 4, 9, 16, 25, 36],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 4, 10, 20, 35, 56],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 5, 14, 35, 82, 189],
        [0, 2, 5, 11, 23, 47, 95],
        [0, 1, 4, 10, 22, 44, 88],
    ]
    
    for seed in structured_seeds:
        if time.time() > deadline - 50:
            break
        s = score_set(seed)
        if s > best_score:
            best_score = s
            best_set = list(seed)
    
    # Phase 3: Local search from best and other good seeds
    seeds = [best_set]
    for seed in structured_seeds[:4]:
        if seed != best_set:
            seeds.append(seed)
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        s, U = local_search(list(base), deadline)
        if s > best_score:
            best_score = s
            best_set = list(U)
            # Update seed list
            if len(seeds) < 6:
                seeds.append(U)
            else:
                seeds[-1] = U
    
    # Phase 4: Final fine polish
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        improved = True
        iters = 0
        while improved and time.time() < deadline - 2 and iters < 300:
            improved = False
            iters += 1
            n = len(current)
            
            # Fine replace
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old = current[i]
                for delta in range(-8, 9):
                    if delta == 0:
                        continue
                    nv = old + delta
                    if nv <= 0 or nv in current:
                        continue
                    new_U = sorted([current[j] if j != i else nv for j in range(n)])
                    ns = score_set(new_U)
                    if ns > current_score:
                        current = new_U
                        current_score = ns
                        improved = True
                        break
                if improved:
                    break
            
            # Try add
            if not improved and n < 22:
                U_set = set(current)
                cands = set()
                for a in current:
                    for b in current:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= 5000 and s not in U_set:
                            cands.add(s)
                        if 0 < d <= 5000 and d not in U_set:
                            cands.add(d)
                for c in list(cands)[:10]:
                    new_U = sorted(current + [c])
                    ns = score_set(new_U)
                    if ns > current_score:
                        current = new_U
                        current_score = ns
                        improved = True
                        break
            
            # Try remove
            if not improved and n > 3:
                for i in range(1, n):
                    new_U = [current[j] for j in range(n) if j != i]
                    ns = score_set(new_U)
                    if ns > current_score:
                        current = new_U
                        current_score = ns
                        improved = True
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
