import time
import math
import random
import numpy as np

BASE = 51975

def _divisors(n):
    ds = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
    return sorted(ds)

MODULI = [m for m in _divisors(BASE) if m > 1]

# Precompute index arrays for each modulus and residue
# idx_cache[m][r] = np.arange(r, BASE, m)
idx_cache = {}
for m in MODULI:
    idx_cache[m] = [np.arange(r, BASE, m) for r in range(m)]

def score_classes(classes):
    covered = np.zeros(BASE, dtype=np.int8)
    for m, r in classes:
        covered[idx_cache[m][r]] = 1
    return BASE - int(np.count_nonzero(covered))

def construct():
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    def get_best_residue(m, cnt):
        """Find residue r maximizing uncovered points covered by class (m, r)."""
        # For each residue r, count how many x in idx_cache[m][r] have cnt[x] == 0
        # We can do this efficiently by checking cnt at those positions
        unc = (cnt == 0)
        gains = np.array([int(np.sum(unc[idx_cache[m][r]])) for r in range(m)], dtype=np.int32)
        return int(np.argmax(gains)), int(gains.max())

    def build_cnt(used):
        cnt = np.zeros(BASE, dtype=np.int32)
        for m, r in used.items():
            cnt[idx_cache[m][r]] += 1
        return cnt

    def covered_count(cnt):
        return int(np.count_nonzero(cnt))

    def greedy(order):
        cnt = np.zeros(BASE, dtype=np.int32)
        used = {}
        for m in order:
            best_r, _ = get_best_residue(m, cnt)
            cnt[idx_cache[m][best_r]] += 1
            used[m] = best_r
        return used

    # Generate greedy starts with different orders
    orders = [
        sorted(MODULI),
        sorted(MODULI, reverse=True),
    ]
    for _ in range(6):
        o = MODULI[:]
        rng.shuffle(o)
        orders.append(o)

    best_score = BASE
    best_classes = None
    starts = []

    for o in orders:
        used = greedy(o)
        cnt = build_cnt(used)
        sc = BASE - covered_count(cnt)
        starts.append(used)
        if sc < best_score:
            best_score = sc
            best_classes = [[m, used[m]] for m in used]

    # Local search with simulated annealing
    def local_search_sa(used_init, max_iter, t_start=1.0, t_end=0.01, cooling=0.9995):
        used = dict(used_init)
        cnt = build_cnt(used)
        cur_sc = BASE - covered_count(cnt)
        best_sc = cur_sc
        best_used = dict(used)
        t = t_start
        
        moduli_list = MODULI[:]
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            action = rng.random()
            
            if action < 0.65:
                # Switch residue of an existing modulus
                if not used:
                    continue
                m = rng.choice(list(used.keys()))
                cur_r = used[m]
                
                # Compute gain for current residue
                unc = (cnt == 0)
                cur_gain = int(np.sum(unc[idx_cache[m][cur_r]]))
                
                # Find best residue
                best_r, best_gain = get_best_residue(m, cnt)
                
                if best_r != cur_r and best_gain > cur_gain:
                    # Apply change
                    cnt[idx_cache[m][cur_r]] -= 1
                    cnt[idx_cache[m][best_r]] += 1
                    used[m] = best_r
                    new_sc = BASE - covered_count(cnt)
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_used = dict(used)
                    cur_sc = new_sc
                elif best_r != cur_r:
                    # Consider accepting worse move (simulated annealing)
                    delta = best_gain - cur_gain  # negative
                    if delta < 0 and rng.random() < math.exp(delta / max(t, 1e-9)):
                        cnt[idx_cache[m][cur_r]] -= 1
                        cnt[idx_cache[m][best_r]] += 1
                        used[m] = best_r
                        cur_sc = BASE - covered_count(cnt)
                
            elif action < 0.85:
                # Add a new modulus
                unused = [m for m in MODULI if m not in used]
                if not unused:
                    continue
                m = rng.choice(unused)
                best_r, best_gain = get_best_residue(m, cnt)
                
                if best_gain > 0:
                    cnt[idx_cache[m][best_r]] += 1
                    used[m] = best_r
                    new_sc = BASE - covered_count(cnt)
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_used = dict(used)
                    cur_sc = new_sc
                elif best_gain == 0:
                    # Accept with small probability to explore
                    if rng.random() < math.exp(-1.0 / max(t, 1e-9)):
                        cnt[idx_cache[m][best_r]] += 1
                        used[m] = best_r
                        cur_sc = BASE - covered_count(cnt)
                        
            else:
                # Remove a modulus
                if not used:
                    continue
                m = rng.choice(list(used.keys()))
                r = used[m]
                idx = idx_cache[m][r]
                
                # Check if redundant
                if np.all(cnt[idx] > 1):
                    cnt[idx] -= 1
                    del used[m]
                    # cur_sc unchanged
                    # best_sc unchanged
                else:
                    # Compute how many points would become uncovered
                    # Points where cnt[x] == 1 in this class
                    would_uncover = int(np.sum(cnt[idx] == 1))
                    # Accept removal with small probability
                    if would_uncover == 0:
                        cnt[idx] -= 1
                        del used[m]
                    elif rng.random() < math.exp(-would_uncover / max(t, 1e-9)):
                        cnt[idx] -= 1
                        del used[m]
                        cur_sc = BASE - covered_count(cnt)
                        if cur_sc < best_sc:
                            best_sc = cur_sc
                            best_used = dict(used)
            
            t *= cooling
        
        return best_used, best_sc

    # Run local search from each start
    for i, used0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        # Use adaptive iteration count based on remaining time
        remaining = deadline - 10 - time.time()
        max_iter = int(min(8000, max(1000, remaining * 30)))
        
        used, sc = local_search_sa(used0, max_iter, t_start=1.0, t_end=0.01, cooling=0.9995)
        
        if sc < best_score:
            best_score = sc
            best_classes = [[m, used[m]] for m in used]

    # Final polish: try all possible residue changes for each modulus in best solution
    if best_classes and time.time() < deadline - 3:
        used = {m: r for m, r in best_classes}
        cnt = build_cnt(used)
        
        # Try switching each modulus to its best residue
        for m in list(used.keys()):
            if time.time() > deadline - 3:
                break
            best_r, best_gain = get_best_residue(m, cnt)
            if best_r != used[m]:
                cnt[idx_cache[m][used[m]]] -= 1
                cnt[idx_cache[m][best_r]] += 1
                used[m] = best_r
        
        # Try adding any unused moduli that help
        for m in MODULI:
            if time.time() > deadline - 3:
                break
            if m in used:
                continue
            best_r, best_gain = get_best_residue(m, cnt)
            if best_gain > 0:
                cnt[idx_cache[m][best_r]] += 1
                used[m] = best_r
        
        # Try removing redundant moduli
        for m in list(used.keys()):
            if time.time() > deadline - 3:
                break
            r = used[m]
            idx = idx_cache[m][r]
            if np.all(cnt[idx] > 1):
                cnt[idx] -= 1
                del used[m]
        
        sc = BASE - covered_count(cnt)
        if sc < best_score:
            best_score = sc
            best_classes = [[m, used[m]] for m in used]

    if best_classes is None:
        used = greedy(sorted(MODULI))
        best_classes = [[m, used[m]] for m in used]
        
    return best_classes
