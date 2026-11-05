import time
import random
from itertools import combinations

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S that is a 'strong Sidon' set.
    A strong Sidon set has all unordered pair sums (including diagonals) unique.
    """
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    best = 0

    # DFS with backtracking
    # State: start_idx, current_indices (tuple), used_sums (set)
    # We iterate through elements in sorted order to avoid permutations of the same subset.
    
    def dfs(start_idx, current_indices, used_sums):
        nonlocal best
        current_size = len(current_indices)
        if current_size > best:
            best = current_size
            
        # Pruning: if we can't beat best with remaining elements, return
        if current_size + (n - start_idx) <= best:
            return

        for i in range(start_idx, n):
            x = S[i]
            # Check if x is compatible with current_indices
            valid = True
            
            # Check diagonal: 2*x
            if 2*x in used_sums:
                valid = False
            
            # Check off-diagonals: x + S[j] for j in current_indices
            if valid:
                for j_idx in current_indices:
                    s = x + S[j_idx]
                    if s in used_sums:
                        valid = False
                        break
            
            if valid:
                # Add x
                new_indices = current_indices + (i,)
                new_used_sums = set(used_sums)
                new_used_sums.add(2*x)
                for j_idx in current_indices:
                    new_used_sums.add(x + S[j_idx])
                
                dfs(i + 1, new_indices, new_used_sums)

    dfs(0, (), set())
    return best

def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)

def _count_distinct_sums(c):
    """Fast proxy: count distinct pair sums. Lower is generally better for minimizing Sidon size."""
    sums = set()
    for i in range(12):
        for j in range(i, 12):
            sums.add(c[i] + c[j])
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_set = None
    best_val = 99
    seen_sets = set()

    def consider(c):
        nonlocal best_set, best_val
        if not _valid(c):
            return
        key = tuple(sorted(c))
        if key in seen_sets:
            return
        seen_sets.add(key)
        
        # Use proxy first to filter out obviously bad candidates if time is tight?
        # For now, evaluate exactly if it's a new candidate.
        # To speed up, we can check the proxy score and only evaluate if it's promising,
        # but since max_sidon_size is fast for n=12, we can just evaluate.
        
        val = max_sidon_size(c)
        if val < best_val:
            best_val = val
            best_set = c.copy()
            # Early exit if we hit theoretical minimum (likely 4 or 5)
            if best_val <= 4:
                return True
        return False

    # Phase 1: Generate and evaluate structured candidates
    # 1. Arithmetic Progressions
    for d in range(1, 12):
        for start in range(0, 128 - 11*d + 1):
            c = [start + i*d for i in range(12)]
            if consider(c):
                if best_val <= 4: return best_set

    # 2. Union of two APs
    # {a + i*d1} U {b + j*d2}
    for d1 in range(2, 15):
        for d2 in range(2, 15):
            if d1 == d2: continue
            for a in range(0, 20):
                for b in range(0, 20):
                    c1 = [a + i*d1 for i in range(6)]
                    c2 = [b + j*d2 for j in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12:
                        if consider(c):
                            if best_val <= 4: return best_set

    # 3. Grids
    for r in range(2, 7):
        for c_cols in range(2, 7):
            if r * c_cols != 12: continue
            for rs in range(5, 20):
                for cs in range(2, 10):
                    c = []
                    for i in range(r):
                        for j in range(c_cols):
                            val = i*rs + j*cs
                            if val <= 128:
                                c.append(val)
                    if len(c) == 12:
                        c = sorted(set(c))
                        if consider(c):
                            if best_val <= 4: return best_set

    # 4. Random candidates with small diameter
    rng = random.Random(42)
    for _ in range(2000):
        if time.time() > deadline - 10:
            break
        lo = rng.randint(0, 128 - 30)
        hi = lo + rng.randint(12, 30)
        if hi > 128:
            hi = 128
        if hi - lo < 11:
            continue
        c = sorted(rng.sample(range(lo, hi + 1), 12))
        if consider(c):
            if best_val <= 4: return best_set

    # Phase 2: Local Search
    if best_set is not None:
        cur = best_set.copy()
        cur_val = best_val
        
        # Sort candidates by proxy score to find better starting points for local search
        # We already have best_set. Let's try to improve it.
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try swapping each element with a random one
            for pos in range(12):
                if time.time() > deadline - 5:
                    break
                
                old_val = cur[pos]
                # Try values near existing elements to maintain structure
                candidates_repl = set()
                for other in cur:
                    for delta in range(-10, 11):
                        nv = other + delta
                        if 0 <= nv <= 128 and nv != old_val:
                            candidates_repl.add(nv)
                
                # Also try some random
                for _ in range(10):
                    nv = rng.randint(0, 128)
                    if nv != old_val:
                        candidates_repl.add(nv)
                        
                for nv in candidates_repl:
                    if nv in cur:
                        continue
                    new_cur = cur.copy()
                    new_cur[pos] = nv
                    new_cur.sort()
                    
                    if not _valid(new_cur):
                        continue
                    
                    # Check proxy first
                    proxy_new = _count_distinct_sums(new_cur)
                    # We don't have proxy_cur stored, but if proxy_new is much higher, skip?
                    # Actually, just evaluate exactly.
                    new_val = max_sidon_size(new_cur)
                    if new_val < cur_val:
                        cur = new_cur
                        cur_val = new_val
                        best_val = cur_val
                        best_set = cur.copy()
                        improved = True
                        break # Restart pos loop
                if improved:
                    break
            
            if not improved:
                # Try random perturbations
                found_better = False
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    new_cur = cur.copy()
                    p1, p2 = rng.sample(range(12), 2)
                    v1 = rng.randint(0, 128)
                    v2 = rng.randint(0, 128)
                    new_cur[p1] = v1
                    new_cur[p2] = v2
                    new_cur.sort()
                    if _valid(new_cur):
                        new_val = max_sidon_size(new_cur)
                        if new_val < cur_val:
                            cur = new_cur
                            cur_val = new_val
                            best_val = cur_val
                            best_set = cur.copy()
                            found_better = True
                            break
                if not found_better:
                    # Stuck in local minimum. Break or restart?
                    # Given time constraints, breaking is safer.
                    break

    if best_set is None:
        # Fallback
        best_set = list(range(12))
    
    return best_set
