import time
import random
import numpy as np

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def objective_fast(A):
    """Vectorized objective using NumPy - ~10-50x faster than set_sizes."""
    arr = np.array(A, dtype=np.int64)
    n = len(arr)
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    prods = arr[i_idx] * arr[j_idx]
    # Count unique values using sorting (faster than set for arrays)
    s_count = len(np.unique(sums))
    p_count = len(np.unique(prods))
    return max(s_count, p_count)

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s
    
    random.seed(42)
    
    def valid_set(A):
        return len(set(A)) == 20 and all(1 <= x <= 10000 for x in A)
    
    # --- Lattice-based construction: minimize |A*A| via exponent structure ---
    # Represent elements as 2^i * 3^j * 5^k. Products become exponent sums.
    # Dense exponent clusters yield small product sets.
    
    def lattice_set_2d(subset_indices):
        """Generate a set from 2^i * 3^j exponent lattice points."""
        vals = []
        for (i, j) in subset_indices:
            v = (2 ** i) * (3 ** j)
            if v <= 10000:
                vals.append(v)
        return vals
    
    def lattice_set_3d(subset_indices):
        """Generate a set from 2^i * 3^j * 5^k exponent lattice points."""
        vals = []
        for (i, j, k) in subset_indices:
            v = (2 ** i) * (3 ** j) * (5 ** k)
            if v <= 10000:
                vals.append(v)
        return vals
    
    # Try various lattice subsets
    lattice_candidates = []
    
    # Dense 2D grid: i in 0..i_max, j in 0..j_max
    for i_max in range(0, 14):
        for j_max in range(0, 10):
            points = [(i, j) for i in range(i_max + 1) for j in range(j_max + 1)]
            if len(points) >= 20:
                points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
                subset = points_sorted[:20]
                vals = lattice_set_2d(subset)
                if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                    lattice_candidates.append(sorted(vals))
    
    # Lines in the 2D lattice
    for c in range(0, 20):
        points = []
        for j in range(0, 10):
            i = c - 2 * j
            if 0 <= i <= 14:
                points.append((i, j))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:20]
            vals = lattice_set_2d(subset)
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))
    
    for c in range(0, 25):
        points = []
        for i in range(0, 14):
            j = c - 2 * i
            if 0 <= j <= 10:
                points.append((i, j))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:20]
            vals = lattice_set_2d(subset)
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))
    
    # 3D lattice: 2^i * 3^j * 5^k
    # Small 3D grids give very compact product sets
    for i_max in range(0, 5):
        for j_max in range(0, 4):
            for k_max in range(0, 4):
                points = [(i, j, k) for i in range(i_max + 1) 
                          for j in range(j_max + 1) 
                          for k in range(k_max + 1)]
                if len(points) >= 20:
                    points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]) * (5 ** p[2]))
                    subset = points_sorted[:20]
                    vals = lattice_set_3d(subset)
                    if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                        lattice_candidates.append(sorted(vals))
    
    # 3D lattice lines/planes
    # Plane: i + j + k = c
    for c in range(0, 12):
        points = []
        for i in range(0, 6):
            for j in range(0, 5):
                k = c - i - j
                if 0 <= k <= 4:
                    points.append((i, j, k))
        if len(points) >= 20:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]) * (5 ** p[2]))
            subset = points_sorted[:20]
            vals = lattice_set_3d(subset)
            if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                lattice_candidates.append(sorted(vals))
    
    # Mixed: combine small grid with some offset points
    base_grid = [(i, j) for i in range(0, 5) for j in range(0, 5)]
    for extra_i in range(5, 12):
        for extra_j in range(0, 8):
            subset = base_grid[:15] + [(extra_i, extra_j)]
            if len(subset) >= 20:
                subset = subset[:20]
                vals = lattice_set_2d(subset)
                if len(set(vals)) == 20 and all(1 <= v <= 10000 for v in vals):
                    lattice_candidates.append(sorted(vals))
    
    # Deduplicate lattice candidates
    seen = set()
    unique_lattice = []
    for cand in lattice_candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique_lattice.append(cand)
    
    # Add the original hand-crafted candidates
    hand_crafted = [
        list(range(1, 21)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96],
    ]
    
    # Geometric progressions
    for base in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]:
        A = [int(round(base ** i)) for i in range(20)]
        seen_g = set()
        uniq = []
        for x in A:
            if 1 <= x <= 10000 and x not in seen_g:
                seen_g.add(x)
                uniq.append(x)
        while len(uniq) < 20:
            uniq.append(uniq[-1] + 1)
        if len(uniq) == 20 and all(1 <= x <= 10000 for x in uniq):
            hand_crafted.append(uniq)
    
    all_candidates = unique_lattice + hand_crafted
    
    # Evaluate all initial candidates (use fast objective)
    best_set = None
    best_obj = float('inf')
    for cand in all_candidates:
        if not valid_set(cand):
            continue
        obj = objective_fast(cand)
        if obj < best_obj:
            best_obj = obj
            best_set = cand[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective_fast(best_set)
    
    # --- Local search refinement (using fast objective) ---
    def local_search(A_init, max_iters, mode="single"):
        nonlocal best_set, best_obj
        A = list(A_init)
        cur_obj = objective_fast(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        
        it = 0
        while it < max_iters:
            if time.time() > deadline:
                return
            it += 1
            
            if mode == "single":
                # Replace one element
                i = random.randrange(20)
                x = A[i]
                cands = set()
                # Multiplicative neighbors (exploit lattice structure)
                for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                    nv = int(round(x * factor))
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Additive neighbors
                for d in [-3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100]:
                    nv = x + d
                    if 1 <= nv <= 10000:
                        cands.add(nv)
                # Random
                cands.add(random.randrange(1, 10001))
                # Lattice-inspired: multiply/divide by 2, 3, or 5
                for div_mult in [2, 3, 5]:
                    cands.add(x * div_mult)
                    if x % div_mult == 0:
                        cands.add(x // div_mult)
                cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                if not cands:
                    continue
                nv = random.choice(cands)
                old = A[i]
                A[i] = nv
                new_obj = objective_fast(A)
                if new_obj <= cur_obj:
                    cur_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                else:
                    A[i] = old
                    
            elif mode == "swap":
                # Swap two elements
                i, j = random.sample(range(20), 2)
                A[i], A[j] = A[j], A[i]
                new_obj = objective_fast(A)
                if new_obj <= cur_obj:
                    cur_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                else:
                    A[i], A[j] = A[j], A[i]
                    
            elif mode == "perturb":
                # Perturb multiple elements
                num_perturb = random.randint(1, 3)
                indices = random.sample(range(20), num_perturb)
                old_vals = {i: A[i] for i in indices}
                for i in indices:
                    x = A[i]
                    cands = set()
                    for factor in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0]:
                        nv = int(round(x * factor))
                        if 1 <= nv <= 10000:
                            cands.add(nv)
                    cands.add(x + random.randint(-20, 20))
                    cands.add(random.randrange(1, 10001))
                    for div_mult in [2, 3, 5]:
                        cands.add(x * div_mult)
                        if x % div_mult == 0:
                            cands.add(x // div_mult)
                    cands = [c for c in cands if c not in A and 1 <= c <= 10000]
                    if cands:
                        A[i] = random.choice(cands)
                new_obj = objective_fast(A)
                if new_obj <= cur_obj:
                    cur_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_set = A[:]
                else:
                    for i in indices:
                        A[i] = old_vals[i]
        
        return
    
    # Run local search from best candidate (more iterations due to speedup)
    local_search(best_set, 80000, mode="single")
    
    # Try local search from top lattice candidates
    top_lattice = []
    for cand in unique_lattice[:15]:
        if valid_set(cand):
            top_lattice.append(cand)
    
    for start_set in top_lattice:
        if time.time() > deadline - 15:
            break
        mode = random.choice(["single", "swap", "perturb"])
        local_search(start_set, 20000, mode=mode)
    
    # Random restarts with lattice-inspired generation (3D-aware)
    for trial in range(40):
        if time.time() > deadline - 8:
            break
        # Generate lattice-inspired random set (mix 2D and 3D)
        A = set()
        if random.random() < 0.5:
            # 2D lattice
            for _ in range(20):
                i = random.randint(0, 13)
                j = random.randint(0, 9)
                v = (2 ** i) * (3 ** j)
                if v <= 10000:
                    A.add(v)
        else:
            # 3D lattice
            for _ in range(20):
                i = random.randint(0, 5)
                j = random.randint(0, 4)
                k = random.randint(0, 4)
                v = (2 ** i) * (3 ** j) * (5 ** k)
                if v <= 10000:
                    A.add(v)
        # Fill with random if needed
        A_list = sorted(A)
        while len(A_list) < 20:
            v = random.randrange(1, 10001)
            if v not in A_list:
                A_list.append(v)
        A_list = sorted(A_list[:20])
        if valid_set(A_list):
            local_search(A_list, 15000, mode="single")
    
    # Final refinement from best (extended due to speedup)
    if time.time() < deadline - 3:
        local_search(best_set, 50000, mode="single")
    
    return best_set
