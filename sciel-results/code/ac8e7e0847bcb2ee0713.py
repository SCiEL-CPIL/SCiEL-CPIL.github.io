import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256
    all_mask = (1 << N) - 1

    # Precompute conflict masks
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    def bit_count(x):
        return x.bit_count()

    # ---- Spectral seeding: compute principal eigenvector of conflict adjacency ----
    # Use power iteration with NumPy if available; fallback to simple degree-based score.
    try:
        import numpy as np
        # Build adjacency matrix (symmetric)
        adj = np.zeros((N, N), dtype=np.float64)
        for i in range(1, N + 1):
            m = conflict_mask[i]
            for j in range(1, N + 1):
                if m & (1 << (j - 1)):
                    adj[i - 1, j - 1] = 1.0
        # Power iteration for largest eigenvector
        v = np.random.rand(N)
        v /= np.linalg.norm(v)
        for _ in range(200):
            v_new = adj @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v = v_new / norm
        # Spectral score: vertices with LOW conflict centrality are good to include
        # We want independent set, so prefer vertices far from the conflict core.
        # Use negative eigenvector component as priority (lower = less central = better)
        spectral_priority = -v  # lower value = include first
    except Exception:
        # Fallback: use degree as proxy (lower degree = less conflict = better)
        spectral_priority = []
        for i in range(1, N + 1):
            spectral_priority.append(-bit_count(conflict_mask[i]))
        spectral_priority = np.array(spectral_priority) if False else spectral_priority

    # ---- Greedy min-degree independent set ----
    def greedy_min_degree(available, randomize=False, k=1, priority_order=None):
        sol = []
        sol_mask = 0
        rem = available

        while rem:
            if randomize and k > 1:
                # Find min degree, collect near-min candidates, pick randomly
                best_deg = float('inf')
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < best_deg:
                        best_deg = deg
                    t ^= lsb

                candidates = []
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg <= best_deg + 1:
                        candidates.append(v)
                    t ^= lsb

                if not candidates:
                    lsb = rem & (-rem)
                    v = lsb.bit_length()
                else:
                    kk = min(k, len(candidates))
                    v = candidates[random.randrange(kk)]
            elif priority_order is not None:
                # Pick the available vertex with best (lowest) priority score
                best_v = -1
                best_score = float('inf')
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    score = priority_order[v - 1]
                    if score < best_score:
                        best_score = score
                        best_v = v
                    t ^= lsb
                v = best_v
            else:
                # Deterministic min-degree
                min_v = -1
                min_deg = float('inf')
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp ^= lsb
                v = min_v

            if v < 0:
                break

            sol.append(v)
            sol_mask |= (1 << (v - 1))
            rem &= ~(1 << (v - 1))
            rem &= ~conflict_mask[v]

        return sol, sol_mask

    # ---- Polish: greedily add any vertex that doesn't conflict ----
    def polish(mask):
        size = bit_count(mask)
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if mask & bit:
                continue
            if conflict_mask[v] & mask:
                continue
            mask |= bit
            size += 1
        return mask, size

    best_mask = 0
    best_size = 0

    # Phase 1: Spectral-seeded greedy (new)
    try:
        sol, sol_mask = greedy_min_degree(all_mask, randomize=False, priority_order=spectral_priority)
        sol_mask, sz = polish(sol_mask)
        if sz > best_size:
            best_size = sz
            best_mask = sol_mask
    except Exception:
        pass

    # Phase 2: Deterministic greedy
    sol, sol_mask = greedy_min_degree(all_mask, randomize=False)
    sol_mask, sz = polish(sol_mask)
    if sz > best_size:
        best_size = sz
        best_mask = sol_mask

    # Phase 3: Randomized greedy restarts
    for it in range(400):
        if time.time() > deadline:
            break
        sol, sol_mask = greedy_min_degree(all_mask, randomize=True, k=5)
        sol_mask, sz = polish(sol_mask)
        if sz > best_size:
            best_size = sz
            best_mask = sol_mask
            if best_size >= 140:
                break

    # Phase 4: Random-order greedy with polish
    for it in range(3000):
        if time.time() > deadline:
            break
        rem = all_mask
        sol_mask = 0
        sz = 0
        verts = list(range(1, N + 1))
        random.shuffle(verts)
        for v in verts:
            bit = 1 << (v - 1)
            if not (rem & bit):
                continue
            if conflict_mask[v] & sol_mask:
                continue
            sol_mask |= bit
            sz += 1
            rem &= ~conflict_mask[v]
            rem &= ~bit
        sol_mask, sz = polish(sol_mask)
        if sz > best_size:
            best_size = sz
            best_mask = sol_mask

    # Phase 5: 1-for-2 swap local search
    def try_improve_1for2(mask, size):
        sol_list = [i + 1 for i in range(N) if mask & (1 << i)]
        if not sol_list:
            return None, size
        for _ in range(300):
            if time.time() > deadline:
                break
            v_out = random.choice(sol_list)
            new_mask = mask & ~(1 << (v_out - 1))
            addable = []
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if new_mask & bit:
                    continue
                if conflict_mask[v] & new_mask:
                    continue
                addable.append(v)
            if len(addable) >= 2:
                a, b = random.sample(addable, 2)
                if conflict_mask[a] & (1 << (b - 1)):
                    continue
                new_mask2 = new_mask | (1 << (a - 1)) | (1 << (b - 1))
                return new_mask2, size + 1
        return None, size

    for round_num in range(100):
        if time.time() > deadline:
            break
        new_mask, new_size = try_improve_1for2(best_mask, best_size)
        if new_mask is not None:
            best_mask = new_mask
            best_size = new_size
        else:
            # Try single adds
            added = False
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1
                added = True
            if not added:
                break

    # Final polish
    best_mask, best_size = polish(best_mask)

    final_sol = [i + 1 for i in range(N) if best_mask & (1 << i)]
    if not final_sol:
        final_sol = [1]

    return final_sol
