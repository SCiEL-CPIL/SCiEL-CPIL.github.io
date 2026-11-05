import time
import random
import numpy as np

def set_sizes(A):
    """Compute sizes of A+A and A*A with equal summands/factors allowed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # All pairs (i,j) with i <= j
    ii, jj = np.triu_indices(n)
    sums = A[ii] + A[jj]
    prods = A[ii] * A[jj]
    return len(set(sums.tolist())), len(set(prods.tolist()))

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    best_set = None
    best_obj = float('inf')

    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = A[:]

    # Family 1: Pure geometric sequences
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        A = []
        v = base
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000:
            try_candidate(A)
        else:
            for start_val in [1, 2, 3]:
                A = []
                v = start_val
                for i in range(20):
                    A.append(v)
                    v *= base
                if max(A) <= 10000 and len(set(A)) == 20:
                    try_candidate(A)
                    break

    # Family 2: Geometric with ratio close to 1 (dense in log space)
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]:
        for start_val in [1, 2, 3, 4, 5, 10]:
            A = []
            v = float(start_val)
            for i in range(20):
                val = int(round(v))
                if val < 1:
                    val = 1
                A.append(val)
                v *= ratio
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)

    # Family 3: Powers of 2 times small odd numbers
    for offset in [1, 2, 3, 5]:
        for shift in [0, 1, 2]:
            A = []
            for i in range(20):
                val = (2**i) * (1 + (i + shift) * offset)
                if val > 10000:
                    val = 10000
                A.append(val)
            seen = set()
            uniq = []
            for x in A:
                if x not in seen and 1 <= x <= 10000:
                    seen.add(x)
                    uniq.append(x)
            if len(uniq) == 20:
                try_candidate(uniq)

    # Family 4: Sparse geometric
    for base in [2, 3, 4, 5]:
        for step in [1, 2, 3]:
            A = [base**(i*step) for i in range(20)]
            if max(A) <= 10000 and len(set(A)) == 20:
                try_candidate(A)

    # Family 5: Mixed
    for base in [2, 3, 4]:
        A = [base**i for i in range(10)]
        last = base**9
        for i in range(10):
            A.append(last * (i + 2))
        seen = set()
        uniq = []
        for x in A:
            if x not in seen and 1 <= x <= 10000:
                seen.add(x)
                uniq.append(x)
        if len(uniq) == 20:
            try_candidate(uniq)

    # Family 6: 2D lattice (2^i * 3^j) - structurally minimizes product set
    # Product of two lattice elements: 2^(i1+i2) * 3^(j1+j2)
    # Sumset is harder to control but product set is very small
    lattice_seeds = []
    for max_i in range(0, 15):
        for max_j in range(0, 12):
            # Generate all (i,j) pairs in range [0,max_i] x [0,max_j]
            pairs = [(i, j) for i in range(max_i + 1) for j in range(max_j + 1)]
            if len(pairs) < 20:
                continue
            # Compute values
            vals = {}
            valid_pairs = []
            for (i, j) in pairs:
                v = (2 ** i) * (3 ** j)
                if v <= 10000:
                    if v not in vals:
                        vals[v] = (i, j)
                        valid_pairs.append((i, j))
            if len(valid_pairs) >= 20:
                # Take first 20 or random 20
                if len(valid_pairs) == 20:
                    A = [vals[p[0] * 100 + p[1]] for p in valid_pairs]
                    if len(set(A)) == 20:
                        lattice_seeds.append(A)
                        try_candidate(A)
                else:
                    # Random subset of 20
                    for _ in range(5):
                        subset = random.sample(valid_pairs, 20)
                        A = [vals[p[0] * 100 + p[1]] for p in subset]
                        if len(set(A)) == 20:
                            lattice_seeds.append(A)
                            try_candidate(A)
                    break
        if time.time() > deadline - 5:
            break

    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = objective(best_set)

    random.seed(42)

    def local_search(start_set, max_iters=50000):
        nonlocal best_set, best_obj
        A = list(start_set)
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
        it = 0
        no_improve = 0
        while it < max_iters:
            if time.time() > deadline:
                return
            it += 1
            i = random.randrange(20)
            x = A[i]
            cands = set()
            for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            for delta in [-10, -5, -2, -1, 1, 2, 5, 10, 50, 100]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            cands.add(random.randrange(1, 10001))
            cands.discard(x)
            cands = [c for c in cands if c not in A]
            if not cands:
                continue
            nv = random.choice(cands)
            old = A[i]
            A[i] = nv
            new_obj = objective(A)
            if new_obj <= cur_obj:
                cur_obj = new_obj
                no_improve = 0
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old
                no_improve += 1
            if no_improve > 5000:
                break

    local_search(best_set, max_iters=30000)

    other_starts = [
        list(range(1, 21)),
        [2**i for i in range(20)] if 2**19 <= 10000 else list(range(1, 21)),
    ]
    for base in [2, 3]:
        A = []
        v = 1
        for i in range(20):
            A.append(v)
            v *= base
        if max(A) <= 10000 and len(set(A)) == 20:
            other_starts.append(A)

    # Add lattice seeds to other starts
    for ls in lattice_seeds[:3]:
        other_starts.append(ls)

    for sp in other_starts:
        if time.time() > deadline - 3:
            break
        if sp is best_set:
            continue
        local_search(sp, max_iters=15000)

    # Final polish: more aggressive local search with tighter neighborhood
    if time.time() < deadline - 2:
        A = list(best_set)
        cur_obj = objective(A)
        no_improve = 0
        it = 0
        while it < 50000 and no_improve < 2000 and time.time() < deadline - 1:
            it += 1
            i = random.randrange(20)
            x = A[i]
            cands = set()
            # Tighter neighborhood for final polish
            for factor in [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]:
                nv = int(round(x * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            for delta in [-3, -2, -1, 1, 2, 3]:
                nv = x + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            cands.discard(x)
            cands = [c for c in cands if c not in A]
            if not cands:
                no_improve += 1
                continue
            nv = random.choice(cands)
            old = A[i]
            A[i] = nv
            new_obj = objective(A)
            if new_obj <= cur_obj:
                cur_obj = new_obj
                no_improve = 0
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_set = A[:]
            else:
                A[i] = old
                no_improve += 1

    return best_set
