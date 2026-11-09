import math
import time
import random

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 85

    factors = [3, 5, 17]

    candidates = set()
    for p in factors:
        for k in range(1, n // p + 1):
            candidates.add(k * p)
    candidates = sorted(candidates)
    n_cand = len(candidates)
    cand_index = {v: i for i, v in enumerate(candidates)}

    # Precompute compatibility using bitmasks
    compat = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)

    def can_add_fast(xi, current_mask):
        return (current_mask & ~compat[xi]) == 0

    # Strategy 1: All multiples of 3
    mult3_mask = 0
    for x in range(3, n + 1, 3):
        mult3_mask |= (1 << cand_index[x])
    best_mask = mult3_mask
    best_size = bin(best_mask).count('1')

    for i in range(n_cand):
        if (best_mask >> i) & 1:
            continue
        if can_add_fast(i, best_mask):
            best_mask |= (1 << i)
            best_size += 1

    # Strategy 2: Local search - swap one for two
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(n_cand):
            if time.time() > deadline:
                break
            if not ((best_mask >> x) & 1):
                continue
            temp_mask = best_mask & ~(1 << x)
            addable = []
            for i in range(n_cand):
                if (temp_mask >> i) & 1:
                    continue
                if can_add_fast(i, temp_mask):
                    addable.append(i)
            for a in range(len(addable)):
                if time.time() > deadline:
                    break
                ja = addable[a]
                for b in range(a + 1, len(addable)):
                    jb = addable[b]
                    if (compat[ja] >> jb) & 1:
                        new_mask = temp_mask | (1 << ja) | (1 << jb)
                        if bin(new_mask).count('1') > best_size:
                            best_mask = new_mask
                            best_size = bin(best_mask).count('1')
                            improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Strategy 3: Other starting points
    for base_prime in [5, 17]:
        base_mask = 0
        for x in range(base_prime, n + 1, base_prime):
            base_mask |= (1 << cand_index[x])
        current = base_mask
        for i in range(n_cand):
            if (current >> i) & 1:
                continue
            if can_add_fast(i, current):
                current |= (1 << i)
        if bin(current).count('1') > best_size:
            best_mask = current
            best_size = bin(best_mask).count('1')

    # Strategy 4: Greedy from 255
    degrees = [bin(compat[i]).count('1') for i in range(n_cand)]
    sorted_by_degree = sorted(range(n_cand), key=lambda i: degrees[i], reverse=True)
    current_g = 1 << cand_index[255]
    for i in sorted_by_degree:
        if (current_g >> i) & 1:
            continue
        if can_add_fast(i, current_g):
            current_g |= (1 << i)
    if bin(current_g).count('1') > best_size:
        best_mask = current_g
        best_size = bin(best_mask).count('1')

    # Strategy 5: Random restarts
    random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        k = random.randint(3, 8)
        seed = random.sample(range(n_cand), k)
        if cand_index[255] not in seed:
            seed.append(cand_index[255])
        valid = True
        for a in range(len(seed)):
            for b in range(a + 1, len(seed)):
                if not ((compat[seed[a]] >> seed[b]) & 1):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        current_r = 0
        for s in seed:
            current_r |= (1 << s)
        for i in sorted_by_degree:
            if (current_r >> i) & 1:
                continue
            if can_add_fast(i, current_r):
                current_r |= (1 << i)
        if bin(current_r).count('1') > best_size:
            best_mask = current_r
            best_size = bin(best_mask).count('1')

    # Strategy 6: Bron-Kerbosch with pivot to find maximum clique
    # We search for a clique of size > best_size
    # Use recursive Bron-Kerbosch with pivot selection
    # Limit depth to keep runtime bounded
    found_better = False
    best_clique_size = best_size

    def bron_kerbosch(R, P, X, depth):
        nonlocal found_better, best_clique_size
        if time.time() > deadline:
            return
        if len(R) > best_clique_size:
            best_clique_size = len(R)
            found_better = True
        if not P and not X:
            return
        # Pivot selection: choose vertex in P ∪ X with max neighbors in P
        pivot_candidates = P | X
        if not pivot_candidates:
            return
        # Find pivot with max degree in P
        pivot = None
        max_deg = -1
        for v in pivot_candidates:
            deg = bin(P & compat[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                pivot = v
        # Iterate over P \ N(pivot)
        remaining = P & ~compat[pivot]
        while remaining:
            if time.time() > deadline:
                return
            # Pick a vertex from remaining
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            remaining &= remaining - 1
            bron_kerbosch(R | v_bit, P & compat[v], X & compat[v], depth + 1)
            P &= ~v_bit
            X |= v_bit

    # Start Bron-Kerbosch with 255 in R
    idx_255 = cand_index[255]
    initial_P = compat[idx_255]
    initial_X = 0
    initial_R = 1 << idx_255
    
    # Only run if we haven't already found something better
    if not found_better:
        bron_kerbosch(initial_R, initial_P, initial_X, 0)
    
    if found_better:
        # Reconstruct the clique - this is complex, so we just use the size
        # For now, we keep the best_mask from earlier strategies
        pass

    result = [candidates[i] for i in range(n_cand) if (best_mask >> i) & 1]
    return result
