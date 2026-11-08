import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Rank = 105 - (number of disjoint 2x2x2 Strassen blocks).
    Uses bitmask overlap structure with enhanced local search (multi-swap moves)
    and simulated annealing to find maximum packing of disjoint Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total = n * m * p  # 105

    # Precompute all blocks: each = (params, frozenset of (i,j,k), bitmask)
    blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            triples = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = (i * m + j) * p + k
                                        mask |= (1 << idx)
                                        triples.append((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(triples), mask))

    B = len(blocks)
    # Overlap matrix as bitmasks: ov[a] has bit b set if blocks a,b share a triple
    ov = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        bits = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]):
                bits |= (1 << b)
        ov[a] = bits

    def greedy(packing_order):
        used = 0
        res = []
        for b in packing_order:
            if (used & blocks[b][2]) == 0:
                used |= blocks[b][2]
                res.append(b)
        return res

    # Precompute: for each block, count how many other blocks it conflicts with
    conflict_count = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        cnt = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]):
                cnt += 1
        conflict_count[a] = cnt

    def local_search_enhanced(packing, max_iters=80):
        """Enhanced local search with multi-swap moves and simulated annealing."""
        used = 0
        for b in packing:
            used |= blocks[b][2]
        
        cur_len = len(packing)
        best_packing = list(packing)
        best_used = used
        best_len = cur_len
        
        temperature = 8.0
        cooling = 0.93
        min_temp = 0.05
        
        rng = random.Random(hash(tuple(sorted(packing))) % (2**31))
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            if temperature < min_temp:
                break
            
            improved = False
            
            # Try single swap: remove one, add one
            for c in packing:
                used2 = used & ~blocks[c][2]
                # Find all blocks that fit
                candidates = []
                for a in range(B):
                    if a not in packing and (used2 & blocks[a][2]) == 0:
                        candidates.append(a)
                
                if candidates:
                    # Score: prefer blocks with lower conflict_count (more compatible)
                    best_cand = None
                    best_score = -1
                    for a in candidates[:30]:
                        score = -conflict_count[a]
                        if score > best_score:
                            best_score = score
                            best_cand = a
                    
                    if best_cand is not None:
                        new_packing = [x for x in packing if x != c]
                        new_packing.append(best_cand)
                        new_len = len(new_packing)
                        
                        if new_len > cur_len:
                            packing = new_packing
                            used = best_used = used2 | blocks[best_cand][2]
                            cur_len = new_len
                            best_packing = list(packing)
                            best_len = new_len
                            improved = True
                            break
                        elif new_len == cur_len and rng.random() < temperature * 0.15:
                            packing = new_packing
                            used = used2 | blocks[best_cand][2]
                            improved = True
                            break
            
            if not improved:
                # Try double swap: remove two, add one or two
                for c1_idx in range(len(packing)):
                    c1 = packing[c1_idx]
                    used_after_c1 = used & ~blocks[c1][2]
                    for c2_idx in range(len(packing)):
                        if c2_idx == c1_idx:
                            continue
                        c2 = packing[c2_idx]
                        used_after_both = used_after_c1 & ~blocks[c2][2]
                        
                        candidates = []
                        for a in range(B):
                            if a not in packing and (used_after_both & blocks[a][2]) == 0:
                                candidates.append(a)
                        
                        if len(candidates) >= 2:
                            for a in candidates[:15]:
                                used3 = used_after_both | blocks[a][2]
                                for b in candidates[:15]:
                                    if b == a:
                                        continue
                                    if (used3 & blocks[b][2]) == 0:
                                        new_packing = [x for x in packing if x != c1 and x != c2]
                                        new_packing.append(a)
                                        new_packing.append(b)
                                        new_len = len(new_packing)
                                        if new_len > cur_len:
                                            packing = new_packing
                                            used = best_used = used3 | blocks[b][2]
                                            cur_len = new_len
                                            best_packing = list(packing)
                                            best_len = new_len
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                        
                        elif len(candidates) == 1:
                            a = candidates[0]
                            new_packing = [x for x in packing if x != c1 and x != c2]
                            new_packing.append(a)
                            new_len = len(new_packing)
                            if new_len > cur_len:
                                packing = new_packing
                                used = best_used = used_after_both | blocks[a][2]
                                cur_len = new_len
                                best_packing = list(packing)
                                best_len = new_len
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # Try triple swap: remove three, add two (net -1, but may enable better future)
                if len(packing) >= 3:
                    for c1_idx in range(len(packing)):
                        if improved:
                            break
                        c1 = packing[c1_idx]
                        for c2_idx in range(len(packing)):
                            if improved or c2_idx == c1_idx:
                                continue
                            c2 = packing[c2_idx]
                            for c3_idx in range(len(packing)):
                                if improved or c3_idx == c1_idx or c3_idx == c2_idx:
                                    continue
                                c3 = packing[c3_idx]
                                used_after_3 = used & ~blocks[c1][2] & ~blocks[c2][2] & ~blocks[c3][2]
                                
                                candidates = []
                                for a in range(B):
                                    if a not in packing and (used_after_3 & blocks[a][2]) == 0:
                                        candidates.append(a)
                                
                                if len(candidates) >= 2:
                                    for a in candidates[:10]:
                                        if improved:
                                            break
                                        used_a = used_after_3 | blocks[a][2]
                                        for b in candidates[:10]:
                                            if b == a:
                                                continue
                                            if (used_a & blocks[b][2]) == 0:
                                                new_packing = [x for x in packing if x != c1 and x != c2 and x != c3]
                                                new_packing.append(a)
                                                new_packing.append(b)
                                                new_len = len(new_packing)
                                                if new_len > cur_len:
                                                    packing = new_packing
                                                    used = best_used = used_a | blocks[b][2]
                                                    cur_len = new_len
                                                    best_packing = list(packing)
                                                    best_len = new_len
                                                    improved = True
                                                    break
                    if improved:
                        temperature *= cooling
                        continue
            
            if not improved:
                temperature *= cooling
        
        # Final fill pass: try to add any remaining compatible blocks
        final_used = 0
        for b in packing:
            final_used |= blocks[b][2]
        changed = True
        while changed:
            changed = False
            for a in range(B):
                if a not in packing and (final_used & blocks[a][2]) == 0:
                    packing.append(a)
                    final_used |= blocks[a][2]
                    changed = True
        
        if len(packing) > best_len:
            best_packing = list(packing)
            best_len = len(packing)
        
        return best_packing

    best = []
    best_len = 0
    seen_packing = set()

    rng = random.Random(1234567)
    
    # Phase 1: Greedy with multiple random orderings
    for trial in range(8000):
        if time.time() > deadline - 20.0:
            break
        
        order = list(range(B))
        rng.shuffle(order)
        cand = greedy(order)
        
        key = frozenset(cand)
        if key in seen_packing:
            continue
        seen_packing.add(key)
        
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            if best_len == 13:
                break
        
        # Local search on the greedy result
        if len(cand) >= best_len - 1:  # only do local search if competitive
            cand2 = local_search_enhanced(list(cand))
            key2 = frozenset(cand2)
            if key2 not in seen_packing:
                seen_packing.add(key2)
                if len(cand2) > best_len:
                    best = cand2
                    best_len = len(cand2)
                    if best_len == 13:
                        break
    
    # Phase 2: Further refinement from best packing
    if best_len < 13 and time.time() < deadline - 15.0:
        for trial in range(1200):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb the best packing and re-optimize
            if len(best) > 2:
                perturbed = list(best)
                # Remove 1-2 random blocks
                num_remove = min(2, len(perturbed))
                for _ in range(num_remove):
                    if perturbed:
                        idx = rng.randint(0, len(perturbed) - 1)
                        perturbed.pop(idx)
                
                # Re-greedy fill
                used = 0
                for b in perturbed:
                    used |= blocks[b][2]
                
                order = list(range(B))
                rng.shuffle(order)
                for b in order:
                    if (used & blocks[b][2]) == 0:
                        perturbed.append(b)
                        used |= blocks[b][2]
                
                key = frozenset(perturbed)
                if key not in seen_packing:
                    seen_packing.add(key)
                
                cand2 = local_search_enhanced(perturbed)
                key2 = frozenset(cand2)
                if key2 not in seen_packing:
                    seen_packing.add(key2)
                    if len(cand2) > best_len:
                        best = cand2
                        best_len = len(cand2)
                        if best_len == 13:
                            break
    
    # Build the matrices from the best packing
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bidx in best:
        i1, i2, j1, j2, k1, k2 = blocks[bidx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for t in blocks[bidx][1]:
            covered.add(t)

    # Schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
