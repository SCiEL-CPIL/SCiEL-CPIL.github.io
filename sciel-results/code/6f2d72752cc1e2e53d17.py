import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s

    total_points = q ** n          # 6561
    ball_size = 1 + n * (q - 1)    # 17
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        neighbors[idx, 0] = idx
        nb = 1
        for pos in range(n):
            orig = word[pos]
            pw = powers[pos]
            # +1
            nd = (orig + 1) % q
            delta = -2 * pw if nd == 0 else pw
            neighbors[idx, nb] = idx + delta
            nb += 1
            # -1
            nd = (orig - 1) % q
            delta = 2 * pw if nd == 2 else -pw
            neighbors[idx, nb] = idx + delta
            nb += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        return counts

    def reduce_code(code, counts):
        current_code = list(code)
        current_counts = counts.copy()
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                if np.all(current_counts[neighbors[c]] > 1):
                    current_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    # --- Fast greedy initialization with potential tracking ---
    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        order = rng.permutation(total_points)
        while True:
            if time.time() > deadline - 40.0:
                break
            maxp = potential.max()
            if maxp <= 0:
                break
            cands = np.where(potential == maxp)[0]
            if len(cands) == 1:
                c = int(cands[0])
            else:
                c = int(cands[rng.integers(len(cands))])
            selected.append(c)
            potential[c] = 0
            for x in neighbors[c]:
                if not is_covered[x]:
                    is_covered[x] = True
                    potential[neighbors[x]] -= 1
                    np.maximum(potential, 0, out=potential)
            if is_covered.all():
                break
        return selected

    # --- Structured initialization using tetracode product ---
    def structured_init():
        # Tetracode: 9 words of length 4
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        # Convert tetracode words to indices
        tet_indices = []
        for w in tetracode:
            idx = 0
            for i in range(4):
                idx += w[i] * powers[i]
            tet_indices.append(idx)
        
        # Product construction: for each pair of tetracode words, form length-8 word
        code_indices = []
        for i in range(9):
            for j in range(9):
                # First 4 symbols from tet_indices[i], last 4 from tet_indices[j]
                idx1 = tet_indices[i]
                idx2 = tet_indices[j]
                # Convert to full 8-index
                # idx1 occupies positions 0-3, idx2 occupies positions 4-7
                full_idx = idx1 + idx2 * (3 ** 4)
                code_indices.append(full_idx)
        
        return code_indices

    # --- Local search with swap moves ---
    def local_search(code, counts, deadline):
        code = list(code)
        counts = counts.copy()
        code_set = set(code)
        best_len = len(code)
        best_code = code[:]
        rng = np.random.default_rng(42)
        no_improve = 0
        max_no_improve = 15000

        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # Pass 1: try pure removals (fast)
            for _ in range(50):
                if not code:
                    break
                if time.time() > deadline:
                    break
                c = code[rng.integers(len(code))]
                risky = neighbors[c][counts[neighbors[c]] == 1]
                if len(risky) == 0:
                    counts[neighbors[c]] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    improved = True
                    break

            # Pass 2: targeted swap for a codeword that has risky points
            if not improved and code:
                # pick a codeword with fewest risky points (easier to cover)
                sample_size = min(30, len(code))
                sample = rng.choice(len(code), size=sample_size, replace=False)
                best_c = None
                best_risky = None
                best_risky_count = float('inf')
                
                for s in sample:
                    c = code[s]
                    risky = neighbors[c][counts[neighbors[c]] == 1]
                    if len(risky) < best_risky_count:
                        best_risky_count = len(risky)
                        best_c = c
                        best_risky = risky
                
                if best_risky is not None and 0 < len(best_risky) <= 4:
                    # Find candidate that covers all risky points
                    if len(best_risky) == 1:
                        cand = neighbors[best_risky[0]]
                    else:
                        common = set(neighbors[best_risky[0]].tolist())
                        for r in best_risky[1:]:
                            common &= set(neighbors[r].tolist())
                            if not common:
                                break
                        cand = np.array(sorted(common), dtype=np.int32) if common else np.array([], dtype=np.int32)
                    
                    if len(cand) > 0:
                        best_new = None
                        best_score = -1
                        if len(cand) > 80:
                            cand = cand[rng.choice(len(cand), size=80, replace=False)]
                        
                        for np_c in cand:
                            if np_c in code_set:
                                continue
                            new_ball = neighbors[np_c]
                            # Prefer candidate that covers more count==1 points (robustness)
                            cnt1 = np.sum(counts[new_ball] == 1)
                            if cnt1 > best_score:
                                best_score = cnt1
                                best_new = int(np_c)
                        
                        if best_new is not None:
                            counts[neighbors[best_c]] -= 1
                            counts[neighbors[best_new]] += 1
                            code.remove(best_c)
                            code.append(best_new)
                            code_set.discard(best_c)
                            code_set.add(best_new)
                            improved = True

            if improved:
                no_improve = 0
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
            else:
                no_improve += 1
                # Periodic full reduce to escape local minima
                if no_improve % 40 == 0:
                    new_code, new_counts = reduce_code(code, counts)
                    if len(new_code) < len(code):
                        code = new_code
                        counts = new_counts
                        code_set = set(code)
                        no_improve = 0
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                
                # Occasionally try a random perturbation to escape
                if no_improve % 100 == 0 and len(code) > 10:
                    # Remove 2 random words and add 2 random new words
                    if len(code) >= 2:
                        to_remove = rng.choice(len(code), size=2, replace=False)
                        to_remove = sorted(to_remove, reverse=True)
                        removed = []
                        for idx_rm in to_remove:
                            c = code[idx_rm]
                            counts[neighbors[c]] -= 1
                            code.remove(c)
                            code_set.discard(c)
                            removed.append(c)
                        
                        # Add 2 random new words
                        for _ in range(2):
                            new_c = int(rng.integers(total_points))
                            while new_c in code_set:
                                new_c = int(rng.integers(total_points))
                            code.append(new_c)
                            code_set.add(new_c)
                            counts[neighbors[new_c]] += 1
                        
                        # Reduce
                        code, counts = reduce_code(code, counts)
                        code_set = set(code)
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                        no_improve = 0

        return best_code, get_counts(best_code)

    # --- Main ---
    best_code = None
    best_len = float('inf')
    best_counts = None

    # Try structured initialization first
    if time.time() < deadline - 50.0:
        struct_code = structured_init()
        struct_counts = get_counts(struct_code)
        struct_code, struct_counts = reduce_code(struct_code, struct_counts)
        if len(struct_code) < best_len:
            best_len = len(struct_code)
            best_code = struct_code
            best_counts = struct_counts

    # Multiple greedy seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 50.0:
            break
        sel = fast_greedy(seed)
        counts = get_counts(sel)
        sel, counts = reduce_code(sel, counts)
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel
            best_counts = counts

    if best_code is None:
        sel = fast_greedy(1)
        best_code = sel
        best_counts = get_counts(sel)

    # Local search from best
    best_code, best_counts = local_search(best_code, best_counts, deadline)

    # Final reduce
    best_code, best_counts = reduce_code(best_code, best_counts)

    # Convert to words
    result = [idx_to_word(int(idx)) for idx in best_code]
    return result
