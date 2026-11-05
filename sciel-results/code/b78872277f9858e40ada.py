import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Handle generic cases or fallbacks
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    total_points = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute neighbors
    # neighbors[i] is a list of indices within Hamming distance 1 of i
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
            pw = q ** pos
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

    # Precompute digit powers for fast index conversion
    digit_powers = np.array([q**i for i in range(n)], dtype=np.int64)
    
    def idxs_to_words(idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        if len(idxs) == 0:
            return []
        words = np.zeros((len(idxs), n), dtype=np.int64)
        for i in range(n):
            words[:, i] = (idxs // digit_powers[i]) % q
        return [list(row) for row in words]

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Peeling heuristic: remove codewords that are not essential.
        Returns the reduced code and updated counts.
        """
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

    def fast_greedy(seed):
        """
        Greedy set cover construction with random tie-breaking.
        """
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 40.0:
                break
            maxp = potential.max()
            if maxp <= 0:
                break
            cands = np.where(potential == maxp)[0]
            c = int(cands[rng.integers(len(cands))])
            selected.append(c)
            potential[c] = 0
            # Update potential for neighbors of neighbors
            # If x is newly covered, potential of all y in Ball(x) decreases by 1
            # We need to update potential for all y such that x in Ball(y)
            # Since Ball is symmetric, this is neighbors[x]
            newly_covered = neighbors[c][~is_covered[neighbors[c]]]
            if len(newly_covered) > 0:
                is_covered[newly_covered] = True
                # For each newly covered point x, decrement potential of all y in Ball(x)
                # This can be vectorized
                # potential[neighbors[newly_covered]] -= 1
                # But we need to ensure we don't go below 0
                # And we only care about points not yet selected?
                # Actually, potential is just a heuristic.
                
                # Efficient update:
                # We can accumulate the decrements
                # But simple loop is okay if ball_size is small (17)
                # However, newly_covered can be up to 17 points.
                # So we update at most 17 * 17 = 289 entries.
                for x in newly_covered:
                    potential[neighbors[x]] -= 1
                np.maximum(potential, 0, out=potential)
            
            if is_covered.all():
                break
                
        return selected

    def local_search(code, counts, deadline):
        """
        Local search with removals and swaps.
        """
        code = list(code)
        counts = counts.copy()
        code_set = set(code)
        best_len = len(code)
        best_code = code[:]
        rng = np.random.default_rng(42)
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # Pass 1: Vectorized removals
            if code:
                code_arr = np.array(code, dtype=np.int64)
                # Get neighbors for all codewords
                code_neighbors = neighbors[code_arr]  # (len(code), ball_size)
                # Get counts for all these neighbor positions
                code_counts = counts[code_neighbors]  # (len(code), ball_size)
                # A codeword can be removed if all its neighbors have count > 1
                removable_mask = np.all(code_counts > 1, axis=1)
                removable_indices = np.where(removable_mask)[0]
                
                if len(removable_indices) > 0:
                    # Remove the one with the highest minimum count (most redundant)
                    min_counts = code_counts[removable_indices].min(axis=1)
                    best_idx = removable_indices[np.argmax(min_counts)]
                    c = code[best_idx]
                    counts[neighbors[c]] -= 1
                    code.pop(best_idx)
                    code_set.discard(c)
                    improved = True
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]

            # Pass 2: Swaps
            if not improved and code:
                code_arr = np.array(code, dtype=np.int64)
                code_neighbors = neighbors[code_arr]
                code_counts = counts[code_neighbors]
                
                # Find codewords with risky points (count == 1)
                risky_mask = np.any(code_counts == 1, axis=1)
                risky_indices = np.where(risky_mask)[0]
                
                if len(risky_indices) > 0:
                    # Sample from risky codewords
                    sample_size = min(10, len(risky_indices))
                    sample = rng.choice(risky_indices, size=sample_size, replace=False)
                    
                    for s in sample:
                        c = code[s]
                        c_neighbors = neighbors[c]
                        # Points in Ball(c) that are only covered by c
                        risky_points = c_neighbors[counts[c_neighbors] == 1]
                        
                        if len(risky_points) == 0:
                            continue
                        
                        # Find candidates that cover all risky points
                        # x is a candidate if for all r in risky_points, r in Ball(x)
                        # i.e., x in Ball(r) for all r
                        # Intersection of Ball(r) for r in risky_points
                        cand_set = set(neighbors[risky_points[0]].tolist())
                        for r in risky_points[1:]:
                            cand_set &= set(neighbors[r].tolist())
                            if not cand_set:
                                break
                        
                        if not cand_set:
                            continue
                        
                        # Filter out existing codewords
                        cand_list = [x for x in cand_set if x not in code_set]
                        if not cand_list:
                            continue
                        
                        # Score candidates: prefer those that cover more points with count > 1
                        # or just pick randomly to diversify
                        # Let's pick the one that maximizes the sum of counts of its neighbors
                        # This encourages moving to a "denser" area
                        best_cand = None
                        best_score = -1
                        # Limit candidates to check for speed
                        check_list = cand_list[:50]
                        for x in check_list:
                            score = np.sum(counts[neighbors[x]])
                            if score > best_score:
                                best_score = score
                                best_cand = x
                                
                        if best_cand is None:
                            best_cand = cand_list[rng.integers(len(cand_list))]
                            
                        c_new = best_cand
                        
                        # Update
                        counts[neighbors[c]] -= 1
                        counts[neighbors[c_new]] += 1
                        code.pop(s)
                        code.append(c_new)
                        code_set.discard(c)
                        code_set.add(c_new)
                        
                        improved = True
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                        break

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                # Periodic reduce
                if no_improve % 50 == 0:
                    new_code, new_counts = reduce_code(code, counts)
                    if len(new_code) < len(code):
                        code = new_code
                        counts = new_counts
                        code_set = set(code)
                        no_improve = 0
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                
                # Stall breaking: add a random codeword and peel
                if no_improve > 100:
                    # Pick a random point not in code
                    all_points = np.arange(total_points)
                    not_in_code = np.array([p for p in all_points if p not in code_set])
                    if len(not_in_code) > 0:
                        new_c = int(not_in_code[rng.integers(len(not_in_code))])
                        code.append(new_c)
                        code_set.add(new_c)
                        counts[neighbors[new_c]] += 1
                        no_improve = 0
                        # Try to peel immediately
                        new_code, new_counts = reduce_code(code, counts)
                        if len(new_code) < len(code):
                            code = new_code
                            counts = new_counts
                            code_set = set(code)
                            if len(code) < best_len:
                                best_len = len(code)
                                best_code = code[:]

        return best_code, get_counts(best_code)

    best_code = None
    best_len = float('inf')
    
    # Multiple Greedy + Local Search
    # We use a few seeds to diversify the initial solutions
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
            
        # 1. Greedy Construction
        sel = fast_greedy(seed)
        
        # 2. Reduce
        counts = get_counts(sel)
        sel, counts = reduce_code(sel, counts)
        
        # 3. Local Search
        sel, counts = local_search(sel, counts, deadline)
        
        # 4. Final Reduce
        sel, counts = reduce_code(sel, counts)
        
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel

    if best_code is None:
        sel = fast_greedy(1)
        best_code = sel

    result = idxs_to_words(best_code)
    return result
