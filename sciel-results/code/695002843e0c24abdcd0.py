import numpy as np
import random
import time
import heapq
from collections import defaultdict

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls as lists and masks
        balls = []
        ball_masks = [0] * N
        ball_np = np.empty((N, ball_size), dtype=np.int32)
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)
            mask = 0
            for w in neighbors:
                mask |= (1 << w)
            ball_masks[i] = mask
            ball_np[i] = neighbors

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute cover_map as sets for fast intersection
        cover_map_sets = [set(lst) for lst in cover_map]

        # Precompute ball overlap counts: overlap[i][j] = |ball[i] ∩ ball[j]|
        # This is used to identify high-overlap pairs for 2-swap filtering
        # For q=3, n=7, radius=1: overlap depends on Hamming distance d between i and j
        # d=0: 15, d=1: 1+6*2=13... actually let's compute via formula
        # Two words at distance d have ball intersection of size:
        # sum over k positions where they differ: if k=0, contributes 1 (the word itself if same)
        # Actually for radius 1 balls in {0,1,2}^n:
        # |B(x) ∩ B(y)| where d(x,y)=d:
        # = 1 (if d=0) + ... let me just compute it numerically for small n
        # For n=7, q=3, r=1:
        # d=0: 15
        # d=1: 1 + 6*2 = 13? No.
        # B(x) contains x and x with one position changed.
        # B(y) contains y and y with one position changed.
        # Intersection: words z with d(z,x)<=1 and d(z,y)<=1.
        # If d(x,y)=0: all 15 words in B(x)
        # If d(x,y)=1: say x and y differ at position p.
        #   z=x: d(z,y)=1 ✓
        #   z=x with pos q≠p changed: d(z,y) = 2 (differs at p and q) ✗
        #   z=y: d(z,x)=1 ✓
        #   z=y with pos q≠p changed: d(z,x) = 2 ✗
        #   z=y with pos p changed to x[p]: z=x, already counted
        #   So intersection = {x, y} plus... wait
        #   Actually z must be within 1 of both x and y.
        #   If d(x,y)=1, the only common neighbors are x and y themselves? 
        #   No: consider z that differs from x at position p (making it y) - that's y.
        #   z that differs from y at position p (making it x) - that's x.
        #   Any other z? z differs from x at q≠p: then d(z,y)≥2. No.
        #   So |B(x)∩B(y)| = 2 when d=1? That seems low. Let me verify with n=2.
        #   n=2, q=3, r=1: B(00) = {00, 01, 02, 10, 20}, size 5
        #   B(10) = {10, 00, 20, 11, 12}
        #   Intersection: {00, 10, 20} = 3. d(00,10)=1.
        #   So for n=2, d=1: intersection = 3 = 1 + 1*2? 
        #   General formula: for d(x,y)=d, |B(x)∩B(y)| = 
        #   For d=0: 1 + n(q-1)
        #   For d=1: 2 + (n-1)(q-1) ... no, from n=2: 3 = 2 + 1*(3-1)? 2+2=4≠3.
        #   Let me think again. n=2, q=3: B(00)={00,01,02,10,20}, B(10)={10,00,20,11,12}
        #   Common: 00, 10, 20. That's 3.
        #   Formula: words z with d(z,00)≤1 and d(z,10)≤1.
        #   z=00: d(00,10)=1 ✓
        #   z=01: d(01,10)=2 ✗
        #   z=02: d(02,10)=2 ✗
        #   z=10: d(10,00)=1 ✓
        #   z=20: d(20,00)=1, d(20,10)=1 ✓
        #   So 3 words. The pattern: for d=1, intersection = 1 (x) + 1 (y) + (n-1) words that differ from x at the differing position to match y's other coords... 
        #   Actually the (n-1) comes from: positions other than the differing one, we can change x's value at that position and still be within 1 of y? No.
        #   Let me just precompute it numerically.
        overlap = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(i+1, N):
                # Compute Hamming distance
                di = int_to_base3(i, n)
                dj = int_to_base3(j, n)
                dist = sum(1 for a, b in zip(di, dj) if a != b)
                # For q=3, n=7, r=1, the overlap depends only on distance
                # d=0: 15, d=1: 3, d=2: 1, d>=3: 0 (for n=7)
                # Let me verify d=2: n=2, q=3: B(00) and B(11)
                # B(00)={00,01,02,10,20}, B(11)={11,01,21,10,12}
                # Common: 01, 10. That's 2. Hmm, not 1.
                # d=2 in n=2: intersection = 2.
                # For n=7, d=2: let's think. x and y differ at positions p and q.
                # z within 1 of both: z=x (d=2 from y ✗), z=y (d=2 from x ✗)
                # z differs from x at p: z has y[p] at p, x elsewhere. d(z,y): differs at q only → d=1 ✓
                # z differs from x at q: z has y[q] at q, x elsewhere. d(z,y): differs at p only → d=1 ✓
                # z differs from x at r≠p,q: d(z,y)≥3 ✗
                # So intersection = 2 for d=2.
                # d=3: z within 1 of both? z=x: d=3✗. Any z within 1 of x changes at most 1 pos, so d(z,y)≥2. 
                #   If z differs from x at p: d(z,y) = d-1 = 2. Still >1. So 0 for d≥3.
                # So for n=7, q=3, r=1: overlap = {0:15, 1:3, 2:2, 3+:0}
                if dist == 0:
                    ov = 15
                elif dist == 1:
                    ov = 3
                elif dist == 2:
                    ov = 2
                else:
                    ov = 0
                overlap[i, j] = ov
                overlap[j, i] = ov

        # Precompute pairwise ball intersection sizes for 2-swap candidate filtering
        # For pairs (i, j) where i < j, store the count of words covered by both
        # We'll compute this lazily or use a vectorized approach
        # Actually, let's precompute for all pairs the intersection mask
        # That's N*(N-1)/2 = ~2.4M pairs, each with a 2187-bit mask - too much memory
        # Instead, for the 2-swap, we'll use a smarter approach:
        # When removing c1 and c2, the hole is (ball[c1] ∪ ball[c2]) \ (covered by others)
        # A candidate c_add must cover all words in the hole.
        # So c_add must be in the intersection of cover_map[w] for all w in the hole.
        # We can precompute, for each pair (c1, c2), the set of words in ball[c1] ∪ ball[c2]
        # that are NOT covered by any other codeword in the current code.
        # But that depends on the current code, so we can't precompute fully.
        
        # Alternative: for 2-swap, only try pairs where the combined ball is "small" 
        # or where the overlap is large (meaning removing both creates a small hole).
        # Precompute pairwise intersection sizes
        pair_inter = {}
        for i in range(N):
            for j in range(i+1, N):
                # Compute intersection size of balls[i] and balls[j]
                # Using the fact that two words in {0,1,2}^7 have Hamming distance d,
                # their balls (radius 1) intersect in 1 + (7-d)*2*(2-d) terms... 
                # Actually, let's just compute it directly for now, but this is O(N^2 * ball_size)
                # 2187^2 * 15 ≈ 72M operations - might be too slow
                pass
        
        # Skip precomputation; use on-the-fly with early termination

        def greedy_cover(seed, start_code=None):
            """Vectorized greedy set cover using numpy for gain computation."""
            rng = random.Random(seed)
            code = []
            uncovered = np.ones(N, dtype=bool)
            uncovered_count = N

            if start_code:
                for c in start_code:
                    code.append(c)
                    ball_c = balls[c]
                    for w in ball_c:
                        if uncovered[w]:
                            uncovered[w] = False
                            uncovered_count -= 1

            # Precompute ball as numpy array for vectorized gain
            # gain[c] = number of uncovered words in ball[c]
            while uncovered_count > 0:
                # Compute gain for all candidates using vectorized approach
                # For each candidate c, gain = sum of uncovered[ball_np[c]]
                # Use broadcasting: for each row of ball_np, sum uncovered at those indices
                gains = np.empty(N, dtype=np.int32)
                for c in range(N):
                    gains[c] = np.sum(uncovered[ball_np[c]])
                
                # Find best candidate with tie-breaking by randomization for diversity
                max_gain = gains.max()
                if max_gain == 0:
                    break
                
                # Get all candidates with max gain
                candidates = np.where(gains == max_gain)[0]
                c = int(candidates[rng.randint(0, len(candidates) - 1)])
                
                code.append(c)
                ball_c = balls[c]
                for w in ball_c:
                    if uncovered[w]:
                        uncovered[w] = False
                        uncovered_count -= 1

            return code

        def refine(code):
            """Remove redundant codewords using bitmasks."""
            if not code:
                return code

            covered_mask = 0
            for c in code:
                covered_mask |= ball_masks[c]

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if c is redundant: its ball is fully covered by others
                    # i.e., (covered_mask ^ ball_masks[c]) covers ball_masks[c]
                    # Actually, simpler: remove c, see if covered_mask still covers ball_masks[c]
                    mask_without_c = covered_mask ^ ball_masks[c] 
                    # Wait, XOR only works if c's bits are a subset.
                    # Better: (covered_mask & ~ball_masks[c]) | ball_masks[c] == covered_mask?
                    # No, just check: does (covered_mask without c) still cover ball_masks[c]?
                    # Let's just recompute or use a count.
                    # With bitmasks, it's hard to "remove" and check quickly without recounting.
                    # Let's stick to the list-based refine for safety, or optimize it.
                    # Actually, we can just check:
                    # Is there any word w in ball_masks[c] that is NOT in (covered_mask ^ ball_masks[c])?
                    # This is getting complicated. Let's use the list-based refine but it's slow.
                    # Alternative: use a count array for refine.
                    pass 

            # Let's implement a fast refine using counts
            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1
            
            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """Local search with deletion, 1-swap, and 2-swap moves using cover_count."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion
                    newly_uncovered = []
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-swap
                    found = -1
                    if len(newly_uncovered) <= 8:
                        hole_mask = 0
                        for w in newly_uncovered:
                            hole_mask |= (1 << w)
                        
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        cand_pool = None
                        for w in crit_sorted:
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if cand_pool is None:
                                cand_pool = set(candidates_w)
                            else:
                                cand_pool.intersection_update(candidates_w)
                            if not cand_pool:
                                break
                        if cand_pool:
                            for c_add in cand_pool:
                                if (ball_masks[c_add] & hole_mask) == hole_mask:
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in balls[found]:
                            cover_count[w] += 1
                        improved = True
                        break
                    else:
                        for w in balls[c_remove]:
                            cover_count[w] += 1

                # Try 2-swap: prioritize high-overlap pairs
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    # First try high-overlap pairs (overlap >= 2), then random
                    high_overlap_pairs = []
                    for a in range(n_codes):
                        for b in range(a+1, n_codes):
                            if overlap[code_list[a], code_list[b]] >= 2:
                                high_overlap_pairs.append((a, b))
                    
                    # Try high-overlap pairs first, then fill with random
                    tried_pairs = set()
                    pairs_to_try = list(high_overlap_pairs)
                    # Add random pairs to fill up to max
                    max_pairs = min(80, n_codes * (n_codes - 1) // 2)
                    while len(pairs_to_try) < max_pairs:
                        a = rng.randint(0, n_codes - 1)
                        b = rng.randint(0, n_codes - 1)
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        if (a, b) in tried_pairs:
                            continue
                        pairs_to_try.append((a, b))
                    
                    for i2, j2 in pairs_to_try[:max_pairs]:
                        if time.time() > time_limit:
                            break
                        if (i2, j2) in tried_pairs:
                            continue
                        tried_pairs.add((i2, j2))
                        
                        c_rem1 = code_list[i2]
                        c_rem2 = code_list[j2]
                        uncov = []
                        for w in balls[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.append(w)
                        for w in balls[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.append(w)
                        
                        if uncov and len(uncov) <= 10:
                            hole_mask = 0
                            for w in uncov:
                                hole_mask |= (1 << w)
                            
                            cand_pool = None
                            for w in sorted(uncov, key=lambda x: len(cover_map[x])):
                                candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_rem1 and c2 != c_rem2]
                                if cand_pool is None:
                                    cand_pool = set(candidates_w)
                                else:
                                    cand_pool.intersection_update(candidates_w)
                                if not cand_pool:
                                    break
                            found2 = -1
                            if cand_pool:
                                for c_add in cand_pool:
                                    if (ball_masks[c_add] & hole_mask) == hole_mask:
                                        found2 = c_add
                                        break
                            if found2 >= 0:
                                code_list[j2] = found2
                                in_code[c_rem2] = False
                                in_code[found2] = True
                                for w in balls[found2]:
                                    cover_count[w] += 1
                                code_list.pop(i2)
                                in_code[c_rem1] = False
                                improved = True
                                break
                            else:
                                for w in balls[c_rem1]:
                                    cover_count[w] += 1
                                for w in balls[c_rem2]:
                                    cover_count[w] += 1
                        else:
                            for w in balls[c_rem1]:
                                cover_count[w] += 1
                            for w in balls[c_rem2]:
                                cover_count[w] += 1
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts
        num_restarts = 100
        for i in range(num_restarts):
            if time.time() > deadline - 15:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 10:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            ls_time = min(deadline - 8, time.time() + 3.5)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6, 5, 4, 6, 7, 3, 5, 4, 6, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 12:
                    break
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Final polish
        if best_code is not None and time.time() < deadline - 5:
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            final_code = refine(best_code)
            for polish_iter in range(3):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 150:
                    break
                ls_code2 = local_search(final_code, 99 + polish_iter * 13, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)
                else:
                    break
        elif best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        best_idx = -1
        max_gain = 0
        for i in range(N):
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if max_gain == 0:
            break
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
