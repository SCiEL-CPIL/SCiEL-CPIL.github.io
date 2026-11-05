import numpy as np
import random
import time
import heapq

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

        # Precompute balls (neighbors including self)
        balls = []
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

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        import heapq

        def greedy_cover(seed, start_code=None):
            """Greedy set cover with heap-based candidate selection."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            # Build heap: (-cnt[i], i) for max-heap behavior
            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)
            
            if start_code:
                for c in start_code:
                    code.append(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], c2))

            while uncovered_count > 0:
                # Pop valid candidates from heap
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not covered[i]:
                        break
                else:
                    break

                c = i
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

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
            """Local search with deletion, 1-swap, and optimized 2-swap moves."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                # Pass 1: Deletion and 1-Swap
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
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
                    if len(newly_uncovered) <= 6:
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        w0 = crit_sorted[0]
                        candidates_w0 = [c2 for c2 in cover_map[w0] if not in_code[c2] and c2 != c_remove]
                        if candidates_w0:
                            for c_add in candidates_w0:
                                bs = ball_sets[c_add]
                                ok = True
                                for w in crit_sorted[1:]:
                                    if w not in bs:
                                        ok = False
                                        break
                                if ok:
                                    found = c_add
                                    break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in ball_indices[found]:
                            cover_count[w] += 1
                        improved = True
                        break
                    else:
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1

                if improved:
                    continue

                # Pass 2: 2-Swap (Remove 2, Add 1)
                # Optimization: Only try if code size is above a threshold
                if time.time() < time_limit and len(code_list) > 160:
                    n_codes = len(code_list)
                    # Sample pairs to save time if code is large
                    max_pairs = 500
                    pairs_tried = 0
                    found_improvement = False
                    
                    # Randomly sample pairs to avoid O(N^2) blowup in tight time
                    while pairs_tried < max_pairs and time.time() < time_limit:
                        i2 = rng.randint(0, n_codes - 1)
                        j2 = rng.randint(0, n_codes - 1)
                        if i2 == j2:
                            continue
                        pairs_tried += 1
                        
                        c_rem1 = code_list[i2]
                        c_rem2 = code_list[j2]
                        
                        # Track uncovered
                        uncov = set()
                        # Decrement counts
                        for w in ball_indices[c_rem1]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.add(w)
                        for w in ball_indices[c_rem2]:
                            cover_count[w] -= 1
                            if cover_count[w] == 0:
                                uncov.add(w)
                                
                        if uncov and len(uncov) <= 10:
                            # Find candidate covering all uncov
                            # Use rarest word
                            uncov_list = sorted(uncov, key=lambda x: len(cover_map[x]))
                            w0 = uncov_list[0]
                            candidates_w0 = [c2 for c2 in cover_map[w0] if not in_code[c2] and c2 != c_rem1 and c2 != c_rem2]
                            found2 = -1
                            if candidates_w0:
                                for c_add in candidates_w0:
                                    bs = ball_sets[c_add]
                                    ok = True
                                    for w in uncov_list[1:]:
                                        if w not in bs:
                                            ok = False
                                            break
                                    if ok:
                                        found2 = c_add
                                        break
                            
                            if found2 >= 0:
                                # Apply swap
                                # Remove c_rem2 (replace with found2)
                                code_list[j2] = found2
                                in_code[c_rem2] = False
                                in_code[found2] = True
                                for w in ball_indices[found2]:
                                    cover_count[w] += 1
                                # Remove c_rem1
                                code_list.pop(i2)
                                if j2 > i2:
                                    j2 -= 1 # Index adjustment, though we break anyway
                                in_code[c_rem1] = False
                                found_improvement = True
                                improved = True
                                break
                            else:
                                # Restore counts
                                for w in ball_indices[c_rem1]:
                                    cover_count[w] += 1
                                for w in ball_indices[c_rem2]:
                                    cover_count[w] += 1
                        else:
                            # Restore counts
                            for w in ball_indices[c_rem1]:
                                cover_count[w] += 1
                            for w in ball_indices[c_rem2]:
                                cover_count[w] += 1
                            
                        if found_improvement:
                            break

            return code_list

        def simulated_annealing(code, seed, time_limit):
            """Simulated Annealing for escaping local minima."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            best_code = list(code_list)
            best_len = len(code_list)
            
            T = 50.0
            T_min = 1.0
            cooling_rate = 0.95
            current_len = len(code_list)
            
            while T > T_min and time.time() < time_limit:
                # Generate a move
                move_type = rng.random()
                if move_type < 0.3:
                    # Delete
                    if len(code_list) <= 1:
                        continue
                    idx = rng.randint(0, len(code_list) - 1)
                    c_remove = code_list[idx]
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)
                    
                    if newly_uncovered:
                        # Restore and reject
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1
                        T *= cooling_rate
                        continue
                        
                    # Accept
                    code_list.pop(idx)
                    in_code[c_remove] = False
                    current_len -= 1
                    
                elif move_type < 0.7:
                    # Swap
                    if len(code_list) <= 1:
                        continue
                    idx = rng.randint(0, len(code_list) - 1)
                    c_remove = code_list[idx]
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)
                    
                    if newly_uncovered:
                        # Try to find a replacement
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        w0 = crit_sorted[0]
                        candidates_w0 = [c2 for c2 in cover_map[w0] if not in_code[c2] and c2 != c_remove]
                        found = -1
                        if candidates_w0:
                            c_add = rng.choice(candidates_w0)
                            bs = ball_sets[c_add]
                            ok = True
                            for w in crit_sorted[1:]:
                                if w not in bs:
                                    ok = False
                                    break
                            if ok:
                                found = c_add
                        
                        if found < 0:
                            # Restore
                            for w in ball_indices[c_remove]:
                                cover_count[w] += 1
                            T *= cooling_rate
                            continue
                        
                        # Accept swap
                        code_list[idx] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in ball_indices[found]:
                            cover_count[w] += 1
                        # current_len unchanged
                    else:
                        # Delete is better, but we are in swap mode? Just delete? 
                        # Or reject swap and let delete handle it? 
                        # Let's just delete if it's safe.
                        code_list.pop(idx)
                        in_code[c_remove] = False
                        current_len -= 1
                        T *= cooling_rate
                        continue

                else:
                    # Random Add (Kick)
                    c_add = rng.randint(0, N - 1)
                    if in_code[c_add]:
                        T *= cooling_rate
                        continue
                    
                    code_list.append(c_add)
                    in_code[c_add] = True
                    for w in ball_indices[c_add]:
                        cover_count[w] += 1
                    current_len += 1

                # Update best
                if current_len < best_len:
                    best_len = current_len
                    best_code = list(code_list)
                
                T *= cooling_rate

            return best_code

        best_code = None
        best_len = float('inf')

        # Phase 1: Multiple Greedy + Local Search restarts
        num_restarts = 50
        for i in range(num_restarts):
            if time.time() > deadline - 30:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if time.time() > deadline - 15:
                if len(refined) < best_len:
                    best_len = len(refined)
                    best_code = refined
                continue
            
            ls_time = min(deadline - 10, time.time() + 2.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Phase 2: Iterated Local Search (ILS) with Kicking
        # This is the key improvement: perturb the best solution and re-optimize
        if best_code is not None:
            # Vary kick strength
            kick_sizes = [4, 6, 8, 10, 12, 5, 7, 9, 11, 15, 20, 10, 8, 6]
            for kick_iter, k in enumerate(kick_sizes):
                if time.time() > deadline - 15:
                    break
                
                rng_k = random.Random(1000 + kick_iter)
                k_actual = min(k, len(best_code))
                
                # Kick: remove k random codewords
                kick_set = set(rng_k.sample(best_code, k_actual))
                remaining = [c for c in best_code if c not in kick_set]
                
                # Re-greedy to fill gap
                re_code = greedy_cover(2000 + kick_iter, start_code=remaining)
                re_refined = refine(re_code)
                
                if len(re_refined) < best_len:
                    best_len = len(re_refined)
                    best_code = re_refined
                
                # Local search on the new candidate
                if time.time() > deadline - 10:
                    break
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 3.0))
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code

        # Phase 3: Simulated Annealing Polish
        # Use SA to escape local minima found by LS
        if best_code is not None and time.time() < deadline - 10:
            sa_code = simulated_annealing(best_code, 42, min(deadline - 5, time.time() + 10.0))
            if len(sa_code) < best_len:
                best_len = len(sa_code)
                best_code = sa_code
            # Refine after SA
            best_code = refine(best_code)

        # Phase 4: Final Polish
        if best_code is not None and time.time() < deadline - 5:
            # Long local search on the very best
            ls_code = local_search(best_code, 42, deadline - 3)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
            
            final_code = refine(best_code)
            
            # Additional short ILS bursts if time allows
            for polish_iter in range(3):
                if time.time() > deadline - 2:
                    break
                if len(final_code) <= 150:
                    break
                
                # Small kick
                rng_p = random.Random(9000 + polish_iter)
                k_small = 3
                kick_set = set(rng_p.sample(final_code, k_small))
                remaining = [c for c in final_code if c not in kick_set]
                re_code = greedy_cover(9100 + polish_iter, start_code=remaining)
                re_refined = refine(re_code)
                
                ls_code2 = local_search(re_refined, 9200 + polish_iter, deadline - 1)
                if len(ls_code2) < len(final_code):
                    final_code = refine(ls_code2)

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
