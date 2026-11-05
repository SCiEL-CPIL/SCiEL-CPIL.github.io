import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0

    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n
        ball_size = 1 + n * (q - 1)

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

        # Precompute balls
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
        
        # Convert to numpy array for vectorized operations
        balls_np = np.array(balls, dtype=np.int32)
        
        # Precompute ball masks (integers)
        ball_masks = [0] * N
        for i in range(N):
            mask = 0
            for w in balls[i]:
                mask |= (1 << w)
            ball_masks[i] = mask

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        def greedy_cover(seed, start_code=None):
            rng = random.Random(seed)
            covered_mask = 0
            cnt = np.array([ball_size] * N, dtype=np.int32)
            code = []
            uncovered_count = N

            heap = [(-ball_size, i) for i in range(N)]
            heapq.heapify(heap)

            if start_code:
                for c in start_code:
                    code.append(c)
                    ball_c = balls[c]
                    for w in ball_c:
                        if not (covered_mask & (1 << w)):
                            covered_mask |= (1 << w)
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                                if cnt[c2] > 0:
                                    heapq.heappush(heap, (-cnt[c2], c2))

            while uncovered_count > 0:
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val:
                        break
                else:
                    break

                c = i
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not (covered_mask & (1 << w)):
                        covered_mask |= (1 << w)
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            if not code:
                return code
            
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[balls_np[c]] += 1
            
            code_list = list(code)
            changed = True
            while changed:
                changed = False
                # Shuffle to avoid bias? Or just iterate.
                for i in range(len(code_list)):
                    c = code_list[i]
                    # Check if any word in balls[c] has count 1
                    # If min(cover_count[balls_np[c]]) == 1, it's not safe to remove?
                    # No, if count is 1, it means ONLY c covers it. So we cannot remove c.
                    # If min > 1, we can remove.
                    if np.min(cover_count[balls_np[c]]) > 1:
                        cover_count[balls_np[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[balls_np[c]] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

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
                    # Check if min count in ball is > 1
                    if np.min(cover_count[balls_np[c_remove]]) > 1:
                        cover_count[balls_np[c_remove]] -= 1
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break
                    else:
                        # Try 1-swap
                        # Find words with count 1 in ball[c_remove]
                        hole_words = balls_np[c_remove][cover_count[balls_np[c_remove]] == 1]
                        if len(hole_words) <= 8:
                            hole_mask = 0
                            for w in hole_words:
                                hole_mask |= (1 << w)
                            
                            # Find candidate
                            # Intersect cover_map of hole words
                            cand_pool = None
                            for w in hole_words:
                                candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                                if cand_pool is None:
                                    cand_pool = set(candidates_w)
                                else:
                                    cand_pool.intersection_update(candidates_w)
                                if not cand_pool:
                                    break
                            
                            found = -1
                            if cand_pool:
                                for c_add in cand_pool:
                                    if (ball_masks[c_add] & hole_mask) == hole_mask:
                                        found = c_add
                                        break
                            
                            if found >= 0:
                                cover_count[balls_np[c_remove]] -= 1
                                cover_count[balls_np[found]] += 1
                                code_list[i] = found
                                in_code[c_remove] = False
                                in_code[found] = True
                                improved = True
                                break

                # Try 2-swap
                if time.time() < time_limit and len(code_list) >= 3:
                    n_codes = len(code_list)
                    # Randomize order of pairs to avoid bias?
                    # Just iterate
                    for i2 in range(n_codes):
                        if time.time() > time_limit:
                            break
                        for j2 in range(i2 + 1, n_codes):
                            if time.time() > time_limit:
                                break
                            c_rem1 = code_list[i2]
                            c_rem2 = code_list[j2]
                            
                            # Tentative removal
                            # We need to check if any word in B1 U B2 becomes 0.
                            # Words in B1 \ B2: count decreases by 1.
                            # Words in B2 \ B1: count decreases by 1.
                            # Words in B1 \cap B2: count decreases by 2.
                            
                            # Efficient check:
                            # new_count = cover_count - indicator(B1) - indicator(B2)
                            # We only care about words in B1 U B2.
                            
                            # Create a mask of B1 U B2?
                            # Or just check min count in B1 U B2 after decrement?
                            # But we don't want to actually modify cover_count if it fails.
                            
                            # Let's compute the new counts for the union.
                            # union_words = np.unique(np.concatenate([balls_np[c_rem1], balls_np[c_rem2]]))
                            # This is expensive.
                            
                            # Alternative:
                            # 1. Decrement cover_count for B1 and B2.
                            # 2. Check min in B1 U B2.
                            # 3. If min < 0 (or 0 if we only check uncovered), revert.
                            
                            # Wait, if a word is in B1 and B2, it gets -2.
                            # If it was 1, it becomes -1 (invalid).
                            # If it was 2, it becomes 0 (uncovered).
                            # So we need min >= 1? No, we need min >= 1 for "covered by at least one".
                            # Actually, we need the new count to be >= 1 for all words in B1 U B2?
                            # No, words outside B1 U B2 are unaffected.
                            # Words in B1 U B2 must have new count >= 1.
                            
                            # So:
                            cover_count[balls_np[c_rem1]] -= 1
                            cover_count[balls_np[c_rem2]] -= 1
                            
                            # Check min in B1 U B2
                            # We can just check min in B1 and min in B2?
                            # If min(B1) >= 1 and min(B2) >= 1, is it sufficient?
                            # Yes, because any word in B1 is checked by min(B1), and any word in B2 by min(B2).
                            # The union is covered if both parts are covered.
                            
                            min1 = np.min(cover_count[balls_np[c_rem1]])
                            min2 = np.min(cover_count[balls_np[c_rem2]])
                            
                            if min1 >= 1 and min2 >= 1:
                                # Valid 2-swap? No, this is just deletion of 2.
                                # We want to ADD one to replace them.
                                # So we have a "hole" of words that are now count 0?
                                # No, if min >= 1, no hole. Then it's a valid 2-deletion.
                                # But we want to swap: remove 2, add 1.
                                # So we need to find words that became 0 (uncovered) and see if one codeword covers them all.
                                
                                # If min1 >= 1 and min2 >= 1, then we can just delete both?
                                # That would be a 2-deletion.
                                # But we want to maintain size or reduce it.
                                # If we can delete 2 without adding, that's better.
                                # But usually we can't.
                                
                                # If min1 == 0 or min2 == 0, we have a hole.
                                # Let's collect the hole words.
                                uncov1 = balls_np[c_rem1][cover_count[balls_np[c_rem1]] == 0]
                                uncov2 = balls_np[c_rem2][cover_count[balls_np[c_rem2]] == 0]
                                uncov = np.unique(np.concatenate([uncov1, uncov2]))
                                
                                if len(uncov) <= 10:
                                    hole_mask = 0
                                    for w in uncov:
                                        hole_mask |= (1 << w)
                                    
                                    # Find candidate c_add
                                    # Must cover all in uncov.
                                    # And c_add must not be in code (or be one of the removed? No, we removed them).
                                    # c_add should be different from c_rem1, c_rem2.
                                    
                                    cand_pool = None
                                    for w in uncov:
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
                                        # Apply swap
                                        cover_count[balls_np[found2]] += 1
                                        # c_rem1 and c_rem2 are already decremented
                                        code_list[j2] = found2
                                        in_code[c_rem2] = False
                                        in_code[found2] = True
                                        code_list.pop(i2)
                                        in_code[c_rem1] = False
                                        improved = True
                                        break
                                    else:
                                        # Revert
                                        cover_count[balls_np[c_rem1]] += 1
                                        cover_count[balls_np[c_rem2]] += 1
                                else:
                                    # Revert
                                    cover_count[balls_np[c_rem1]] += 1
                                    cover_count[balls_np[c_rem2]] += 1
                            else:
                                # Revert
                                cover_count[balls_np[c_rem1]] += 1
                                cover_count[balls_np[c_rem2]] += 1
                            
                            if improved:
                                break
                        if improved:
                            break

            return code_list

        best_code = None
        best_len = float('inf')

        num_restarts = 60
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
            ls_time = min(deadline - 8, time.time() + 4.0)
            ls_code = local_search(refined, i * 7 + 1, ls_time)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Kick / re-greedy restarts
        if best_code is not None:
            kick_sizes = [3, 4, 5, 6, 7, 4, 5, 6, 3, 5, 4, 6]
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
                ls_code = local_search(re_refined, 3000 + kick_iter, min(deadline - 8, time.time() + 3.5))
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
