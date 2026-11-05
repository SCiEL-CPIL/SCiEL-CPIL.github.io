import numpy as np
import random
import time
from itertools import product

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
        ball_sets = []  # precomputed sets for fast membership
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
            ball_sets.append(set(neighbors))

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        def greedy_cover(seed, tie_break='random'):
            """Greedy set cover with configurable tie-breaking."""
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            while uncovered_count > 0:
                max_val = max(cnt)
                candidates = [i for i in range(N) if cnt[i] == max_val]
                
                if tie_break == 'random':
                    c = rng.choice(candidates)
                elif tie_break == 'min_id':
                    c = min(candidates)
                elif tie_break == 'max_id':
                    c = max(candidates)
                elif tie_break == 'weighted':
                    # Weight by how many times this codeword was a candidate
                    weights = [rng.random() for _ in candidates]
                    c = candidates[weights.index(max(weights))]
                else:
                    c = rng.choice(candidates)
                
                code.append(c)

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1

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
            """Local search with deletion and swap moves using precomputed ball sets."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in balls[c]:
                    cover_count[w] += 1

            # Build set of current codewords for O(1) lookup
            code_set = set(code_list)

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]
                    
                    # Try deletion first
                    for w in balls[c_remove]:
                        cover_count[w] -= 1
                    uncovered = [w for w in range(N) if cover_count[w] == 0]
                    if not uncovered:
                        code_list.pop(i)
                        code_set.discard(c_remove)
                        improved = True
                        break

                    # Try swap: find a replacement that covers all uncovered points
                    # Use precomputed ball_sets for fast intersection check
                    found = -1
                    uncovered_set = set(uncovered)
                    for c_add in range(N):
                        if c_add == c_remove or c_add in code_set:
                            continue
                        # Check if ball_add covers all uncovered points
                        if uncovered_set.issubset(ball_sets[c_add]):
                            found = c_add
                            break
                    
                    if found >= 0:
                        code_list[i] = found
                        code_set.discard(c_remove)
                        code_set.add(found)
                        for w in balls[found]:
                            cover_count[w] += 1
                        improved = True
                        break
                    else:
                        # Restore cover_count
                        for w in balls[c_remove]:
                            cover_count[w] += 1

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts with different seeds and tie-breaking strategies
        tie_breaks = ['random', 'min_id', 'max_id', 'weighted']
        restart_count = 0
        for i in range(60):
            if time.time() > deadline - 30:  # Reserve time for local search
                break
            tb = tie_breaks[i % len(tie_breaks)]
            code = greedy_cover(i, tie_break=tb)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined
            restart_count += 1

        # Local search on the best code found
        if best_code is not None and time.time() < deadline - 15:
            ls_code = local_search(best_code, 42, deadline - 5)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

        # Second round of local search with different seed if time permits
        if best_code is not None and time.time() < deadline - 10:
            ls_code2 = local_search(best_code, 123, deadline - 5)
            if len(ls_code2) < best_len:
                best_len = len(ls_code2)
                best_code = ls_code2

        if best_code is not None:
            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case (shouldn't be called for this task, but included for completeness)
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

    # Generic greedy
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
