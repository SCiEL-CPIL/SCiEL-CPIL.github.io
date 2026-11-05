import time
import numpy as np

# ---------------------------------------------------------------------------
# Ternary covering code K_3(7,1): minimum hitting set over 2187 radius-1 balls
# ---------------------------------------------------------------------------

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 110  # safety margin before the 150s kill

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)

    return _greedy_set_cover(q, n, radius, deadline)


def _solve_ternary_7_1(deadline):
    q = 3
    n = 7
    S = 2187                       # 3**7
    ball = 15                      # 1 + n*(q-1)

    powers = [3 ** i for i in range(n)]

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w

    def word_to_idx(w):
        v = 0
        for i in range(n):
            v += w[i] * powers[i]
        return v

    # --- precompute radius-1 balls (as lists and bitsets) ------------------
    ball_list = [[] for _ in range(S)]
    ball_bit = [0] * S
    for c in range(S):
        w = idx_to_word(c)
        nb = [c]
        for pos in range(n):
            for d in (1, 2):
                w2 = list(w)
                w2[pos] = (w2[pos] + d) % 3
                nb.append(word_to_idx(w2))
        ball_list[c] = nb
        bits = 0
        for e in nb:
            bits |= 1 << e
        ball_bit[c] = bits

    # reverse incidence: elem -> list of balls covering it
    elem_cws = [[] for _ in range(S)]
    for c in range(S):
        for e in ball_list[c]:
            elem_cws[e].append(c)

    full_mask = (1 << S) - 1

    # --- greedy set cover (bitset) -----------------------------------------
    uncovered = full_mask
    code = []
    while uncovered:
        if time.time() > deadline - 60:
            break
        # for each uncovered point, find the ball covering the most uncovered
        gains = [0] * S
        u = uncovered
        while u:
            lsb = u & -u
            e = lsb.bit_length() - 1
            u ^= lsb
            for c in elem_cws[e]:
                gains[c] += 1
        best = 0
        bg = gains[0]
        for c in range(1, S):
            g = gains[c]
            if g > bg:
                bg = g
                best = c
        if bg == 0:
            break
        code.append(best)
        uncovered &= ~ball_bit[best]

    best_code = list(code)
    best_size = len(code)

    # --- bitset local search: drop / 1-1 swap / 2-1 merge ------------------
    def coverage_count(code):
        cnt = [0] * S
        for c in code:
            for e in ball_list[c]:
                cnt[e] += 1
        return cnt

    def is_covered(code):
        cov = 0
        for c in code:
            cov |= ball_bit[c]
        return cov == full_mask

    def local_search(code):
        nonlocal best_code, best_size
        code = list(code)
        cnt = coverage_count(code)
        cset = set(code)
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            # 1) drop redundant
            for c in list(code):
                if time.time() > deadline - 20:
                    break
                if all(cnt[e] > 1 for e in ball_list[c]):
                    code.remove(c)
                    cset.discard(c)
                    for e in ball_list[c]:
                        cnt[e] -= 1
                    improved = True
            # 2) 1-1 swap
            for c in list(code):
                if time.time() > deadline - 20:
                    break
                crit = [e for e in ball_list[c] if cnt[e] == 1]
                if not crit:
                    continue
                fc = crit[0]
                best_w = -1
                best_sc = -1
                for w in elem_cws[fc]:
                    if w in cset:
                        continue
                    wb = ball_bit[w]
                    if all((wb >> e) & 1 for e in crit):
                        sc = sum(1 for e in ball_list[w] if cnt[e] == 1)
                        if sc > best_sc:
                            best_sc = sc
                            best_w = w
                if best_w != -1 and best_sc > 0:
                    code.remove(c)
                    cset.discard(c)
                    for e in ball_list[c]:
                        cnt[e] -= 1
                    code.append(best_w)
                    cset.add(best_w)
                    for e in ball_list[best_w]:
                        cnt[e] += 1
                    improved = True
                    break
            # 3) 2-1 merge
            if len(code) >= 2:
                found = False
                for i in range(len(code)):
                    if time.time() > deadline - 20:
                        break
                    c1 = code[i]
                    for j in range(i + 1, len(code)):
                        crit1 = [e for e in ball_list[c1] if cnt[e] == 1]
                        crit2 = [e for e in ball_list[c2 := code[j]] if cnt[e] == 1]
                        if not crit1 or not crit2:
                            continue
                        union = crit1 + crit2
                        fc = union[0]
                        best_w = -1
                        best_sc = -1
                        for w in elem_cws[fc]:
                            if w in cset:
                                continue
                            wb = ball_bit[w]
                            if all((wb >> e) & 1 for e in union):
                                sc = sum(1 for e in ball_list[w] if cnt[e] <= 1)
                                if sc > best_sc:
                                    best_sc = sc
                                    best_w = w
                        if best_w != -1 and best_sc >= 2:
                            code.remove(c1)
                            code.remove(code[j])
                            cset.discard(c1)
                            cset.discard(code[j])
                            for e in ball_list[c1]:
                                cnt[e] -= 1
                            for e in ball_list[code[j]]:
                                cnt[e] -= 1
                            code.append(best_w)
                            cset.add(best_w)
                            for e in ball_list[best_w]:
                                cnt[e] += 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
            if len(code) < best_size and is_covered(code):
                best_size = len(code)
                best_code = list(code)
        return code

    local_search(code)

    # --- iterative-deepening branch & bound (bitset) -----------------------
    def bnb(uncovered, depth, last_idx, target_k):
        if time.time() > deadline - 2:
            return None
        if uncovered == 0:
            return []
        if depth >= target_k:
            return None
        remaining = uncovered.bit_count()
        if depth + (remaining + ball - 1) // ball > target_k:
            return None
        # pick uncovered element with fewest available covering candidates
        best_e = -1
        min_count = 10 ** 9
        u = uncovered
        while u:
            lsb = u & -u
            e = lsb.bit_length() - 1
            u ^= lsb
            cnt = 0
            for c in elem_cws[e]:
                if c > last_idx:
                    cnt += 1
            if cnt == 0:
                return None
            if cnt < min_count:
                min_count = cnt
                best_e = e
                if min_count == 1:
                    break
        if best_e == -1:
            return None
        cands = [c for c in elem_cws[best_e] if c > last_idx]
        cands.sort(key=lambda c: (uncovered & ball_bit[c]).bit_count(), reverse=True)
        for c in cands:
            res = bnb(uncovered & ~ball_bit[c], depth + 1, c, target_k)
            if res is not None:
                return [c] + res
        return None

    # try to prove a smaller solution exists, working downward from best_size
    for target_k in range(best_size - 1, max(150, best_size - 12), -1):
        if time.time() > deadline - 4:
            break
        sol = bnb(full_mask, 0, -1, target_k)
        if sol is not None:
            if is_covered(sol) and len(sol) == target_k:
                best_size = target_k
                best_code = list(sol)
                # keep trying lower
                continue
            else:
                break

    # final polish on the best code
    polished = local_search(list(best_code))
    if len(polished) < best_size and is_covered(polished):
        best_code = polished
        best_size = len(polished)

    result = [idx_to_word(idx) for idx in best_code]

    # safety verification; fall back to greedy if something broke
    cov = 0
    for idx in best_code:
        cov |= ball_bit[idx]
    if cov != full_mask:
        result = [idx_to_word(idx) for idx in code]
    return result


def _greedy_set_cover(q, n, radius, deadline):
    space_size = q ** n
    if space_size > 50000:
        import random
        random.seed(42)
        words = []
        seen = set()
        while len(words) < 200 and time.time() < deadline - 5:
            w = tuple(random.randint(0, q - 1) for _ in range(n))
            if w not in seen:
                seen.add(w)
                words.append(list(w))
        return words

    powers = [q ** i for i in range(n)]

    def word_to_index(word):
        v = 0
        for i in range(n):
            v += word[i] * powers[i]
        return v

    def index_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    neighborhoods = []
    for idx in range(space_size):
        w = index_to_word(idx)
        neigh = [idx]
        for pos in range(n):
            for new_val in range(q):
                if new_val != w[pos]:
                    w2 = list(w)
                    w2[pos] = new_val
                    neigh.append(word_to_index(w2))
        neighborhoods.append(neigh)

    uncovered = set(range(space_size))
    selected = []
    while time.time() < deadline - 3 and uncovered:
        gains = {}
        for e in uncovered:
            for c in neighborhoods[e]:
                gains[c] = gains.get(c, 0) + 1
        best_cw = max(gains, key=lambda c: gains[c])
        if gains[best_cw] == 0:
            break
        selected.append(index_to_word(best_cw))
        for e in neighborhoods[best_cw]:
            uncovered.discard(e)
    return selected
