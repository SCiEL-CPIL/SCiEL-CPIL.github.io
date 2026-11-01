import time
import math
import random
from itertools import permutations

def construct(n):
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s wall-clock

    def classic(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                o = 0
                for cand in range(min(len(new), len(block)), 0, -1):
                    if new[len(new) - cand:] == block[:cand]:
                        o = cand
                        break
                new += block[o:]
            word = new
        return word

    def verify(word, n):
        seen = set()
        for i in range(len(word) - n + 1):
            w = word[i:i + n]
            if len(set(w)) != n:
                return False
            seen.add(tuple(w))
        return len(seen) == math.factorial(n)

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123132123"

    base = list(range(1, n))
    perms = [list(p) for p in permutations(base)]
    K = len(perms)
    L = n - 1

    # Precompute overlap table: ov[i][j] = longest suffix of perms[i] == prefix of perms[j]
    ov = [bytearray(K) for _ in range(K)]
    for i in range(K):
        pi = perms[i]
        row = ov[i]
        for j in range(K):
            if i == j:
                continue
            pj = perms[j]
            o = 0
            for cand in range(L - 1, 0, -1):
                if pi[-cand:] == pj[:cand]:
                    o = cand
                    break
            row[j] = o

    # Convert ov to list of lists for faster indexing in tight loops
    # Actually bytearray is fine; keep as is.

    def nn_greedy(start):
        """Nearest-neighbor construction from a given start."""
        order = [start]
        vis = bytearray(K)
        vis[start] = 1
        cur = start
        score = 0
        for _ in range(K - 1):
            row = ov[cur]
            bo = -1
            bo2 = -1
            for j in range(K):
                if not vis[j]:
                    o = row[j]
                    if o > bo2:
                        bo2 = o
                        bo = j
            order.append(bo)
            vis[bo] = 1
            score += bo2
            cur = bo
        return order, score

    def random_order():
        """Random permutation as starting order."""
        order = list(range(K))
        random.shuffle(order)
        # Compute score
        score = 0
        for i in range(K - 1):
            score += ov[order[i]][order[i + 1]]
        return order, score

    # Multiple diverse starts: greedy from different starts + random
    best_order = None
    best_score = -1

    # Greedy from first min(K, 12) starts
    for s in range(min(K, 12)):
        order, score = nn_greedy(s)
        if score > best_score:
            best_score = score
            best_order = order

    # Add a few random starts
    for _ in range(min(5, max(0, (DEADLINE - (time.time() - t0)) // 2))):
        if time.time() > DEADLINE - 30:
            break
        order, score = random_order()
        if score > best_score:
            best_score = score
            best_order = order

    order = best_order[:]
    cur_score = best_score

    # Efficient local search using precomputed adjacency scores
    # adj_score[i] = ov[order[i-1]][order[i]] + ov[order[i]][order[i+1]] for internal nodes
    # We'll recompute as needed.

    def compute_adj_scores(ord_):
        """Return list of (left_ov, right_ov) for each position."""
        scores = [0] * K
        for i in range(K):
            left = ov[ord_[i - 1]][ord_[i]] if i > 0 else 0
            right = ov[ord_[i]][ord_[i + 1]] if i < K - 1 else 0
            scores[i] = left + right
        return scores

    def try_improve_sa(temp=0.0):
        """2-opt + Or-opt with optional SA acceptance. Returns (improved, new_score)."""
        nonlocal cur_score
        # 2-opt: try all reversals, take best single improvement
        best_gain = 0
        best_i = -1
        best_j = -1
        for i in range(1, K - 1):
            a = order[i - 1]
            oi = order[i]
            for j in range(i + 1, K):
                oj = order[j]
                old = ov[a][oi]
                if j + 1 < K:
                    old += ov[oj][order[j + 1]]
                new = ov[a][oj]
                if j + 1 < K:
                    new += ov[oi][order[j + 1]]
                gain = new - old
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        if best_i >= 0:
            order[best_i:best_j + 1] = order[best_i:best_j + 1][::-1]
            cur_score += best_gain
            return True

        # Or-opt: move a single element to best position
        best_gain = 0
        best_i = -1
        best_j = -1  # insert after position j (i.e., between j and j+1)
        for i in range(1, K - 1):
            x = order[i]
            left = order[i - 1]
            right = order[i + 1]
            base_old = ov[left][x] + ov[x][right]
            base_new = ov[left][right]
            for j in range(0, K - 1):
                if j == i or j == i - 1:
                    continue
                ins_old = ov[order[j]][order[j + 1]]
                ins_new = ov[order[j]][x] + ov[x][order[j + 1]]
                gain = (base_new + ins_new) - (base_old + ins_old)
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        if best_i >= 0:
            x = order.pop(best_i)
            jj = best_j if best_j < best_i else best_j - 1
            order.insert(jj + 1, x)
            cur_score += best_gain
            return True

        # 3-opt: reverse segments of length 3,4,5
        for seg_len in (3, 4, 5):
            best_gain = 0
            best_i = -1
            for i in range(1, K - seg_len + 1):
                j = i + seg_len - 1
                a = order[i - 1]
                old = ov[a][order[i]]
                if j + 1 < K:
                    old += ov[order[j]][order[j + 1]]
                new = ov[a][order[j]]
                if j + 1 < K:
                    new += ov[order[i]][order[j + 1]]
                gain = new - old
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i >= 0:
                j = best_i + seg_len - 1
                order[best_i:j + 1] = order[best_i:j + 1][::-1]
                cur_score += best_gain
                return True

        return False

    # Main local search loop with restarts
    best_found_order = order[:]
    best_found_score = cur_score

    while time.time() < DEADLINE - 25:
        # Run local search to convergence
        while time.time() < DEADLINE - 25:
            if not try_improve_sa():
                break

        # Update best
        if cur_score > best_found_score:
            best_found_score = cur_score
            best_found_order = order[:]

        # Check if we should restart
        if time.time() > DEADLINE - 30:
            break

        # Try a restart: perturb the current best or start fresh
        if random.random() < 0.3:
            # Perturb: do a few random 2-opt moves
            order = best_found_order[:]
            cur_score = best_found_score
            for _ in range(3):
                i = random.randint(1, K - 2)
                j = random.randint(i + 1, K - 1)
                a = order[i - 1]
                old = ov[a][order[i]]
                if j + 1 < K:
                    old += ov[order[j]][order[j + 1]]
                new = ov[a][order[j]]
                if j + 1 < K:
                    new += ov[order[i]][order[j + 1]]
                order[i:j + 1] = order[i:j + 1][::-1]
                cur_score += (new - old)
        else:
            # Fresh random start
            order, cur_score = random_order()

    # Use the best found
    order = best_found_order
    cur_score = best_found_score

    # Build the word
    word = []
    prev = None
    for idx in order:
        p = perms[idx]
        block = p + [n] + p
        if prev is None:
            word += block
        else:
            word += block[ov[prev][idx]:]
        prev = idx

    if not verify(word, n):
        word = classic(n)

    return "".join(map(str, word))
