import numpy as np
import itertools
import time
import random

Q = 3
N = 6
NUM_WORDS = Q ** N

# Precompute all words and ball membership
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute ball masks: ball_masks[i] is a boolean vector of length NUM_WORDS
ball_masks = []
ball_idx_lists = []
for i in range(NUM_WORDS):
    c = all_words[i]
    mask = np.zeros(NUM_WORDS, dtype=bool)
    mask[i] = True
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            nb = c.copy()
            nb[pos] = val
            idx = word_to_idx[tuple(nb)]
            mask[idx] = True
    ball_masks.append(mask)
    ball_idx_lists.append(np.where(mask)[0])

# Convert to a numpy array for faster operations
ball_masks_arr = np.array(ball_masks, dtype=bool)  # shape (NUM_WORDS, NUM_WORDS)


def get_tetracode_code():
    """Generate the direct sum of the tetracode with length-2 ternary code.
    Tetracode: [a, b, a+b, a+2b] for a,b in {0,1,2} -> 9 codewords of length 4.
    Direct sum with all 9 length-2 words -> 81 codewords of length 6.
    """
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code


def reduction(code):
    """Remove redundant codewords from the code."""
    code = list(code)
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_masks[c]] += 1

    changed = True
    while changed:
        changed = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        for idx in indices:
            c = code[idx]
            if np.all(cov_counts[ball_masks[c]] > 1):
                cov_counts[ball_masks[c]] -= 1
                code.pop(idx)
                changed = True
                break
    return code


def greedy_cover(seed=None):
    """Greedy set cover: repeatedly pick the codeword covering the most uncovered words."""
    if seed is not None:
        random.seed(seed)
    uncovered = np.ones(NUM_WORDS, dtype=np.int32)
    code = []
    for _ in range(NUM_WORDS):
        if uncovered.sum() == 0:
            break
        # Compute coverage score for each candidate
        scores = ball_masks_arr.dot(uncovered)
        c = int(np.argmax(scores))
        if scores[c] == 0:
            break
        code.append(c)
        uncovered[ball_masks[c]] = 0
    return code


def greedy_randomized(seed=None):
    """Randomized greedy: pick from top candidates."""
    if seed is not None:
        random.seed(seed)
    uncovered = np.ones(NUM_WORDS, dtype=np.int32)
    code = []
    for _ in range(NUM_WORDS):
        if uncovered.sum() == 0:
            break
        scores = ball_masks_arr.dot(uncovered)
        max_s = scores.max()
        if max_s == 0:
            break
        # Pick randomly among top 80% candidates
        top = np.where(scores >= max_s * 0.7)[0]
        c = top[random.randint(len(top))]
        code.append(c)
        uncovered[ball_masks[c]] = 0
    return code


def local_search(code, deadline, max_iters=30000):
    """Simulated annealing local search with remove, add, and swap moves."""
    code = list(code)
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_masks[c]] += 1

    def energy():
        unc = (cov_counts == 0).sum()
        return 1000.0 * unc + 1.0 * len(code)

    best_code = list(code)
    best_e = energy()

    T = 30.0
    cool = (0.05 / 30.0) ** (1.0 / max_iters)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        mt = random.randint(3)

        if mt == 0 and len(code) > 0:
            # Remove
            idx = random.randint(len(code))
            r = code[idx]
            would_lose = (cov_counts[ball_masks[r]] == 1).sum()
            delta = 1000.0 * would_lose - 1.0
            if delta < 0 or random.random() < np.exp(-delta / T):
                code.pop(idx)
                cov_counts[ball_masks[r]] -= 1
                e = energy()
                if e < best_e:
                    best_e = e
                    best_code = list(code)

        elif mt == 1:
            # Add
            a = random.randint(NUM_WORDS)
            if a in code:
                continue
            new_cov = (cov_counts[ball_masks[a]] == 0).sum()
            delta = -1000.0 * new_cov + 1.0
            if delta < 0 or random.random() < np.exp(-delta / T):
                code.append(a)
                cov_counts[ball_masks[a]] += 1
                e = energy()
                if e < best_e:
                    best_e = e
                    best_code = list(code)

        elif mt == 2 and len(code) > 0:
            # Swap: remove r, add a
            idx = random.randint(len(code))
            r = code[idx]
            a = random.randint(NUM_WORDS)
            if a == r or a in code:
                continue

            unique_r = ball_masks[r] & ~ball_masks[a]
            unique_a = ball_masks[a] & ~ball_masks[r]

            L = (cov_counts[unique_r] == 1).sum()
            Nv = (cov_counts[unique_a] == 0).sum()

            delta = 1000.0 * L - 1000.0 * Nv

            if delta < 0 or random.random() < np.exp(-delta / T):
                code[idx] = a
                cov_counts[ball_masks[r]] -= 1
                cov_counts[ball_masks[a]] += 1
                e = energy()
                if e < best_e:
                    best_e = e
                    best_code = list(code)

        T *= cool

    return best_code


def swap_local_search(code, deadline, max_iters=20000):
    """Focused swap-based local search: only swap moves for faster convergence."""
    code = list(code)
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cov_counts[ball_masks[c]] += 1

    def energy():
        unc = (cov_counts == 0).sum()
        return 1000.0 * unc + 1.0 * len(code)

    best_code = list(code)
    best_e = energy()

    T = 15.0
    cool = (0.02 / 15.0) ** (1.0 / max_iters)

    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Try a swap
        idx = random.randint(len(code))
        r = code[idx]
        a = random.randint(NUM_WORDS)
        if a == r or a in code:
            continue

        unique_r = ball_masks[r] & ~ball_masks[a]
        unique_a = ball_masks[a] & ~ball_masks[r]

        L = (cov_counts[unique_r] == 1).sum()
        Nv = (cov_counts[unique_a] == 0).sum()

        delta = 1000.0 * L - 1000.0 * Nv

        if delta < 0 or random.random() < np.exp(-delta / T):
            code[idx] = a
            cov_counts[ball_masks[r]] -= 1
            cov_counts[ball_masks[a]] += 1
            e = energy()
            if e < best_e:
                best_e = e
                best_code = list(code)

        T *= cool

    return best_code


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        if q is None or n is None or radius is None:
            return []
        # Generic fallback: greedy set cover
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_code = get_tetracode_code()
    best_size = len(best_code)  # 81

    # Phase 1: Generate diverse initial solutions
    inits = []
    for i in range(20):
        if time.time() > deadline - 60:
            break
        if i < 6:
            sol = greedy_cover(seed=i * 42 + 7)
        else:
            sol = greedy_randomized(seed=i * 13 + 3)
        sol = reduction(sol)
        inits.append(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]

    # Sort by size
    inits.sort(key=len)

    # Phase 2: Local search from best initial solutions
    # Use multiple local search variants and more trials
    for sol in inits[:10]:
        if time.time() > deadline - 15:
            break
        for trial in range(4):
            if time.time() > deadline - 10:
                break
            if trial % 2 == 0:
                res = local_search(sol, deadline, max_iters=20000)
            else:
                res = swap_local_search(sol, deadline, max_iters=15000)
            res = reduction(res)
            if len(res) < best_size:
                best_size = len(res)
                best_code = [list(all_words[c]) for c in res]

    # Phase 3: If we have time, do additional perturbation + repair
    if time.time() < deadline - 20 and best_size > 73:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution
            perturbed = list(best_code_idx) if 'best_code_idx' in dir() else list(inits[0])
            # Remove 3-5 random codewords
            n_remove = random.randint(3, 5)
            indices_to_remove = random.sample(range(len(perturbed)), min(n_remove, len(perturbed)))
            for idx in sorted(indices_to_remove, reverse=True):
                perturbed.pop(idx)
            # Repair with greedy
            cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
            for c in perturbed:
                cov_counts[ball_masks[c]] += 1
            uncovered = (cov_counts == 0)
            while uncovered.any():
                scores = ball_masks_arr.dot(uncovered.astype(np.int32))
                c = int(np.argmax(scores))
                if scores[c] == 0:
                    break
                perturbed.append(c)
                cov_counts[ball_masks[c]] += 1
                uncovered = (cov_counts == 0)
            perturbed = reduction(perturbed)
            # Local search
            res = local_search(perturbed, deadline, max_iters=15000)
            res = reduction(res)
            if len(res) < best_size:
                best_size = len(res)
                best_code = [list(all_words[c]) for c in res]

    return best_code
