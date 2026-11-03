import numpy as np
import itertools
import time
import random

Q = 3
N = 6
NUM_WORDS = Q ** N

all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute ball masks (boolean) and ball index lists
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

# Also precompute pairwise intersection masks? Too large (729x729x729). 
# Instead, compute on the fly using boolean AND.

def get_tetracode_code():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        if q is None or n is None or radius is None:
            return []
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
    deadline = start_time + 115

    best_code = get_tetracode_code()
    best_size = len(best_code)

    def greedy_cover():
        uncovered = np.ones(NUM_WORDS, dtype=np.int32)
        code = []
        for _ in range(NUM_WORDS):
            if uncovered.sum() == 0:
                break
            scores = np.array([uncovered[ball_masks[i]].sum() for i in range(NUM_WORDS)])
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            uncovered[ball_masks[c]] = 0
        return code

    def greedy_randomized():
        uncovered = np.ones(NUM_WORDS, dtype=np.int32)
        code = []
        for _ in range(NUM_WORDS):
            if uncovered.sum() == 0:
                break
            scores = np.array([uncovered[ball_masks[i]].sum() for i in range(NUM_WORDS)])
            max_s = scores.max()
            if max_s == 0:
                break
            # Pick randomly among top candidates
            top = np.where(scores >= max_s * 0.8)[0]
            c = top[np.random.randint(len(top))]
            code.append(c)
            uncovered[ball_masks[c]] = 0
        return code

    def local_search(code, max_iters=20000):
        """Correct local search with remove, add, and swap moves."""
        code = list(code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        def energy():
            unc = (cov_counts == 0).sum()
            return 1000.0 * unc + 1.0 * len(code)
        
        best_code = list(code)
        best_e = energy()
        
        T = 20.0
        cool = (0.1 / 20.0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            mt = np.random.randint(3)
            
            if mt == 0 and len(code) > 0:
                # Remove
                idx = np.random.randint(len(code))
                r = code[idx]
                # Words that would lose coverage
                would_lose = (cov_counts[ball_masks[r]] == 1).sum()
                delta = 1000.0 * would_lose - 1.0
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    code.pop(idx)
                    cov_counts[ball_masks[r]] -= 1
                    e = energy()
                    if e < best_e:
                        best_e = e
                        best_code = list(code)
            
            elif mt == 1:
                # Add
                a = np.random.randint(NUM_WORDS)
                if a in code:
                    continue
                new_cov = (cov_counts[ball_masks[a]] == 0).sum()
                delta = -1000.0 * new_cov + 1.0
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    code.append(a)
                    cov_counts[ball_masks[a]] += 1
                    e = energy()
                    if e < best_e:
                        best_e = e
                        best_code = list(code)
            
            elif mt == 2 and len(code) > 0:
                # Swap: remove r, add a
                idx = np.random.randint(len(code))
                r = code[idx]
                a = np.random.randint(NUM_WORDS)
                if a == r or a in code:
                    continue
                
                # Correct delta computation
                # unique_to_r: in ball[r] but not in ball[a]
                # unique_to_a: in ball[a] but not in ball[r]
                # Words in unique_to_r with cov_count==1 -> become uncovered
                # Words in unique_to_a with cov_count==0 -> become covered
                unique_r = ball_masks[r] & ~ball_masks[a]
                unique_a = ball_masks[a] & ~ball_masks[r]
                
                L = (cov_counts[unique_r] == 1).sum()  # newly uncovered
                Nv = (cov_counts[unique_a] == 0).sum()  # newly covered
                
                delta = 1000.0 * L - 1000.0 * Nv
                
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    code[idx] = a
                    cov_counts[ball_masks[r]] -= 1
                    cov_counts[ball_masks[a]] += 1
                    e = energy()
                    if e < best_e:
                        best_e = e
                        best_code = list(code)
            
            T *= cool
        
        return best_code

    def reduction(code):
        """Remove redundant codewords."""
        code = list(code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_masks[c]] > 1):
                    cov_counts[ball_masks[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
        return code

    # Phase 1: Greedy initializations
    inits = []
    for i in range(15):
        if time.time() > deadline:
            break
        if i < 5:
            sol = greedy_cover()
        else:
            sol = greedy_randomized()
        sol = reduction(sol)
        inits.append(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]
    
    # Phase 2: Local search from best inits
    inits.sort(key=len)
    for sol in inits[:8]:
        if time.time() > deadline:
            break
        for trial in range(3):
            if time.time() > deadline:
                break
            res = local_search(sol, max_iters=15000)
            res = reduction(res)
            if len(res) < best_size:
                best_size = len(res)
                best_code = [list(all_words[c]) for c in res]
    
    return best_code
