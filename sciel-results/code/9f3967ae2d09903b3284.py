"""Ternary covering code K_3(8,1) - optimized heuristic search."""
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    deadline = time.time() + 140.0
    rng = random.Random(12345)
    
    total = q ** n  # 6561
    nbr_count = 1 + n * (q - 1)  # 17 for q=3, n=8, radius=1
    
    # Precompute neighbors
    neighbors = np.zeros((total, nbr_count), dtype=np.int32)
    for idx in range(total):
        w = [0] * n
        tmp = idx
        for i in range(n - 1, -1, -1):
            w[i] = tmp % q
            tmp //= q
        
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                flat = 0
                for d in w:
                    flat = flat * q + d
                nbrs.append(flat)
            w[pos] = orig
        neighbors[idx] = nbrs
    
    # Coverage matrix
    cov = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        cov[i, neighbors[i]] = 1
    
    def flat_to_word(idx):
        w = []
        tmp = idx
        for i in range(n - 1, -1, -1):
            w.append(tmp % q)
            tmp //= q
        return w
    
    def is_fully_covered(code_list):
        if len(code_list) == 0:
            return False
        covered = np.zeros(total, dtype=bool)
        for idx in code_list:
            covered |= (cov[idx] == 1)
        return covered.all()
    
    def local_search(code, deadline, rng):
        code = list(code)
        code_set = set(code)
        
        def get_uncovered_after_removal(idx):
            covered = np.zeros(total, dtype=bool)
            for j in code:
                if j == idx:
                    continue
                covered |= (cov[j] == 1)
            uncovered = (cov[idx] == 1) & ~covered
            return uncovered
        
        # Redundancy removal
        for pass_num in range(20):
            if time.time() > deadline - 20.0:
                return code
            improved = False
            order = list(range(len(code)))
            rng.shuffle(order)
            for i in order:
                if i >= len(code):
                    continue
                if time.time() > deadline - 20.0:
                    return code
                uncovered = get_uncovered_after_removal(code[i])
                if not uncovered.any():
                    code.pop(i)
                    improved = True
            if not improved:
                break
        
        # Swap optimization
        for iteration in range(20):
            if time.time() > deadline - 20.0:
                return code
            improved = False
            order = list(range(len(code)))
            rng.shuffle(order)
            for i in order:
                if i >= len(code):
                    continue
                if time.time() > deadline - 20.0:
                    return code
                uncovered = get_uncovered_after_removal(code[i])
                if not uncovered.any():
                    code.pop(i)
                    improved = True
                    continue
                uncount = int(uncovered.sum())
                candidates = set()
                for u in np.where(uncovered)[0]:
                    for nb in neighbors[u]:
                        candidates.add(int(nb))
                candidates -= code_set
                best_repl = -1
                best_gain = -1
                for j in candidates:
                    gain = int((cov[j] == 1)[uncovered].sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = j
                    if best_gain >= uncount:
                        break
                if best_repl != -1 and best_gain >= uncount and best_repl != code[i]:
                    code_set.discard(code[i])
                    code[i] = best_repl
                    code_set.add(best_repl)
                    improved = True
            if not improved:
                break
        
        # Final redundancy
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 15.0:
                break
            uncovered = get_uncovered_after_removal(code[i])
            if not uncovered.any():
                code.pop(i)
        
        return code
    
    # Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code_init = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            flat = 0
            for d in w:
                flat = flat * q + d
            code_init.append(flat)
    
    best_code = local_search(code_init, deadline, rng)
    best_size = len(best_code)
    
    # Try greedy if time permits
    if time.time() < deadline - 40.0:
        uncovered = np.ones(total, dtype=bool)
        code_greedy = []
        while uncovered.any():
            if time.time() > deadline - 40.0:
                break
            gains = (cov & uncovered[np.newaxis, :]).sum(axis=1)
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            code_greedy.append(best_idx)
            uncovered[cov[best_idx] == 1] = False
        
        if len(code_greedy) < best_size:
            code_greedy = local_search(code_greedy, deadline, rng)
            if len(code_greedy) < best_size:
                best_code = code_greedy
                best_size = len(best_code)
    
    # Verify
    if not is_fully_covered(best_code):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [flat_to_word(idx) for idx in best_code]
