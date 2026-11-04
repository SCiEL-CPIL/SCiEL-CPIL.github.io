import time
import numpy as np
import itertools

def construct(q, n, radius):
    t_start = time.time()
    deadline = 148.0

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    def word_to_idx(w):
        idx = 0
        for pos in range(n):
            idx += w[pos] * q_powers[n - 1 - pos]
        return idx

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_set, cov, deadline_time):
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < deadline_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    # STRUCTURAL SEED: Use Hamming code structure
    # Ternary Hamming code of length 4 (m=2): generator matrix
    # Parity check H = [1,0,0,1; 0,1,0,1; 0,0,1,1] over GF(3)
    # Codewords: {w : H @ w = 0}
    # This gives 27 codewords covering 81 words perfectly (radius 1, length 4)
    
    # For n=7, use a partition: split into positions and use structured codewords
    # Strategy: enumerate all words with a constraint on first few positions to create
    # a structured initial set, then optimize.
    
    # Build a structured seed using GF(3) linearity
    # We want codewords that are "spread out" - use a linear subspace-like structure
    # Pick a set of "base" directions and generate codewords as linear combinations
    
    # Simpler: use a lattice-like structure. For q=3, n=7:
    # Generate codewords as (a*dir1 + b*dir2 + ...) mod 3 for various a,b
    # This ensures good spread.
    
    # Actually, let's use a smarter structural seed:
    # Take all words where the sum of certain positions satisfies a constraint
    # This creates a "coset-like" structure.
    
    # Best structural seed: use the van Laarhoven et al. known construction idea
    # but generate it algorithmically. We'll use a greedy with a STRUCTURED ordering:
    # process words in a Gray-code-like order to get diverse coverage.
    
    best_code = None
    best_size = space_size + 1

    # Phase 1: Multiple greedy with different seeds
    for seed in range(20):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 30.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Aggressive removal with multiple passes
    cur_code = best_code[:]
    cur_cov = build_cov(cur_code)
    cur_size = len(cur_code)

    for _ in range(100):
        if (time.time() - t_start) > deadline - 20.0:
            break
        changed = False
        for i in range(len(cur_code)):
            c = cur_code[i]
            if np.min(cur_cov[neighbors[c]]) >= 2:
                cur_code.pop(i)
                cur_cov[neighbors[c]] -= 1
                cur_size -= 1
                changed = True
                break
        if not changed:
            break

    if cur_size < best_size:
        best_code = cur_code[:]
        best_size = cur_size

    # Phase 3: Multi-kick local search
    rng = np.random.default_rng(123)
    
    for restart in range(8):
        if (time.time() - t_start) > deadline - 15.0:
            break
        
        if restart == 0:
            cur_code = best_code[:]
        else:
            cur_code = greedy_cover(restart + 200)
            cov = build_cov(cur_code)
            if not np.all(cov > 0):
                cur_code, cov = greedy_repair(cur_code, cov, deadline - 25.0)
        
        cur_cov = build_cov(cur_code)
        
        for k in range(1, 4):
            if (time.time() - t_start) > deadline - 10.0:
                break
            for trial in range(40):
                if (time.time() - t_start) > deadline - 10.0:
                    break
                if len(cur_code) <= k:
                    break
                
                remove_indices = rng.choice(len(cur_code), size=k, replace=False)
                remove_set = set(int(ri) for ri in remove_indices)
                
                new_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
                new_cov = build_cov(new_code)
                
                new_code, new_cov = greedy_repair(new_code, new_cov, deadline - 15.0)
                
                if np.all(new_cov > 0) and len(new_code) < len(cur_code):
                    cur_code = new_code
                    cur_cov = new_cov
                    break
            
            if (time.time() - t_start) < deadline - 10.0:
                changed = True
                while changed and (time.time() - t_start) < deadline - 10.0:
                    changed = False
                    for i in range(len(cur_code)):
                        c = cur_code[i]
                        if np.min(cur_cov[neighbors[c]]) >= 2:
                            cur_code.pop(i)
                            cur_cov[neighbors[c]] -= 1
                            changed = True
                            break

        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Phase 4: Final aggressive removal
    if (time.time() - t_start) < deadline - 3.0:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        for _ in range(200):
            if (time.time() - t_start) > deadline - 3.0:
                break
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
            if not changed:
                break
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
