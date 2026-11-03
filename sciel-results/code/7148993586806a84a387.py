import time
import random
import math
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.monotonic()
    deadline = start_time + 115.0
    
    total = q ** n  # 729
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: covers[c] = set of word indices within distance 1 of c
    covers = [None] * total
    for i in range(total):
        w = all_words[i]
        cov = {i}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                cov.add(word_to_idx[tuple(nw)])
        covers[i] = cov
    
    # Precompute distance-2 neighbors
    dist2 = [set() for _ in range(total)]
    for i in range(total):
        w1 = all_words[i]
        for j in range(i + 1, total):
            w2 = all_words[j]
            d = sum(1 for k in range(n) if w1[k] != w2[k])
            if d == 2:
                dist2[i].add(j)
                dist2[j].add(i)
    
    # Precompute covered_by: for each word, which codewords cover it
    covered_by = [set() for _ in range(total)]
    for c in range(total):
        for w in covers[c]:
            covered_by[w].add(c)
    
    # Build distance-2 graph adjacency matrix for spectral analysis
    adj = np.zeros((total, total), dtype=np.float64)
    for i in range(total):
        for j in dist2[i]:
            adj[i, j] = 1.0
    
    # Compute Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    
    # Compute eigenvectors of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Spectral centrality: use the second smallest eigenvalue's eigenvector
    # (Fiedler vector) and higher modes for diversity
    # Score = weighted sum of absolute eigenvalues * eigenvector components
    spectral_scores = np.zeros(total)
    # Use top few non-trivial eigenvectors
    for k in range(1, min(6, total)):
        spectral_scores += abs(eigenvalues[k]) * eigenvectors[:, k] ** 2
    
    # Normalize
    spectral_scores = spectral_scores / (spectral_scores.max() + 1e-10)
    
    def greedy_init_with_spectral(seed, use_spectral=True):
        """Greedy set cover with optional spectral-guided initialization."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        code_set = set()
        
        if use_spectral and len(spectral_scores) > 0:
            # Initialize with top spectral vertices (diverse starting point)
            # Take top ~20% of spectral scores as initial candidates
            sorted_by_spectral = np.argsort(-spectral_scores)
            n_init = max(10, total // 5)  # ~146 vertices
            for idx in sorted_by_spectral[:n_init]:
                idx = int(idx)
                if idx not in code_set:
                    code.append(idx)
                    code_set.add(idx)
                    for w in covers[idx]:
                        counts[w] += 1
        
        # Greedy fill: add codewords that cover most uncovered words
        while True:
            if time.monotonic() > deadline - 5:
                break
            uncovered_mask = (counts == 0).astype(np.int32)
            if uncovered_mask.sum() == 0:
                break
            
            # Compute gain for each candidate
            best_gain = 0
            best_c = -1
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for w in covers[c]:
                    if counts[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_gain <= 0 or best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            for w in covers[best_c]:
                counts[w] += 1
        
        return code, counts, code_set
    
    def pure_greedy_init(seed):
        """Standard greedy set cover."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        code_set = set()
        
        while True:
            if time.monotonic() > deadline - 5:
                break
            uncovered_mask = (counts == 0).astype(np.int32)
            if uncovered_mask.sum() == 0:
                break
            
            best_gain = 0
            best_c = -1
            for c in range(total):
                if c in code_set:
                    continue
                gain = sum(1 for w in covers[c] if counts[w] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_gain <= 0 or best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            for w in covers[best_c]:
                counts[w] += 1
        
        return code, counts, code_set
    
    def polish(code, counts, code_set, time_budget):
        """Local search with 2-for-1 moves, swaps, and removals."""
        end_time = time.monotonic() + time_budget
        best_code = code[:]
        best_size = len(code)
        rng = random.Random(int(time.monotonic() * 1000) % 100000)
        
        def is_redundant(idx):
            for w in covers[idx]:
                if counts[w] == 1:
                    return False
            return True
        
        def remove_codeword(idx):
            code.remove(idx)
            code_set.discard(idx)
            for w in covers[idx]:
                counts[w] -= 1
        
        def add_codeword(idx):
            code.append(idx)
            code_set.add(idx)
            for w in covers[idx]:
                counts[w] += 1
        
        # Phase 1: Remove all redundant codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for idx in list(code):
                if is_redundant(idx):
                    remove_codeword(idx)
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # Phase 2: Simulated annealing with 2-for-1, swap, and add moves
        temp = 5.0
        cool_rate = 0.9995
        min_temp = 0.1
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.25:
                # Try 2-for-1: find c1, c2 in code such that some c3 covers union
                if len(code) >= 2:
                    c1, c2 = rng.sample(code, 2)
                    union_cov = covers[c1] | covers[c2]
                    # Find a word in union_cov to restrict c3 search
                    if union_cov:
                        sample_word = rng.choice(list(union_cov))
                        candidate_c3s = covered_by[sample_word]
                        found_c3 = -1
                        for c3 in candidate_c3s:
                            if c3 in code_set:
                                continue
                            if union_cov.issubset(covers[c3]):
                                found_c3 = c3
                                break
                        if found_c3 != -1:
                            remove_codeword(c1)
                            remove_codeword(c2)
                            add_codeword(found_c3)
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                        
            elif move_type < 0.60:
                # Try swap: replace one codeword with another
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    
                    # Choose new candidate from distance-2 neighbors or random
                    if rng.random() < 0.6 and dist2[old_c]:
                        new_c = rng.choice(list(dist2[old_c]))
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check if swap is safe: no word becomes uncovered
                    # Words uniquely covered by old_c and not by new_c
                    unique_old = 0
                    for w in covers[old_c]:
                        if counts[w] == 1 and w not in covers[new_c]:
                            unique_old += 1
                    
                    if unique_old == 0:
                        # Safe swap
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        for w in covers[old_c]:
                            counts[w] -= 1
                        for w in covers[new_c]:
                            counts[w] += 1
                    elif unique_old <= 3 and rng.random() < math.exp(-unique_old / temp):
                        # Risky swap with repair
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        for w in covers[old_c]:
                            counts[w] -= 1
                        for w in covers[new_c]:
                            counts[w] += 1
                        
                        # Repair uncovered words
                        for w in range(total):
                            if counts[w] == 0:
                                # Find a codeword that covers w
                                for nc in covered_by[w]:
                                    if nc not in code_set:
                                        add_codeword(nc)
                                        break
            
            elif move_type < 0.85:
                # Try remove (if redundant)
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_redundant(c):
                        remove_codeword(c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
            
            else:
                # Try add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set and rng.random() < math.exp(-1.5 / temp):
                    add_codeword(new_c)
            
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for idx in list(code):
                if is_redundant(idx):
                    remove_codeword(idx)
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size
    
    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    num_restarts = 15
    time_per_restart = (deadline - time.monotonic() - 5) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10:
            break
        
        # Alternate between spectral and pure greedy initialization
        use_spectral = (restart % 2 == 0)
        seed = 42 + restart * 13
        
        if use_spectral:
            code, counts, code_set = greedy_init_with_spectral(seed, True)
        else:
            code, counts, code_set = pure_greedy_init(seed)
        
        if not code:
            continue
        
        polished_code, polished_size = polish(code, counts, code_set, time_per_restart - 0.5)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
        
        if overall_best_size <= 73:
            break
    
    if overall_best_code is None:
        code, counts, code_set = pure_greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result


def _generic_construct(q, n, radius):
    """Generic fallback for other parameters."""
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage
    covers = [None] * total
    for i in range(total):
        w = all_words[i]
        cov = {i}
        # Generate all words within Hamming distance 'radius'
        for d in range(1, radius + 1):
            for positions in itertools.combinations(range(n), d):
                for new_vals in itertools.product(range(q), repeat=d):
                    nw = list(w)
                    for pos, val in zip(positions, new_vals):
                        nw[pos] = val
                    nw = tuple(nw)
                    if nw != w:
                        cov.add(word_to_idx[nw])
        covers[i] = cov
    
    # Greedy set cover
    counts = np.zeros(total, dtype=np.int32)
    code = []
    code_set = set()
    
    while True:
        uncovered = sum(1 for c in counts if c == 0)
        if uncovered == 0:
            break
        
        best_gain = 0
        best_c = -1
        for c in range(total):
            if c in code_set:
                continue
            gain = sum(1 for w in covers[c] if counts[w] == 0)
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_gain <= 0:
            break
        
        code.append(best_c)
        code_set.add(best_c)
        for w in covers[best_c]:
            counts[w] += 1
    
    result = []
    for idx in code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
