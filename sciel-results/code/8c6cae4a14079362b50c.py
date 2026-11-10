import numpy as np
import time
import itertools
import random
from collections import deque

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        # Precompute distance-2 neighbors for swap candidates
        # A word d is a valid replacement for c if dist(c,d) <= 2
        # and d covers all essential words of c
        # We'll compute this on-the-fly but cache the neighbor sets
        
        def greedy_init(rng, mode=0):
            """
            mode 0: deterministic greedy (max gain, first candidate)
            mode 1: randomized top-k greedy
            mode 2: random start, then greedy
            """
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            
            if mode == 2:
                # Start with a random word
                w = rng.integers(0, N)
                code.add(w)
                uncovered[neighbors[w, :]] = False
            
            while np.any(uncovered) and time.time() < deadline:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                
                if mode == 1 and len(candidates) > 1:
                    top_k = min(10, len(candidates))
                    w = rng.choice(candidates[:top_k])
                elif mode == 0:
                    w = candidates[0]
                else:
                    w = rng.choice(candidates)
                
                code.add(w)
                uncovered[neighbors[w, :]] = False
            
            return code
        
        def reduce_code(code_set, neighbors, deadline):
            if not code_set:
                return code_set
            
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            
            return code_set
        
        def compute_coverage(code_set, neighbors):
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            return coverage
        
        def find_replacement(c, essential_words, code_set, neighbors, rng, max_candidates=30):
            """Find a word d not in code_set that covers all essential_words."""
            if len(essential_words) == 0:
                return None  # c is redundant
            
            # Candidate d must be in the intersection of neighbors of all essential words
            # Start with neighbors of the first essential word
            ew0 = essential_words[0]
            candidates = neighbors[ew0, :].tolist()
            
            # Filter: d must be in neighbors of ALL essential words
            for ew in essential_words[1:]:
                ew_set = set(neighbors[ew, :].tolist())
                candidates = [d for d in candidates if d in ew_set]
            
            # Filter out existing code words
            candidates = [d for d in candidates if d not in code_set]
            
            if not candidates:
                return None
            
            # Shuffle and return a random candidate
            random.shuffle(candidates)
            return candidates[0] if candidates else None
        
        def try_single_swap(code_set, coverage, neighbors, deadline, rng):
            """Try to remove one codeword and add another (or just remove if redundant)."""
            if not code_set:
                return code_set, coverage, False
            
            # Prioritize codewords with fewer essential words (easier to replace)
            code_list = list(code_set)
            rng.shuffle(code_list)
            
            for c in code_list[:20]:  # Try up to 20 random codewords
                c_neighbors = neighbors[c, :]
                essential_words = c_neighbors[coverage[c_neighbors] == 1]
                
                if len(essential_words) == 0:
                    # c is redundant
                    code_set.remove(c)
                    coverage[neighbors[c, :]] -= 1
                    return code_set, coverage, True
                
                # Find a replacement
                d = find_replacement(c, essential_words, code_set, neighbors, rng)
                
                if d is not None:
                    # Verify: after removing c and adding d, all words are still covered
                    # Quick check: essential_words of c must be in neighbors[d]
                    d_neighbors_set = set(neighbors[d, :].tolist())
                    if all(ew in d_neighbors_set for ew in essential_words):
                        code_set.remove(c)
                        code_set.add(d)
                        coverage[neighbors[c, :]] -= 1
                        coverage[neighbors[d, :]] += 1
                        
                        # Try to reduce further
                        code_set = reduce_code(code_set, neighbors, deadline)
                        coverage = compute_coverage(code_set, neighbors)
                        return code_set, coverage, True
            
            return code_set, coverage, False
        
        def try_double_swap(code_set, coverage, neighbors, deadline, rng):
            """Try to remove two codewords and add one or two new ones."""
            if len(code_set) < 3:
                return code_set, coverage, False
            
            code_list = list(code_set)
            rng.shuffle(code_list)
            
            # Try pairs of codewords
            for i in range(min(10, len(code_list) - 1)):
                c1 = code_list[i]
                c2 = code_list[i + 1]
                
                # Compute combined essential words
                combined_cov = coverage.copy()
                combined_cov[neighbors[c1, :]] -= 1
                combined_cov[neighbors[c2, :]] -= 1
                
                # Words that become uncovered
                uncovered_words = np.where(combined_cov == 0)[0]
                
                if len(uncovered_words) > 4:
                    continue  # Too many to cover with 1-2 new words
                
                if len(uncovered_words) == 0:
                    # Both are redundant
                    code_set.remove(c1)
                    code_set.remove(c2)
                    coverage[neighbors[c1, :]] -= 1
                    coverage[neighbors[c2, :]] -= 1
                    return code_set, coverage, True
                
                # Try to find 1-2 words that cover all uncovered_words
                uw = uncovered_words.tolist()
                
                # Try single word
                if len(uw) <= 3:
                    # Find intersection of neighbors of all uncovered words
                    candidates = neighbors[uw[0], :].tolist()
                    for w in uw[1:]:
                        w_set = set(neighbors[w, :].tolist())
                        candidates = [d for d in candidates if d in w_set]
                    candidates = [d for d in candidates if d not in code_set]
                    
                    if candidates:
                        random.shuffle(candidates)
                        d = candidates[0]
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d)
                        coverage[neighbors[c1, :]] -= 1
                        coverage[neighbors[c2, :]] -= 1
                        coverage[neighbors[d, :]] += 1
                        
                        code_set = reduce_code(code_set, neighbors, deadline)
                        coverage = compute_coverage(code_set, neighbors)
                        return code_set, coverage, True
                
                # Try two words
                if len(uw) <= 6:
                    # Split uncovered words into two groups
                    for split in range(1, len(uw)):
                        group1 = uw[:split]
                        group2 = uw[split:]
                        
                        # Find word for group1
                        cand1 = neighbors[group1[0], :].tolist()
                        for w in group1[1:]:
                            w_set = set(neighbors[w, :].tolist())
                            cand1 = [d for d in cand1 if d in w_set]
                        cand1 = [d for d in cand1 if d not in code_set]
                        
                        if not cand1:
                            continue
                        
                        # Find word for group2
                        cand2 = neighbors[group2[0], :].tolist()
                        for w in group2[1:]:
                            w_set = set(neighbors[w, :].tolist())
                            cand2 = [d for d in cand2 if d in w_set]
                        cand2 = [d for d in cand2 if d not in code_set and d not in cand1]
                        
                        if not cand2:
                            continue
                        
                        random.shuffle(cand1)
                        random.shuffle(cand2)
                        d1 = cand1[0]
                        d2 = cand2[0]
                        
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d1)
                        code_set.add(d2)
                        coverage[neighbors[c1, :]] -= 1
                        coverage[neighbors[c2, :]] -= 1
                        coverage[neighbors[d1, :]] += 1
                        coverage[neighbors[d2, :]] += 1
                        
                        code_set = reduce_code(code_set, neighbors, deadline)
                        coverage = compute_coverage(code_set, neighbors)
                        return code_set, coverage, True
            
            return code_set, coverage, False
        
        def add_reduce(code_set, coverage, neighbors, deadline, rng):
            """Add a random word, then reduce to escape local minima."""
            if not code_set:
                return code_set, coverage, False
            
            w = rng.integers(0, N)
            if w in code_set:
                return code_set, coverage, False
            
            new_set = set(code_set)
            new_set.add(w)
            new_coverage = coverage.copy()
            new_coverage[neighbors[w, :]] += 1
            
            new_set = reduce_code(new_set, neighbors, deadline)
            
            if len(new_set) < len(code_set):
                new_cov = compute_coverage(new_set, neighbors)
                return new_set, new_cov, True
            
            return code_set, coverage, False
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts with different strategies
        modes = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
        
        for i in range(20):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.default_rng(42 + i * 13)
            mode = modes[i % len(modes)]
            
            code_set = greedy_init(rng, mode)
            code_set = reduce_code(code_set, neighbors, deadline)
            
            coverage = compute_coverage(code_set, neighbors)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for _ in range(150):
                    if time.time() > deadline - 5:
                        break
                    
                    r = rng.random()
                    if r < 0.5:
                        new_set, new_coverage, was_improved = try_single_swap(
                            code_set, coverage, neighbors, deadline, rng
                        )
                    elif r < 0.8:
                        new_set, new_coverage, was_improved = try_double_swap(
                            code_set, coverage, neighbors, deadline, rng
                        )
                    else:
                        new_set, new_coverage, was_improved = add_reduce(
                            code_set, coverage, neighbors, deadline, rng
                        )
                    
                    if was_improved:
                        code_set = new_set
                        coverage = new_coverage
                        improved = True
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
                
            if best_size <= 486:
                break
        
        # Polish pass: thorough 1-swap and 2-swap on the best code
        if best_code is not None and time.time() < deadline - 3:
            code_set = set(best_code)
            coverage = compute_coverage(code_set, neighbors)
            rng = np.random.default_rng(999)
            
            for _ in range(500):
                if time.time() > deadline - 3:
                    break
                
                r = rng.random()
                if r < 0.7:
                    _, _, was_improved = try_single_swap(
                        code_set, coverage, neighbors, deadline, rng
                    )
                    if was_improved:
                        coverage = compute_coverage(code_set, neighbors)
                else:
                    _, _, was_improved = try_double_swap(
                        code_set, coverage, neighbors, deadline, rng
                    )
                    if was_improved:
                        coverage = compute_coverage(code_set, neighbors)
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set
        
        if best_code is None:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
