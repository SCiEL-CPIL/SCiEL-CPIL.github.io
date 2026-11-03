import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 115.0  # Safety margin under 120s limit
    
    # Build space and spheres
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Build spheres (Hamming balls of given radius)
    sphere = []
    for w in words:
        s = [word_to_idx[w]]
        if radius >= 1:
            for pos in range(n):
                cur = w[pos]
                for v in range(q):
                    if v != cur:
                        w2 = list(w)
                        w2[pos] = v
                        s.append(word_to_idx[tuple(w2)])
        sphere.append(s)
    
    def coverage_counts(code_idx):
        """Compute coverage count for each word index."""
        counts = np.zeros(total, dtype=np.int32)
        for ci in code_idx:
            for w in sphere[ci]:
                counts[w] += 1
        return counts
    
    def is_valid(code_idx):
        """Check if code covers all words."""
        if not code_idx:
            return total == 0
        counts = coverage_counts(code_idx)
        return np.all(counts > 0)
    
    def local_search(code_idx):
        """Remove redundant codewords until fixed point."""
        code = sorted(set(code_idx))
        if not code:
            return code
        
        counts = coverage_counts(code)
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # Check if removing ci leaves all words in sphere[ci] still covered
                redundant = True
                for w in sphere[ci]:
                    if counts[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    for w in sphere[ci]:
                        counts[w] -= 1
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code
    
    def greedy_cover(uncovered_set):
        """Greedy set cover: repeatedly pick center covering most uncovered."""
        if not uncovered_set:
            return []
        
        chosen = []
        uncov = set(uncovered_set)
        
        while uncov:
            best_c = -1
            best_cnt = -1
            # Sample a subset of centers if too many for speed
            candidates = list(range(total))
            random.shuffle(candidates)
            # For efficiency, check all but this is fine for 729 words
            for c in candidates:
                cnt = sum(1 for w in sphere[c] if w in uncov)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            chosen.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
        
        return chosen
    
    def repair(code_idx, uncovered):
        """Add centers to cover uncovered words, then local search."""
        added = greedy_cover(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)
    
    # Seed with tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            seed_idx.append(word_to_idx[tuple(w)])
    
    best_code = local_search(seed_idx)
    best_size = len(best_code)
    
    rng = random.Random(20240517)
    
    # Multiple greedy restarts with random tie-breaking
    max_trials = 100
    for trial in range(max_trials):
        if time.time() > DEADLINE - 20:
            break
        
        # Randomized greedy: shuffle candidates before each selection
        uncov = set(range(total))
        code = []
        
        while uncov and len(code) < best_size + 10:
            # Compute gain for each center
            best_c = -1
            best_cnt = -1
            # For efficiency, use numpy for gain computation
            # But for 729 words, simple loop is fine
            for c in range(total):
                cnt = sum(1 for w in sphere[c] if w in uncov)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
                elif cnt == best_cnt and best_c != -1:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_c = c
            
            if best_c == -1 or best_cnt <= 0:
                break
            
            code.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
        
        if uncov:
            continue
        
        code2 = local_search(code)
        if len(code2) < best_size:
            best_size = len(code2)
            best_code = code2
    
    # Perturbation: drop 1-2 codewords, repair, local search
    iterations = 0
    while time.time() < DEADLINE - 5:
        iterations += 1
        code = list(best_code)
        k = len(code)
        if k < 2:
            break
        
        # Drop 1 codeword
        drop_idx = rng.randrange(k)
        new_code = [c for j, c in enumerate(code) if j != drop_idx]
        
        counts = coverage_counts(new_code)
        uncovered = set()
        for w in range(total):
            if counts[w] == 0:
                uncovered.add(w)
        
        if not uncovered:
            # Already valid without the dropped codeword
            repaired = local_search(new_code)
        else:
            repaired = repair(new_code, uncovered)
        
        if len(repaired) < best_size:
            best_size = len(repaired)
            best_code = repaired
        
        # Every few iterations, try dropping 2 codewords
        if iterations % 3 == 0 and k >= 2:
            d1 = rng.randrange(k)
            d2 = rng.randrange(k)
            if d1 == d2:
                d2 = (d2 + 1) % k
            new_code = [c for j, c in enumerate(code) if j != d1 and j != d2]
            
            counts = coverage_counts(new_code)
            uncovered = set()
            for w in range(total):
                if counts[w] == 0:
                    uncovered.add(w)
            
            if not uncovered:
                repaired = local_search(new_code)
            else:
                repaired = repair(new_code, uncovered)
            
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired
    
    # Final validation
    if not is_valid(best_code):
        # Fallback to seed
        fallback = local_search(seed_idx)
        return [list(words[i]) for i in fallback]
    
    return [list(words[i]) for i in best_code]
