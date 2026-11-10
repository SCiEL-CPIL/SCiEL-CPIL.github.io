"""Ternary covering code K_3(6,1) via greedy set cover with local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill
    
    total = q ** n
    # Generate all words as integers 0..total-1
    # Convert int to word
    def int_to_word(x):
        return [x % q, (x // q) % q, (x // q**2) % q, (x // q**3) % q, (x // q**4) % q, (x // q**5) % q]
    
    def word_to_int(w):
        return sum(w[i] * (q ** i) for i in range(n))
    
    # Precompute the ball of radius 1 around each word (as a set of ints)
    # Each word covers itself + all words differing in exactly one position
    # For q=3, n=6: each ball has 1 + 6*2 = 13 elements
    balls = []
    for x in range(total):
        w = int_to_word(x)
        ball = {x}
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w[pos] = val
                    ball.add(word_to_int(w))
            w[pos] = orig
        balls.append(ball)
    
    # Convert balls to numpy bitsets for fast operations
    all_bits = np.zeros(total, dtype=np.uint64)
    for x in range(total):
        for y in balls[x]:
            all_bits[x] |= (1 << (y % 64)) if y < 64 else 0
    # Actually for 729 elements, use a different representation
    # Use numpy arrays of shape (total, total) boolean - 729*729 = 531441 booleans, fine
    
    # coverage_matrix[i] = boolean array of which words are covered by codeword i
    cov = np.zeros((total, total), dtype=bool)
    for i in range(total):
        for y in balls[i]:
            cov[i, y] = True
    
    def greedy_cover(uncovered_mask, candidates=None, rng=None):
        """Greedy set cover. uncovered_mask: boolean array of what's still uncovered.
        Returns list of chosen codeword indices."""
        if candidates is None:
            candidates = list(range(total))
        chosen = []
        remaining = uncovered_mask.copy()
        rem_count = remaining.sum()
        cands = list(candidates)
        
        while rem_count > 0 and cands:
            # Find candidate that covers most uncovered
            # Coverage count = sum of cov[c] & remaining
            gains = cov[cands, :].dot(remaining.astype(np.uint8))
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            if best_gain == 0:
                break
            chosen.append(cands[best_idx])
            remaining[cov[cands[best_idx], :]] = False
            rem_count -= best_gain
            cands.pop(best_idx)
        
        return chosen
    
    def is_valid(chosen):
        """Check if chosen codewords cover everything."""
        if not chosen:
            return total == 0
        covered = np.zeros(total, dtype=bool)
        for c in chosen:
            covered[cov[c, :]] = True
        return covered.all()
    
    def local_search(chosen, rng):
        """Try to improve by swapping or removing codewords."""
        chosen = list(chosen)
        improved = True
        while improved:
            improved = False
            # Try removing each codeword
            for i in range(len(chosen)):
                trial = chosen[:i] + chosen[i+1:]
                if is_valid(trial):
                    chosen = trial
                    improved = True
                    break
            if improved:
                continue
            # Try swapping one codeword with another
            for i in range(len(chosen)):
                for c in range(total):
                    if c in chosen:
                        continue
                    trial = chosen[:i] + [c] + chosen[i+1:]
                    if is_valid(trial):
                        # Check if it allows further removal
                        chosen = trial
                        improved = True
                        break
                if improved:
                    break
        return chosen
    
    # Strategy 1: Pure greedy
    rng = random.Random(42)
    full_mask = np.ones(total, dtype=bool)
    
    best_code = None
    best_len = total + 1
    
    # Multiple greedy attempts with different orderings
    for attempt in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        if attempt == 0:
            # Standard greedy: pick in order of max coverage
            candidates = list(range(total))
        else:
            # Randomize: shuffle candidates
            candidates = list(range(total))
            rng.shuffle(candidates)
        
        code = greedy_cover(full_mask, candidates, rng)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
        
        # Local search on the result
        if time.time() - start_time > DEADLINE:
            break
        code_ls = local_search(code, rng)
        if len(code_ls) < best_len:
            best_len = len(code_ls)
            best_code = code_ls
    
    # Strategy 2: Greedy with priority based on uncovered density
    for attempt in range(30):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random perturbation of the greedy
        candidates = list(range(total))
        rng.shuffle(candidates)
        
        # Modified greedy: at each step, consider top-k candidates
        remaining = full_mask.copy()
        rem_count = total
        chosen = []
        cands = list(candidates)
        
        while rem_count > 0 and cands:
            if time.time() - start_time > DEADLINE:
                break
            # Sample a subset of candidates for speed
            if len(cands) > 200:
                sample = rng.sample(cands, 200)
            else:
                sample = cands
            
            gains = cov[sample, :].dot(remaining.astype(np.uint8))
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            if best_gain == 0:
                break
            chosen.append(sample[best_idx])
            remaining[cov[sample[best_idx], :]] = False
            rem_count -= best_gain
            cands.remove(sample[best_idx])
        
        if is_valid(chosen) and len(chosen) < best_len:
            best_len = len(chosen)
            best_code = chosen
        
        code_ls = local_search(chosen, rng)
        if len(code_ls) < best_len:
            best_len = len(code_ls)
            best_code = code_ls
    
    # Strategy 3: Start from a structured code and try to improve
    # The tetracode-based construction gives 73
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    structured = []
    for h in tetracode:
        for tail in product(range(3), repeat=n - 4):
            structured.append(h + list(tail))
    structured_ints = [word_to_int(w) for w in structured]
    
    if is_valid(structured_ints):
        if len(structured_ints) < best_len:
            best_len = len(structured_ints)
            best_code = structured_ints
        code_ls = local_search(structured_ints, rng)
        if len(code_ls) < best_len:
            best_len = len(code_ls)
            best_code = code_ls
    
    # Strategy 4: Iterated local search - random starts
    for attempt in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initial code
        size = rng.randint(60, 80)
        init = rng.sample(range(total), size)
        if not is_valid(init):
            # Add more until valid
            remaining = np.ones(total, dtype=bool)
            for c in init:
                remaining[cov[c, :]] = False
            cands = [c for c in range(total) if c not in init]
            while remaining.sum() > 0 and cands:
                gains = cov[cands, :].dot(remaining.astype(np.uint8))
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                init.append(cands[best_idx])
                remaining[cov[cands[best_idx], :]] = False
                cands.pop(best_idx)
        
        if not is_valid(init):
            continue
        
        code_ls = local_search(init, rng)
        if len(code_ls) < best_len:
            best_len = len(code_ls)
            best_code = code_ls
    
    # Final: ensure we have a valid code
    if best_code is None or not is_valid(best_code):
        # Fallback to the structured code
        best_code = structured_ints
    
    # Convert back to list of words
    result = [int_to_word(c) for c in best_code]
    
    # Verify distinctness
    assert len(set(tuple(w) for w in result)) == len(result)
    
    return result
