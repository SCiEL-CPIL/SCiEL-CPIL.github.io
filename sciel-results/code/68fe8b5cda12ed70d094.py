"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Precompute the space
    total = q ** n  # 2187
    # Convert word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % q)
            idx //= q
        word.reverse()
        return word
    
    # Precompute coverage: for each index, which indices it covers (itself + all distance-1 neighbors)
    coverage = []
    for idx in range(total):
        w = idx_to_word(idx)
        covered = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                covered.append(word_to_idx(w))
            w[pos] = orig
        coverage.append(covered)
    
    # Greedy set cover with randomization
    def greedy_cover(rng):
        uncovered = [True] * total
        code = []
        uncovered_count = total
        # For each candidate, track how many currently-uncovered words it covers
        # We'll recompute each iteration for simplicity (2187 * 15 = ~33K ops, fine)
        while uncovered_count > 0:
            if time.time() > DEADLINE:
                return None
            best_gain = -1
            best_candidates = []
            for cand in range(total):
                gain = 0
                for cov_idx in coverage[cand]:
                    if uncovered[cov_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [cand]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(cand)
            if best_gain <= 0:
                break
            # Random tie-breaking
            chosen = rng.choice(best_candidates)
            code.append(chosen)
            for cov_idx in coverage[chosen]:
                if uncovered[cov_idx]:
                    uncovered[cov_idx] = False
                    uncovered_count -= 1
        return code
    
    # Check if a code covers everything
    def is_valid(code):
        covered = [False] * total
        for c in code:
            for cov_idx in coverage[c]:
                covered[cov_idx] = True
        return all(covered)
    
    # Local improvement: try removing redundant codewords
    def local_improve(code, rng):
        improved = True
        while improved:
            if time.time() > DEADLINE:
                return code
            improved = False
            # Try removing each codeword
            for i in range(len(code)):
                if time.time() > DEADLINE:
                    return code
                test_code = code[:i] + code[i+1:]
                if is_valid(test_code):
                    code = test_code
                    improved = True
                    break
        return code
    
    # Local improvement: try swapping
    def local_swap(code, rng):
        improved = True
        while improved:
            if time.time() > DEADLINE:
                return code
            improved = False
            for i in range(len(code)):
                if time.time() > DEADLINE:
                    return code
                # Try removing code[i] and adding a random other word
                for _ in range(50):  # limited random trials per position
                    if time.time() > DEADLINE:
                        return code
                    new_word = rng.randint(0, total - 1)
                    if new_word in code:
                        continue
                    test_code = code[:i] + [new_word] + code[i+1:]
                    if is_valid(test_code):
                        code = test_code
                        improved = True
                        break
                if improved:
                    break
        return code
    
    best_code = None
    best_len = total + 1
    
    # Multiple randomized greedy restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 10.0:
            break
        rng = random.Random(restart * 12345 + 67890)
        code = greedy_cover(rng)
        if code is None:
            continue
        # Local improvement
        code = local_improve(code, rng)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            # Try swap improvement on the best
            code2 = local_swap(code, rng)
            if len(code2) < best_len:
                best_len = len(code2)
                best_code = code2[:]
    
    if best_code is None:
        # Fallback: use the seed construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert indices back to words
    return [idx_to_word(idx) for idx in best_code]
# EVOLVE-BLOCK-END
