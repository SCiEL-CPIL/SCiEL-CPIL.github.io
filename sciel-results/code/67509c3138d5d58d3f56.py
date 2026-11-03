"""Ternary covering code K_3(7,1) construction via greedy + local search."""
import time
import random
import itertools
from collections import defaultdict

def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))

def construct(q, n, radius):
    # Only the ternary n=7 r=1 case is the target; handle others gracefully.
    start = time.time()
    deadline = start + 140.0  # 10 s safety margin

    if q != 3 or n != 7 or radius != 1:
        # Fallback: trivial covering (all words) for small spaces, or seed construction
        if q ** n <= 10000:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        # Fallback similar to seed
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    q = 3
    n = 7
    total = q ** n  # 2187

    # Precompute all words as integers 0..2186, and their ball (self + 21 neighbors)
    # Word index = sum(d_i * 3^i) for digit d_i at position i (0-indexed from right)
    pow3 = [3 ** i for i in range(n)]

    all_words = []
    for idx in range(total):
        digits = []
        tmp = idx
        for i in range(n):
            digits.append(tmp % 3)
            tmp //= 3
        all_words.append(digits)

    # Precompute ball for each codeword index: list of word indices covered
    balls = []
    for idx in range(total):
        digits = all_words[idx]
        ball = [idx]
        for pos in range(n):
            orig = digits[pos]
            for v in range(q):
                if v != orig:
                    new_digits = digits[:]
                    new_digits[pos] = v
                    new_idx = sum(new_digits[i] * pow3[i] for i in range(n))
                    ball.append(new_idx)
        balls.append(ball)

    # Precompute for each word, the list of codeword indices that cover it (within radius 1)
    # This is the dual: for each word w, which codewords c have w in balls[c]
    coverers = [[] for _ in range(total)]
    for c_idx in range(total):
        for w_idx in balls[c_idx]:
            coverers[w_idx].append(c_idx)

    # Greedy set cover
    def greedy_init():
        covered = [False] * total
        selected = []
        # Track uncovered count per candidate codeword
        # For efficiency, maintain a count of how many newly-covered words each codeword would add
        # Simple approach: repeatedly pick codeword covering most uncovered words
        # For speed, we use a heuristic: pick codeword with max coverage of uncovered
        remaining = set(range(total))
        # Count uncovered per codeword
        uncovered_cnt = [len(balls[c]) for c in range(total)]
        # Actually, we need to track which words are still uncovered
        # Let's do a simpler greedy: at each step, pick the codeword that covers the most uncovered words
        # To speed up, we can precompute and update incrementally
        # For 2187 words, this is small enough to do naively with some optimization
        
        selected_set = set()
        covered = [False] * total
        uncovered_count = total
        
        while uncovered_count > 0:
            best_c = -1
            best_cov = 0
            for c in range(total):
                if c in selected_set:
                    continue
                cov = 0
                for w in balls[c]:
                    if not covered[w]:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_c = c
                if cov == 0:
                    continue
            if best_c == -1:
                break
            selected.append(best_c)
            selected_set.add(best_c)
            for w in balls[best_c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        
        return selected

    # Local search: try to remove codewords, then try 1-exchange swaps
    def local_search(selected, max_time):
        selected = list(selected)
        selected_set = set(selected)
        
        # Build coverage count: for each word, how many selected codewords cover it
        cov_count = [0] * total
        for c in selected:
            for w in balls[c]:
                cov_count[w] += 1
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Phase 1: Try to remove codewords
            for i in range(len(selected)):
                if time.time() >= max_time:
                    break
                c = selected[i]
                # Check if any word is only covered by c
                can_remove = True
                for w in balls[c]:
                    if cov_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove c
                    for w in balls[c]:
                        cov_count[w] -= 1
                    selected.pop(i)
                    selected_set.discard(c)
                    improved = True
                    break  # Restart removal phase
        
        # Phase 2: Try 1-exchange swaps
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try swapping a selected codeword with an unselected one
            for i in range(len(selected)):
                if time.time() >= max_time:
                    break
                c_old = selected[i]
                # Try replacing c_old with some unselected c_new
                # For efficiency, only try c_new that covers at least one word currently
                # covered only by c_old
                # First, find words covered only by c_old
                only_words = [w for w in balls[c_old] if cov_count[w] == 1]
                if not only_words:
                    continue
                
                # Find unselected codewords that cover at least one of these only_words
                candidates = set()
                for w in only_words:
                    for c_new in coverers[w]:
                        if c_new not in selected_set:
                            candidates.add(c_new)
                
                for c_new in candidates:
                    if time.time() >= max_time:
                        break
                    # Check if swapping c_old -> c_new keeps all words covered
                    # Words that lose coverage: words in ball(c_old) not in ball(c_new)
                    # Words that gain coverage: words in ball(c_new) not in ball(c_old)
                    # After swap, a word w is covered if cov_count[w] >= 1
                    # For w in ball(c_old) \ ball(c_new): cov_count[w] decreases by 1
                    # For w in ball(c_new) \ ball(c_old): cov_count[w] increases by 1
                    # So we need: for all w in ball(c_old) \ ball(c_new), cov_count[w] > 1
                    can_swap = True
                    for w in balls[c_old]:
                        if w not in set(balls[c_new]) and cov_count[w] == 1:
                            can_swap = False
                            break
                    if not can_swap:
                        continue
                    
                    # Perform swap
                    for w in balls[c_old]:
                        cov_count[w] -= 1
                    for w in balls[c_new]:
                        cov_count[w] += 1
                    selected[i] = c_new
                    selected_set.discard(c_old)
                    selected_set.add(c_new)
                    improved = True
                    break  # Restart
                if improved:
                    break
        
        return selected

    # Run multiple restarts
    best_code = None
    best_size = float('inf')
    rng = random.Random(42)

    # First, do a greedy init
    init_code = greedy_init()
    opt_code = local_search(init_code, deadline - 5.0)
    if len(opt_code) < best_size:
        best_size = len(opt_code)
        best_code = opt_code

    # If time remains, do additional restarts with random perturbations
    while time.time() < deadline - 5.0:
        # Perturb: start from best_code, remove some random codewords, re-greedy
        if best_code is None:
            break
        perturbed = list(best_code)
        # Remove up to 5 random codewords
        remove_count = min(5, len(perturbed))
        for _ in range(remove_count):
            if perturbed:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        # Re-greedy from perturbed
        covered = [False] * total
        selected = list(perturbed)
        selected_set = set(selected)
        uncovered_count = sum(1 for w in range(total) if not any(covered[x] for x in balls[w]))
        # Actually, let's just do a full greedy from scratch but seed with perturbed
        # Simpler: just do greedy_init again with a different random order
        init2 = greedy_init()
        opt2 = local_search(init2, deadline - 5.0)
        if len(opt2) < best_size:
            best_size = len(opt2)
            best_code = opt2

    if best_code is None:
        best_code = greedy_init()
        best_code = local_search(best_code, deadline)

    return [list(all_words[c]) for c in best_code]
