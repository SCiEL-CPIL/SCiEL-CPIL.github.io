"""Greedy set-cover for K_q(n, radius) with time guard and local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq


def _ball_indices(word, q, n):
    """Return the set of indices (as ints in 0..q^n-1) within Hamming distance 1 of word."""
    base = 0
    for i in range(n):
        base = base * q + word[i]
    ball = [base]
    for i in range(n):
        pw = q ** (n - 1 - i)
        w_i = word[i]
        for v in range(q):
            if v != w_i:
                ball.append(base + (v - w_i) * pw)
    return ball


def _ball_indices_fast(word, q, n, pows):
    """Fast ball: word as int index + offsets per position."""
    base = 0
    for i in range(n):
        base = base * q + word[i]
    ball = [base]
    for i in range(n):
        pw = pows[i]
        w_i = word[i]
        for v in range(q):
            if v != w_i:
                ball.append(base + (v - w_i) * pw)
    return ball


def _int_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s kill

    total = q ** n
    pows = [q ** (n - 1 - i) for i in range(n)]

    # Precompute all balls and reverse balls
    all_balls = [None] * total
    all_rev = [None] * total  # rev[b] = list of u such that b in all_balls[u]
    for idx in range(total):
        w = _int_to_word(idx, q, n)
        all_balls[idx] = _ball_indices_fast(w, q, n, pows)

    for idx in range(total):
        for b in all_balls[idx]:
            if all_rev[b] is None:
                all_rev[b] = []
            all_rev[b].append(idx)

    # Ensure all_rev is fully populated
    for b in range(total):
        if all_rev[b] is None:
            all_rev[b] = []

    def to_idx(word):
        b = 0
        for x in word:
            b = b * q + x
        return b

    # --- Strong initial seed: direct sum of ternary tetracode (for q=3) ---
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed = [h + list(tail)
                for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        seed = []

    # Convert to index-based representation
    code_indices = []
    covered = [False] * total  # covered[b] = True if b is covered by some codeword

    for w in seed:
        idx = to_idx(w)
        code_indices.append(idx)
        for b in all_balls[idx]:
            covered[b] = True

    # Initialize gain: for each word u, gain[u] = number of uncovered words in ball(u)
    gain = [0] * total
    for u in range(total):
        g = 0
        for b in all_balls[u]:
            if not covered[b]:
                g += 1
        gain[u] = g

    # Use a max-heap for greedy selection
    heap = [(-gain[idx], idx) for idx in range(total) if gain[idx] > 0]
    heapq.heapify(heap)

    # Greedy set cover
    while heap:
        if time.time() > deadline:
            break
        # Pop until we find a valid (current) max
        while heap:
            neg_g, idx = heap[0]
            g = -neg_g
            if g == gain[idx] and gain[idx] > 0:
                break
            heapq.heappop(heap)
        if not heap:
            break
        neg_g, idx = heapq.heappop(heap)
        g = -neg_g
        if g <= 0:
            break

        # Add idx as a codeword
        code_indices.append(idx)
        ball = all_balls[idx]

        # Update covered and gains
        for b in ball:
            if not covered[b]:
                covered[b] = True
                # For every word u whose ball contains b, gain[u] decreases by 1
                for u in all_rev[b]:
                    gain[u] -= 1
                    # Push updated gain
                    if gain[u] > 0:
                        heapq.heappush(heap, (-gain[u], u))

    # Convert to words
    best_code = list(code_indices)
    best_count = len(code_indices)

    # --- Local search: try to remove redundant codewords ---
    def is_redundant(code_set, idx):
        """Check if removing idx from code_set still covers everything."""
        # Compute coverage of all other codewords
        covered_check = [False] * total
        for c in code_set:
            if c == idx:
                continue
            for b in all_balls[c]:
                covered_check[b] = True
        # Check if ball(idx) is fully covered by others
        for b in all_balls[idx]:
            if not covered_check[b]:
                return False
        return True

    # Try removing redundant codewords
    removed = True
    while removed:
        removed = False
        for i in range(len(code_indices) - 1, -1, -1):
            if is_redundant(code_indices, code_indices[i]):
                code_indices.pop(i)
                removed = True
                best_count = len(code_indices)
                best_code = list(code_indices)

    # --- Local search: try swaps to reduce size ---
    # For each pair of codewords, try swapping one for another that covers more uncovered
    # Actually, try: for each codeword c, try replacing it with another word c' 
    # such that the new code is still valid and has same or smaller size.
    # Better: try to find if we can remove one codeword and add one that covers more.
    
    # More targeted: try all possible single swaps
    # For each codeword c in the code, and each word c' not in the code,
    # check if (code - {c} + {c'}) is valid. If so, and if it allows further removal, accept.
    
    # This is expensive. Instead, try a simpler heuristic:
    # For each codeword c, check if there's a word c' not in code such that
    # ball(c') covers all of ball(c) that isn't covered by others.
    
    # Let's try a different approach: iterated local search with random restarts
    # Try multiple random perturbations of the current code
    
    rng = random.Random(42)
    
    def verify_coverage(code_set):
        """Check if code_set covers all words."""
        covered_check = [False] * total
        for c in code_set:
            for b in all_balls[c]:
                covered_check[b] = True
        return all(covered_check)
    
    def try_remove(code_list):
        """Try to remove redundant codewords from code_list."""
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                if is_redundant(code_list, code_list[i]):
                    code_list.pop(i)
                    changed = True
        return code_list
    
    # Try random swaps: replace a codeword with a random other word, then try to remove
    for iteration in range(500):
        if time.time() > deadline:
            break
        if len(code_indices) <= best_count:
            break
            
        # Pick a random codeword to replace
        c_idx = rng.randint(0, len(code_indices) - 1)
        c = code_indices[c_idx]
        
        # Try replacing with a random word
        c_new = rng.randint(0, total - 1)
        if c_new in code_indices:
            continue
        
        new_code = list(code_indices)
        new_code[c_idx] = c_new
        new_code = try_remove(new_code)
        
        if len(new_code) < len(code_indices):
            code_indices = new_code
            best_count = len(code_indices)
            best_code = list(code_indices)
        elif len(new_code) == len(code_indices) and rng.random() < 0.1:
            # Accept same-size with some probability for diversification
            code_indices = new_code
            best_code = list(code_indices)

    # Final: try greedy repair if not fully covered (shouldn't happen but safety)
    if not verify_coverage(code_indices):
        # Add back necessary words
        covered_check = [False] * total
        for c in code_indices:
            for b in all_balls[c]:
                covered_check[b] = True
        for b in range(total):
            if not covered_check[b]:
                # Find a word that covers b
                for u in all_rev[b]:
                    if u not in code_indices:
                        code_indices.append(u)
                        for b2 in all_balls[u]:
                            covered_check[b2] = True
                        break

    result = [_int_to_word(idx, q, n) for idx in code_indices]
    return result
# EVOLVE-BLOCK-END
