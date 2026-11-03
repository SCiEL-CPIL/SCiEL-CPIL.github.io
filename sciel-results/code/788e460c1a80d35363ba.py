"""Greedy set-cover for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
import heapq

def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0

    total = q ** n
    # Precompute powers for index conversion
    pows = [q ** (n - 1 - i) for i in range(n)]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute ball lists for all indices
    # ball_list[idx] is a list of indices within Hamming distance radius of idx
    ball_lists = [[] for _ in range(total)]
    if radius == 1:
        for idx in range(total):
            w = idx_to_word(idx)
            ball = [idx]
            for i in range(n):
                pw = pows[i]
                w_i = w[i]
                for v in range(q):
                    if v != w_i:
                        new_idx = idx + (v - w_i) * pw
                        ball.append(new_idx)
            ball_lists[idx] = ball
    elif radius == 0:
        for idx in range(total):
            ball_lists[idx] = [idx]
    else:
        # General radius (slower, but correct)
        from itertools import combinations
        for idx in range(total):
            w = idx_to_word(idx)
            ball = {idx}
            for r in range(1, radius + 1):
                for positions in combinations(range(n), r):
                    for vals in itertools.product(range(q), repeat=r):
                        new_w = w[:]
                        for pos, val in zip(positions, vals):
                            if val == new_w[pos]:
                                break
                        else:
                            new_idx = word_to_idx(new_w)
                            ball.add(new_idx)
            ball_lists[idx] = list(ball)

    # --- Initial Seed ---
    seed_indices = []
    if q == 3 and n >= 4 and radius == 1:
        # Generate Tetracode words (9 words of length 4, radius 1)
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        # Generate tails for remaining n-4 positions
        tails = list(itertools.product(range(q), repeat=n - 4))
        
        for h in tetracode:
            for tail in tails:
                word = h + list(tail)
                seed_indices.append(word_to_idx(word))
    else:
        seed_indices = []

    code = list(seed_indices)
    
    # Compute coverage counts and uncovered mask
    coverage_count = [0] * total
    for c in code:
        for idx in ball_lists[c]:
            coverage_count[idx] += 1
    
    uncovered_mask = 0
    for j in range(total):
        if coverage_count[j] == 0:
            uncovered_mask |= (1 << j)
    
    best_code = list(code)
    best_count = len(code)

    # --- Lattice Reduction Heuristic for Removal Ordering ---
    # Map ternary space to a complex lattice using omega = e^(2*pi*i/3)
    omega = complex(-0.5, math.sqrt(3)/2)
    
    def ternary_to_complex(word):
        z = 0 + 0j
        for i, digit in enumerate(word):
            if digit == 0:
                z += omega ** (2 * i)
            elif digit == 1:
                z += (omega ** 2) ** (2 * i)
            else:
                z += 1.0 + 0j
        return z
    
    # Compute lattice scores for initial code
    def compute_lattice_scores(codes):
        scores = {}
        for c in codes:
            w = idx_to_word(c)
            z = ternary_to_complex(w)
            scores[c] = abs(z) ** 2
        return scores

    # --- Removal Phase with Lattice Guidance ---
    # We'll use a randomized restart strategy with lattice-guided removal
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Reset to initial seed for each restart (or keep best so far)
        if restart == 0:
            current_code = list(code)
        else:
            # Start from best found so far
            current_code = list(best_code)
        
        # Recompute coverage
        coverage_count = [0] * total
        for c in current_code:
            for idx in ball_lists[c]:
                coverage_count[idx] += 1
        
        # Check if fully covered
        is_covered = all(coverage_count[j] > 0 for j in range(total))
        
        # Lattice-guided removal
        if is_covered:
            lattice_scores = compute_lattice_scores(current_code)
            
            max_passes = 100
            for _ in range(max_passes):
                if time.time() > deadline - 10:
                    break
                
                # Sort by lattice complexity (descending) - remove high complexity first
                # But also add some randomness to escape local minima
                if random.random() < 0.3:
                    # Random order
                    random.shuffle(current_code)
                    sorted_code = current_code
                else:
                    # Lattice-guided order
                    sorted_code = sorted(current_code, key=lambda c: lattice_scores.get(c, 0), reverse=True)
                
                removed_any = False
                new_code = []
                for c in sorted_code:
                    # Check if c is redundant
                    is_redundant = True
                    for idx in ball_lists[c]:
                        if coverage_count[idx] <= 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        for idx in ball_lists[c]:
                            coverage_count[idx] -= 1
                        removed_any = True
                    else:
                        new_code.append(c)
                
                current_code = new_code
                
                if not removed_any:
                    break
        
        # Track best
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = list(current_code)
        
        # If we've reached the theoretical lower bound, stop
        if best_count <= 71:
            break

    # --- Final Greedy Fill (if needed) ---
    # Check if best_code covers everything
    coverage_count = [0] * total
    for c in best_code:
        for idx in ball_lists[c]:
            coverage_count[idx] += 1
    
    uncovered_mask = 0
    for j in range(total):
        if coverage_count[j] == 0:
            uncovered_mask |= (1 << j)
    
    while uncovered_mask:
        if time.time() > deadline - 5:
            break
        
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            gain = 0
            for b in ball_lists[idx]:
                if (uncovered_mask >> b) & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_gain == 0:
            break
            
        best_code.append(best_idx)
        
        for b in ball_lists[best_idx]:
            coverage_count[b] += 1
            if coverage_count[b] == 1:
                uncovered_mask &= ~(1 << b)

    # Convert back to words
    result = [idx_to_word(idx) for idx in best_code]
    return result
# EVOLVE-BLOCK-END
