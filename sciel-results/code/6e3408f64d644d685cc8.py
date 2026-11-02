"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _word_to_idx(word, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _hamming_dist(w1, w2):
    return sum(1 for a, b in zip(w1, w2) if a != b)


def _compute_balls(q, n, radius):
    """Precompute ball membership as bitmasks for fast coverage checks."""
    total = q ** n
    # balls[i] is a bitmask of all words within radius of word i
    balls = [0] * total
    
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    for i, word in enumerate(all_words):
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = word[:]
                        new_word[pos] = new_val
                        mask |= 1 << _word_to_idx(new_word, q)
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = word[pos1]
                    orig2 = word[pos2]
                    for new_val1 in range(q):
                        if new_val1 == orig1:
                            continue
                        for new_val2 in range(q):
                            if new_val2 == orig2:
                                continue
                            new_word = word[:]
                            new_word[pos1] = new_val1
                            new_word[pos2] = new_val2
                            mask |= 1 << _word_to_idx(new_word, q)
        balls[i] = mask
    
    return balls


def _coverage_mask(balls, code_indices):
    """Compute the union of balls for all codewords as a single bitmask."""
    mask = 0
    for idx in code_indices:
        mask |= balls[idx]
    return mask


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    total = q ** n
    full_mask = (1 << total) - 1
    
    # Start with the direct-sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    code_indices = [_word_to_idx(c, q) for c in code]
    code_set = set(code_indices)
    
    # Precompute balls as bitmasks
    balls = _compute_balls(q, n, radius)
    
    # Phase 1: Greedy removal of redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code_indices) - 1, -1, -1):
            if time.time() >= deadline:
                break
            candidate_set = code_set - {code_indices[i]}
            covered = _coverage_mask(balls, candidate_set)
            if covered == full_mask:
                code_set.discard(code_indices[i])
                code_indices.pop(i)
                code.pop(i)
                improved = True
    
    best_code_indices = list(code_indices)
    best_size = len(code_indices)
    
    # Phase 2: Simulated Annealing
    # State: set of selected codeword indices
    # Objective: minimize number of codewords while maintaining full coverage
    # Moves: remove a codeword (if still covered), add a codeword, swap a codeword
    
    current_code = list(code_indices)
    current_set = set(code_indices)
    current_mask = _coverage_mask(balls, current_code)
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        # Try different move types
        move_type = rng.random()
        
        if move_type < 0.3 and len(current_code) > 1:
            # Remove a codeword
            i = rng.randint(0, len(current_code) - 1)
            removed = current_code[i]
            new_set = current_set - {removed}
            new_mask = _coverage_mask(balls, new_set)
            
            if new_mask == full_mask:
                # Always accept if still covered
                current_set.discard(removed)
                current_code.pop(i)
                current_mask = new_mask
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_indices = list(current_code)
            else:
                # Compute uncovered count
                uncovered = (full_mask & ~new_mask).bit_count()
                delta = -1  # We're removing, so size decreases by 1
                # Energy: we prefer smaller codes, but need coverage
                # Penalty for uncovered words
                energy_change = delta + 100 * uncovered
                if energy_change < 0 or rng.random() < np.exp(-energy_change / T):
                    current_set.discard(removed)
                    current_code.pop(i)
                    current_mask = new_mask
        elif move_type < 0.5:
            # Add a random codeword
            new_idx = rng.randint(0, total - 1)
            if new_idx not in current_set:
                new_set = current_set | {new_idx}
                new_mask = _coverage_mask(balls, new_set)
                delta = 1  # Size increases by 1
                energy_change = delta
                if energy_change < 0 or rng.random() < np.exp(-energy_change / T):
                    current_set.add(new_idx)
                    current_code.append(new_idx)
                    current_mask = new_mask
        else:
            # Swap: remove one, add another
            if len(current_code) > 0:
                i = rng.randint(0, len(current_code) - 1)
                removed = current_code[i]
                new_idx = rng.randint(0, total - 1)
                if new_idx != removed and new_idx not in current_set:
                    new_set = current_set - {removed} | {new_idx}
                    new_mask = _coverage_mask(balls, new_set)
                    delta = 0  # Size unchanged
                    uncovered = (full_mask & ~new_mask).bit_count()
                    energy_change = 100 * uncovered
                    if energy_change < 0 or rng.random() < np.exp(-energy_change / T):
                        current_set.discard(removed)
                        current_set.add(new_idx)
                        current_code[i] = new_idx
                        current_mask = new_mask
        
        T *= cooling
        
        # Periodic greedy cleanup
        if int((deadline - time.time()) * 10) % 50 == 0 and time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for j in range(len(current_code) - 1, -1, -1):
                    if time.time() >= deadline:
                        break
                    cand_set = current_set - {current_code[j]}
                    cand_mask = _coverage_mask(balls, cand_set)
                    if cand_mask == full_mask:
                        current_set.discard(current_code[j])
                        current_code.pop(j)
                        current_mask = cand_mask
                        improved = True
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_indices = list(current_code)
    
    # Final greedy cleanup
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_code) - 1, -1, -1):
                if time.time() >= deadline:
                    break
                cand_set = current_set - {current_code[i]}
                cand_mask = _coverage_mask(balls, cand_set)
                if cand_mask == full_mask:
                    current_set.discard(current_code[i])
                    current_code.pop(i)
                    current_mask = cand_mask
                    improved = True
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_indices = list(current_code)
    
    # Verify and return the best solution found
    result = best_code_indices
    # Final verification
    final_mask = _coverage_mask(balls, result)
    if final_mask != full_mask:
        # Fallback to a valid solution
        result = list(code_indices)
    
    return [_idx_to_word(idx, q, n) for idx in result]
# EVOLVE-BLOCK-END
