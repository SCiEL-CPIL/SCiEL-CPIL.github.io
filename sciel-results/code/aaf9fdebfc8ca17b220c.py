"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for i, v in enumerate(word):
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_ball_indices(word_idx, q, n, radius):
    """Generate all word indices within Hamming distance `radius` from `word_idx`."""
    word = _int_to_word(word_idx, q, n)
    ball = set()
    ball.add(word_idx)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(_word_to_int(new_word, q, n))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave safety margin before 150s kill
    
    total_words = q ** n
    ball_size = sum(
        (q - 1) ** r * (n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5) //
         (1 * 2 * 3 * 4 * 5) if r <= 5 else 0)
        for r in range(radius + 1)
    ) if n >= 5 else sum((q - 1) ** r for r in range(radius + 1))
    
    # Precompute balls as bitmasks (using Python's arbitrary-precision ints)
    # Each ball is represented as an integer where bit i is set if word i is in the ball
    balls = [0] * total_words
    for idx in range(total_words):
        ball_indices = _generate_ball_indices(idx, q, n, radius)
        mask = 0
        for i in ball_indices:
            mask |= (1 << i)
        balls[idx] = mask
    
    all_masks = balls  # alias
    
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random restarts with greedy + local search
    num_restarts = 50
    for attempt in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(12345 + attempt)
        
        # Greedy set cover using bitmasks
        current_solution = []
        covered_mask = 0
        
        # Start with a random word
        first_idx = rng.randint(0, total_words - 1)
        current_solution.append(first_idx)
        covered_mask |= all_masks[first_idx]
        
        # Greedy: repeatedly add the word that covers the most uncovered words
        while covered_mask != (1 << total_words) - 1:
            if time.time() > deadline - 20:
                break
            
            uncovered_mask = ((1 << total_words) - 1) ^ covered_mask
            best_gain = -1
            best_idx = -1
            
            for i in range(total_words):
                if i in current_solution:
                    continue
                gain = bin(all_masks[i] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(best_idx)
            covered_mask |= all_masks[best_idx]
        
        # Local search: try to remove redundant codewords
        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            for i in range(len(current_solution)):
                # Check if removing current_solution[i] still covers everything
                remaining_mask = 0
                for j in range(len(current_solution)):
                    if j != i:
                        remaining_mask |= all_masks[current_solution[j]]
                if remaining_mask == (1 << total_words) - 1:
                    current_solution.pop(i)
                    improved = True
                    break
        
        # Additional local search: try swaps
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            if len(current_solution) < 2:
                break
            
            i = rng.randint(0, len(current_solution) - 1)
            # Try replacing current_solution[i] with a random other word
            new_idx = rng.randint(0, total_words - 1)
            if new_idx in current_solution:
                continue
            
            old_idx = current_solution[i]
            current_solution[i] = new_idx
            
            # Check if still valid
            new_mask = 0
            for j in range(len(current_solution)):
                new_mask |= all_masks[current_solution[j]]
            
            if new_mask != (1 << total_words) - 1:
                # Revert
                current_solution[i] = old_idx
            else:
                # Valid, keep it and try to remove redundant ones
                improved = True
                while improved:
                    improved = False
                    for k in range(len(current_solution)):
                        remaining_mask = 0
                        for j in range(len(current_solution)):
                            if j != k:
                                remaining_mask |= all_masks[current_solution[j]]
                        if remaining_mask == (1 << total_words) - 1:
                            current_solution.pop(k)
                            improved = True
                            break
        
        # Validate and update best
        final_mask = 0
        for idx in current_solution:
            final_mask |= all_masks[idx]
        
        if final_mask == (1 << total_words) - 1:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [_int_to_word(idx, q, n) for idx in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
