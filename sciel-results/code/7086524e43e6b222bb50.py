"""Improved ternary covering code K_3(6,1) using simulated annealing."""
import itertools
import time
import random
import numpy as np


def _tetracode(q, n):
    """Generate tetracode-based initial code (9 * q^(n-4) words)."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _covered_count(code, q, n, radius=1):
    """Count how many words in the space are covered by the code."""
    # For radius 1, a word covers itself and its 2*n neighbors
    # We need to check coverage of all q^n words
    code_set = set(tuple(w) for w in code)
    total_space = q ** n
    covered = 0
    for word in itertools.product(range(q), repeat=n):
        word_t = tuple(word)
        if word_t in code_set:
            covered += 1
            continue
        # Check if within distance 1 of any codeword
        for i in range(n):
            for v in range(q):
                if v == word[i]:
                    continue
                neighbor = list(word_t)
                neighbor[i] = v
                if tuple(neighbor) in code_set:
                    covered += 1
                    break
            else:
                continue
            break
    return covered


def _is_covering(code, q, n, radius=1):
    """Check if code covers all words within radius."""
    code_set = set(tuple(w) for w in code)
    total_space = q ** n
    for word in itertools.product(range(q), repeat=n):
        word_t = tuple(word)
        if word_t in code_set:
            continue
        found = False
        for i in range(n):
            for v in range(q):
                if v == word[i]:
                    continue
                neighbor = list(word_t)
                neighbor[i] = v
                if tuple(neighbor) in code_set:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _fast_is_covering(code, q, n, radius=1):
    """Faster coverage check using numpy for the whole space."""
    code_arr = np.array(code, dtype=np.int8)
    code_set = set(map(tuple, code_arr))
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total = len(all_words)
    
    # Check each word
    uncovered = 0
    for i in range(total):
        w = tuple(all_words[i])
        if w in code_set:
            continue
        found = False
        for j in range(n):
            for v in range(q):
                if v == w[j]:
                    continue
                nb = list(w)
                nb[j] = v
                if tuple(nb) in code_set:
                    found = True
                    break
            if found:
                break
        if not found:
            uncovered += 1
    return uncovered == 0


def construct(q, n, radius):
    """Construct a covering code using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial code from tetracode
    code = _tetracode(q, n)
    best_code = list(code)
    best_size = len(code)
    
    # Verify initial code is valid
    if not _is_covering(code, q, n, radius):
        # Fallback: return all words (shouldn't happen)
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Simulated annealing: try to remove words
    # Temperature schedule
    T0 = 1.0
    T_final = 0.001
    cooling = 0.995
    
    T = T0
    attempts = 0
    max_attempts = 50000
    
    while T > T_final and time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try removing a random word
        idx = random.randint(0, len(code) - 1)
        removed = code.pop(idx)
        
        # Check if still covering
        if _is_covering(code, q, n, radius):
            # Accept: we removed a word
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
        else:
            # Reject: put word back
            code.insert(idx, removed)
        
        # Cool down
        T *= cooling
        
        # Periodic check for deadline
        if time.time() > deadline - 5.0:
            break
    
    return best_code
