# EVOLVE-BLOCK-START
import itertools
import time
import random

def _seed_code(q, n, radius):
    """Direct sum construction: tetracode x all words on remaining coords."""
    if n < 4:
        # Fallback: brute force all words (small space)
        return [list(c) for c in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n == 4:
        return tetracode
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

def _neighbors(x, q, n, radius=1):
    """Generate all words within Hamming distance radius from x."""
    # For radius=1, this is just x plus single-position changes
    yield tuple(x)
    for i in range(n):
        for v in range(q):
            if v != x[i]:
                y = list(x)
                y[i] = v
                yield tuple(y)

def _is_covering(code, q, n, radius):
    """Check if code covers all words."""
    if not code:
        return False
    covered = set()
    for c in code:
        for nb in _neighbors(c, q, n, radius):
            covered.add(nb)
    return len(covered) == q ** n

def _greedy_init(q, n, radius, deadline):
    """Greedy set cover to get a starting solution."""
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    uncovered = set(all_words)
    code = []
    
    # Precompute neighbors for each possible word
    # This is expensive for large spaces, but for 3^6=729 it's fine
    neighbor_map = {}
    for w in all_words:
        neighbor_map[w] = set(_neighbors(w, q, n, radius))
    
    while uncovered and time.time() < deadline:
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        for w in all_words:
            count = len(neighbor_map[w] & uncovered)
            if count > best_count:
                best_count = count
                best_word = w
            # Early termination if we found perfect coverage
            if best_count == len(uncovered):
                break
        if best_word is None:
            break
        code.append(list(best_word))
        uncovered -= neighbor_map[best_word]
    
    return code

def _local_search(code, q, n, radius, deadline):
    """Local search: try removing redundant codewords and swapping."""
    code = [list(c) for c in code]
    all_words = set(tuple(w) for w in itertools.product(range(q), repeat=n))
    
    # Precompute neighbors
    neighbor_map = {}
    for w in all_words:
        neighbor_map[w] = set(_neighbors(list(w), q, n, radius))
    
    while time.time() < deadline:
        improved = False
        
        # Try removing each codeword
        for i in range(len(code)):
            if time.time() > deadline:
                break
            trial = code[:i] + code[i+1:]
            if not trial:
                continue
            covered = set()
            for c in trial:
                covered |= neighbor_map[tuple(c)]
            if covered == all_words:
                code = trial
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: replace one codeword with another
        if len(code) > 1:
            # Sample random pairs
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = random.randint(0, len(code) - 1)
                # Try a random new word
                new_word = list(random.randint(0, q-1) for _ in range(n))
                trial = code[:i] + [new_word] + code[i+1:]
                covered = set()
                for c in trial:
                    covered |= neighbor_map[tuple(c)]
                if covered == all_words:
                    code = trial
                    improved = True
                    break
        
        if not improved:
            # Stuck, try a more aggressive restart with perturbation
            if len(code) > 5:
                # Remove 2 random codewords and re-add greedily
                remove_idx = random.sample(range(len(code)), 2)
                for idx in sorted(remove_idx, reverse=True):
                    code.pop(idx)
                # Re-add greedily
                uncovered = all_words
                for c in code:
                    uncovered -= neighbor_map[tuple(c)]
                while uncovered:
                    best_word = None
                    best_count = -1
                    for w in list(uncovered):
                        count = len(neighbor_map[w] & uncovered)
                        if count > best_count:
                            best_count = count
                            best_word = w
                    if best_word is None:
                        break
                    code.append(list(best_word))
                    uncovered -= neighbor_map[best_word]
    
    return code

def _solve_3_6_1(deadline):
    """Solve the specific 3,6,1 case."""
    q, n, radius = 3, 6, 1
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute neighbor map
    neighbor_map = {}
    for w in all_words:
        neighbor_map[w] = set(_neighbors(list(w), q, n, radius))
    
    # Start with seed code
    code = _seed_code(q, n, radius)
    best_code = [list(c) for c in code]
    best_len = len(best_code)
    
    # Phase 1: Greedy initialization to potentially get better start
    greedy_code = _greedy_init(q, n, radius, min(deadline, time.time() + 10))
    if greedy_code and len(greedy_code) < best_len:
        best_code = [list(c) for c in greedy_code]
        best_len = len(best_code)
    
    # Phase 2: Local search
    if time.time() < deadline:
        code = _local_search(best_code, q, n, radius, deadline)
        if len(code) < best_len:
            best_code = [list(c) for c in code]
            best_len = len(code)
    
    return best_code

def _greedy_cover(q, n, radius, deadline):
    """Generic greedy set cover for other parameter settings."""
    return _greedy_init(q, n, radius, deadline)

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # For the specific task parameters
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1(deadline)
    else:
        # Generic fallback
        return _greedy_cover(q, n, radius, deadline)
# EVOLVE-BLOCK-END
