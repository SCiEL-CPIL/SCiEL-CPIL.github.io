"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

_START_TIME = None
_DEADLINE_FRAC = 0.85  # safety margin: stop at 85% of 120s


def _elapsed():
    return time.time() - _START_TIME


def _within_deadline(margin=0.0):
    if _START_TIME is None:
        return True
    limit = 120.0 * _DEADLINE_FRAC
    return (time.time() - _START_TIME) < (limit - margin)


def _tetracode_sum(q, n, radius):
    """Direct sum of ternary tetracode with all tails: 9 * 3**(n-4) codewords."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _ball_indices(codeword, q, n, radius):
    """Yield the flat index of every word within Hamming distance <= radius of codeword."""
    codeword = tuple(codeword)
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for vals in itertools.product(range(1, q), repeat=r):
                w = list(codeword)
                for pos, val in zip(positions, vals):
                    w[pos] = (codeword[pos] + val) % q
                yield _flat(w, q, n)


def _flat(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _uncovered_mask(codewords, q, n, radius):
    """Return a boolean numpy array of size q**n; True = covered."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        for idx in _ball_indices(cw, q, n, radius):
            covered[idx] = True
    return covered


def _is_valid(codewords, q, n, radius):
    covered = _uncovered_mask(codewords, q, n, radius)
    return covered.all()


def _greedy_construct(q, n, radius):
    """Greedy set-cover: repeatedly pick the codeword covering the most uncovered words."""
    total = q ** n
    # Precompute all possible codewords and their balls
    all_cw = [list(w) for w in itertools.product(range(q), repeat=n)]
    all_balls = [set(_ball_indices(cw, q, n, radius)) for cw in all_cw]
    
    uncovered = set(range(total))
    codewords = []
    
    while uncovered and _within_deadline():
        best_i = -1
        best_count = 0
        for i, ball in enumerate(all_balls):
            if i in codewords:
                continue
            count = len(ball & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
            if best_count == len(uncovered):
                break
        if best_i < 0 or best_count == 0:
            break
        codewords.append(all_cw[best_i])
        uncovered -= all_balls[best_i]
    
    return codewords


def _local_search(codewords, q, n, radius, max_iters=500):
    """Local search: try removing redundant codewords and add-then-remove swaps."""
    codewords = [list(cw) for cw in codewords]
    total = q ** n
    
    if not codewords:
        return codewords
    
    # Precompute balls for current codewords
    balls = [set(_ball_indices(cw, q, n, radius)) for cw in codewords]
    
    # Compute coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for ball in balls:
        for idx in ball:
            coverage[idx] += 1
    
    # Track which codewords are in the set for quick lookup
    current_set = set(tuple(cw) for cw in codewords)
    
    # All possible words not in current set
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    candidates = [w for w in all_words if w not in current_set]
    
    rng = random.Random(123)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _within_deadline(margin=2.0):
        improved = False
        iters += 1
        
        # Phase 1: Try removing each codeword if it doesn't break coverage
        for i in range(len(codewords) - 1, -1, -1):
            if not _within_deadline(margin=2.0):
                break
            can_remove = True
            for idx in balls[i]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for idx in balls[i]:
                    coverage[idx] -= 1
                del codewords[i]
                del balls[i]
                current_set.discard(tuple(codewords[i]) if i < len(codewords) else tuple())
                # Rebuild candidates since set changed
                current_set = set(tuple(cw) for cw in codewords)
                candidates = [w for w in all_words if w not in current_set]
                improved = True
                break
        
        # Phase 2: Add-then-remove: add a random candidate, then try to remove one or more
        if not improved and len(codewords) > 1 and candidates:
            if not _within_deadline(margin=2.0):
                break
            
            # Try several random candidates
            attempts = 0
            max_attempts = min(20, len(candidates))
            while attempts < max_attempts and _within_deadline(margin=2.0):
                attempts += 1
                cand_idx = rng.randint(0, len(candidates) - 1)
                cand_tuple = candidates[cand_idx]
                cand_ball = set(_ball_indices(list(cand_tuple), q, n, radius))
                
                # Tentatively add the candidate
                coverage_copy = coverage.copy()
                for idx in cand_ball:
                    coverage_copy[idx] += 1
                
                # Now try to remove codewords in random order
                removed_any = False
                remove_order = list(range(len(codewords)))
                rng.shuffle(remove_order)
                
                for j in remove_order:
                    if not _within_deadline(margin=2.0):
                        break
                    can_remove = True
                    for idx in balls[j]:
                        if coverage_copy[idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove codeword j
                        for idx in balls[j]:
                            coverage_copy[idx] -= 1
                        removed_any = True
                        break
                
                if removed_any:
                    # Commit the changes: add cand, remove codeword j
                    # First, add the candidate
                    for idx in cand_ball:
                        coverage[idx] += 1
                    codewords.append(list(cand_tuple))
                    balls.append(cand_ball)
                    current_set.add(cand_tuple)
                    
                    # Then remove codeword j
                    for idx in balls[j]:
                        coverage[idx] -= 1
                    del codewords[j]
                    del balls[j]
                    current_set.discard(tuple(codewords[j]) if j < len(codewords) else tuple())
                    
                    # Rebuild candidates
                    current_set = set(tuple(cw) for cw in codewords)
                    candidates = [w for w in all_words if w not in current_set]
                    improved = True
                    break
    
    return codewords


def _swap_local_search_sa(codewords, q, n, radius, max_iters=2000, initial_temp=10.0, cooling=0.999):
    """Add-then-remove local search: add a candidate codeword, then remove redundant ones."""
    codewords = [list(cw) for cw in codewords]
    total = q ** n
    
    if not codewords:
        return codewords
    
    # Precompute balls for current codewords
    balls = [set(_ball_indices(cw, q, n, radius)) for cw in codewords]
    
    # Compute coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for ball in balls:
        for idx in ball:
            coverage[idx] += 1
    
    # Verify initial solution is valid
    if not (coverage >= 1).all():
        return codewords
    
    # Generate all possible codewords not currently in the set
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    current_set = set(tuple(cw) for cw in codewords)
    candidates = [w for w in all_words if w not in current_set]
    
    if not candidates:
        return codewords
    
    # Precompute balls for all candidates
    candidate_balls_cache = {}
    
    def get_candidate_ball(cw_tuple):
        if cw_tuple not in candidate_balls_cache:
            candidate_balls_cache[cw_tuple] = set(_ball_indices(list(cw_tuple), q, n, radius))
        return candidate_balls_cache[cw_tuple]
    
    best_codewords = [list(cw) for cw in codewords]
    best_len = len(codewords)
    current_len = len(codewords)
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if not _within_deadline(margin=5.0):
            break
        
        # Pick a random candidate to add
        cand_idx = rng.randint(0, len(candidates) - 1)
        cand_tuple = candidates[cand_idx]
        cand_ball = get_candidate_ball(cand_tuple)
        
        # Tentatively add the candidate
        coverage_copy = coverage.copy()
        for idx in cand_ball:
            coverage_copy[idx] += 1
        
        # Try to remove codewords in random order until we can't anymore
        remove_order = list(range(len(codewords)))
        rng.shuffle(remove_order)
        
        removed_indices = []
        for j in remove_order:
            if not _within_deadline(margin=5.0):
                break
            can_remove = True
            for idx in balls[j]:
                if coverage_copy[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for idx in balls[j]:
                    coverage_copy[idx] -= 1
                removed_indices.append(j)
        
        # If we removed at least one codeword, commit the change
        if removed_indices:
            # Add the candidate
            for idx in cand_ball:
                coverage[idx] += 1
            codewords.append(list(cand_tuple))
            balls.append(cand_ball)
            current_set.add(cand_tuple)
            
            # Remove the redundant codewords (in reverse order to keep indices valid)
            for j in sorted(removed_indices, reverse=True):
                for idx in balls[j]:
                    coverage[idx] -= 1
                del codewords[j]
                del balls[j]
                current_set.discard(tuple(codewords[j]) if j < len(codewords) else tuple())
            
            current_len = len(codewords)
            
            if current_len < best_len:
                best_len = current_len
                best_codewords = [list(cw) for cw in codewords]
            
            # Rebuild candidates
            current_set = set(tuple(cw) for cw in codewords)
            candidates = [w for w in all_words if w not in current_set]
            if not candidates:
                break
    
    return best_codewords


def _greedy_then_local(q, n, radius):
    """Start from tetracode sum, then apply local search."""
    # Start from the 81-word tetracode sum
    codewords = _tetracode_sum(q, n, radius)
    
    # Apply local search to reduce
    codewords = _local_search(codewords, q, n, radius)
    
    return codewords


def _greedy_with_removal(q, n, radius):
    """Greedy construction followed by removal-based local search."""
    # First, build a greedy solution
    codewords = _greedy_construct(q, n, radius)
    
    # Then apply local search
    codewords = _local_search(codewords, q, n, radius)
    
    return codewords


def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    # For the specific case q=3, n=6, radius=1, use the known structure
    if q == 3 and n == 6 and radius == 1:
        # Start from tetracode sum (81 words) and try to reduce to 73
        best = _tetracode_sum(q, n, radius)
        best_len = len(best)
        
        # Try local search starting from tetracode sum
        if _within_deadline(margin=10.0):
            candidate = _local_search(best, q, n, radius)
            if len(candidate) < best_len:
                best = candidate
                best_len = len(candidate)
        
        # Try add-then-remove local search starting from tetracode sum
        if _within_deadline(margin=10.0):
            candidate = _swap_local_search_sa(best, q, n, radius)
            if len(candidate) < best_len:
                best = candidate
                best_len = len(candidate)
        
        # Try greedy construction + local search
        if _within_deadline(margin=10.0):
            candidate = _greedy_with_removal(q, n, radius)
            if len(candidate) < best_len:
                best = candidate
                best_len = len(candidate)
        
        # Try add-then-remove on greedy result
        if _within_deadline(margin=5.0):
            candidate = _swap_local_search_sa(best, q, n, radius)
            if len(candidate) < best_len:
                best = candidate
                best_len = len(candidate)
        
        # Verify validity
        if _is_valid(best, q, n, radius):
            return best
        
        # Fallback: return the tetracode sum (known valid)
        return _tetracode_sum(q, n, radius)
    else:
        # General case: use greedy construction
        codewords = _greedy_construct(q, n, radius)
        if _is_valid(codewords, q, n, radius):
            return codewords
        # Fallback: brute-force all words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
