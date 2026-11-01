import time
import itertools
import numpy as np
import random

_DEADLINE = 140.0

def _direct_sum(q, n, radius):
    """Always-valid baseline: a radius-1 perfect code of length 4 over the alphabet,
    tensored with every word on the remaining n-4 coordinates.  Valid for R>=1, n>=4.
    Size = q^2 * q^(n-4) = q^(n-2)."""
    # Perfect (q^2, 4, 1) covering: positions 0,1 free; 2 = a+b; 3 = a+2b (mod q).
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]

def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx

def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space (only used for small spaces)."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        # Generate ball
        ball = [c]
        for d in range(1, radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = list(c)
                    for p, v in zip(pos, vals):
                        w[p] = v
                    ball.append(w)
        for w in ball:
            covered[_idx(w, q)] = True
    return covered.all()

def _greedy_with_local_search(q, n, radius, time_budget, rng):
    """Greedy set-cover with group-structured local search (ILS).
    Returns (codewords, verified)."""
    total = q ** n
    if total > 3 * 10**6:
        return None, False

    start = time.time()
    deadline = start + time_budget

    # Precompute balls and reverse balls
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Precompute ball membership as a list of lists of indices
    balls = []
    reverse_balls = [[] for _ in range(total)]
    
    def get_ball(center):
        idx = _idx(center, q)
        ball = [idx]
        for d in range(1, radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = list(center)
                    for p, v in zip(pos, vals):
                        w[p] = v
                    ball.append(_idx(w, q))
        return ball

    for i in range(total):
        b = get_ball(all_words[i])
        balls.append(b)
        for w in b:
            reverse_balls[w].append(i)
            
    covered = np.zeros(total, dtype=bool)
    ball_size = len(balls[0])
    gain = np.full(total, ball_size, dtype=np.int32)
    
    code_indices = []
    
    while not covered.all():
        if time.time() > deadline:
            return None, False
        
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        candidates = np.where(gain == max_gain)[0]
        best = rng.choice(candidates)
        
        code_indices.append(best)
        
        b = balls[best]
        for w in b:
            if not covered[w]:
                covered[w] = True
                for c in reverse_balls[w]:
                    gain[c] -= 1
                    
    if not covered.all():
        return None, False
        
    # Build coverage count
    code_set = set(code_indices)
    cover_count = np.zeros(total, dtype=np.int32)
    for c_idx in code_set:
        for w in balls[c_idx]:
            cover_count[w] += 1
            
    if not (cover_count > 0).all():
        return None, False
        
    # Group-structured perturbations: short vectors in Z_q^n
    # For q=3: single-position changes (±1, ±2 mod 3) and double-position changes
    def short_vectors(q, n, max_dist=2):
        vecs = []
        for d in range(1, max_dist + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    v = [0] * n
                    for p, val in zip(pos, vals):
                        v[p] = val
                    vecs.append(tuple(v))
        return vecs
    
    all_short_vecs = short_vectors(q, n, max_dist=2)
    
    def idx_from_word(w):
        return _idx(w, q)
    
    def move_center(c_idx, vec):
        """Move codeword c_idx by vector vec (mod q), return new index."""
        new_word = [(all_words[c_idx][p] + vec[p]) % q for p in range(n)]
        return idx_from_word(new_word)
    
    # ILS: iteratively try to move or remove codewords
    best_size = len(code_set)
    best_code_set = set(code_set)
    best_cover_count = cover_count.copy()
    
    current_code_set = set(code_set)
    current_cover_count = cover_count.copy()
    
    # Prune first
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_code = list(current_code_set)
        rng.shuffle(current_code)
        for c_idx in current_code:
            if time.time() > deadline:
                break
            is_essential = False
            for w in balls[c_idx]:
                if current_cover_count[w] == 1:
                    is_essential = True
                    break
            if not is_essential:
                current_code_set.remove(c_idx)
                for w in balls[c_idx]:
                    current_cover_count[w] -= 1
                changed = True
                if not (current_cover_count > 0).all():
                    return None, False
    
    if len(current_code_set) < best_size:
        best_size = len(current_code_set)
        best_code_set = set(current_code_set)
        best_cover_count = current_cover_count.copy()
    
    # ILS moves: try moving a codeword to a nearby position
    while time.time() < deadline:
        if len(current_code_set) == 0:
            break
        
        # Pick a random codeword to try moving
        current_code_list = list(current_code_set)
        c_idx = rng.choice(current_code_list)
        
        # Try all short vectors
        found_move = False
        for vec in all_short_vecs:
            new_idx = move_center(c_idx, vec)
            
            # Skip if new position already in code set
            if new_idx in current_code_set:
                continue
            
            # Check if removing c_idx and adding new_idx keeps coverage valid
            # First, check if c_idx is essential (covers a word uniquely)
            # After removal, those words must be covered by new_idx or others
            
            # Compute what would be uncovered if we remove c_idx
            # and what new_idx would cover
            old_ball = balls[c_idx]
            new_ball = balls[new_idx]
            
            # Check: for each word in old_ball not in new_ball, is it still covered by another codeword?
            valid = True
            for w in old_ball:
                if current_cover_count[w] == 1:
                    # This word is only covered by c_idx
                    # It must be covered by new_idx
                    if w not in new_ball:
                        valid = False
                        break
            if not valid:
                continue
            
            # Also check: new_idx's ball words must all be covered after the move
            # (they were covered before, and we're adding new_idx, so they're fine)
            
            # Perform the move
            current_code_set.remove(c_idx)
            for w in old_ball:
                current_cover_count[w] -= 1
            current_code_set.add(new_idx)
            for w in new_ball:
                current_cover_count[w] += 1
            
            found_move = True
            
            # Verify coverage
            if not (current_cover_count > 0).all():
                # Roll back
                current_code_set.remove(new_idx)
                for w in new_ball:
                    current_cover_count[w] -= 1
                current_code_set.add(c_idx)
                for w in old_ball:
                    current_cover_count[w] += 1
                continue
            
            # Prune after move
            changed2 = True
            while changed2 and time.time() < deadline:
                changed2 = False
                current_code_list2 = list(current_code_set)
                rng.shuffle(current_code_list2)
                for c2 in current_code_list2:
                    if time.time() > deadline:
                        break
                    is_ess = False
                    for w in balls[c2]:
                        if current_cover_count[w] == 1:
                            is_ess = True
                            break
                    if not is_ess:
                        current_code_set.remove(c2)
                        for w in balls[c2]:
                            current_cover_count[w] -= 1
                        changed2 = True
                        if not (current_cover_count > 0).all():
                            return None, False
            
            if len(current_code_set) < best_size:
                best_size = len(current_code_set)
                best_code_set = set(current_code_set)
                best_cover_count = current_cover_count.copy()
            
            break  # One successful move per iteration
        
        if not found_move:
            # Try a removal-only pass
            pass
    
    # Use best found
    code_set = best_code_set
    if len(code_set) == 0:
        return None, False
        
    # Convert indices to words
    final_code = [all_words[i].tolist() for i in code_set]
    
    # Final verification
    if not _covers_all(final_code, q, n, radius):
        return None, False
        
    return final_code, True


def construct(q, n, radius):
    start = time.time()
    # Always-valid baseline (used as the guaranteed fallback).
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        # Degenerate small cases: cover every word directly (exact, valid).
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]

    best = baseline
    best_len = len(baseline)

    # Try to improve with the greedy set-cover, only when the space is tractable.
    if q ** n <= 3 * 10**6 and best_len > 1:
        time_budget_total = max(0.5, _DEADLINE - (time.time() - start) - 5.0)
        
        # Fewer restarts but longer time per restart for deeper ILS
        num_restarts = 10
        time_per_restart = time_budget_total / num_restarts
        
        rng = random.Random(42)
        
        for i in range(num_restarts):
            if time.time() - start > _DEADLINE - 5.0:
                break
            try:
                cand, verified = _greedy_with_local_search(q, n, radius, time_per_restart, rng)
            except Exception:
                cand, verified = None, False
            if cand is not None and verified and len(cand) < best_len:
                best = cand
                best_len = len(cand)
                if best_len == 73:
                    break

    # Final safety: if we are running out of time, return the valid baseline.
    if time.time() - start > _DEADLINE - 5.0:
        return baseline

    return best
