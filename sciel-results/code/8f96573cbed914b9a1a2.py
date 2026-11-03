import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 120.0

    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline)

    # Generic fallback
    return _greedy_set_cover(q, n, radius, deadline)

def _solve_ternary_6_1(deadline):
    q, n, radius = 3, 6, 1
    total = q ** n  # 729
    
    # Precompute balls: list of lists of indices
    # ball[i] contains indices of words within distance 1 of i
    balls = []
    # Precompute conflict: conflict[i] is a set of indices j such that dist(i,j) == 2
    conflicts = [set() for _ in range(total)]
    
    # Helper: index to tuple
    def idx_to_tuple(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        return tuple(digits)
    
    # Helper: tuple to index
    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    # Precompute balls and conflicts
    for i in range(total):
        w_i = idx_to_tuple(i)
        ball_i = [i]
        for pos in range(n):
            for val in range(q):
                if val != w_i[pos]:
                    new_w = list(w_i)
                    new_w[pos] = val
                    j = tuple_to_idx(tuple(new_w))
                    ball_i.append(j)
                    # Conflict check: dist(i, j) == 2?
                    # We will compute dist in a separate pass or here.
                    # Actually, dist(i, j) is 1 for these.
                    # We need dist(i, j) == 2 for conflicts.
        balls.append(ball_i)
        
    # Compute conflicts (distance 2)
    # This is O(total * n * q) roughly if done smartly, or O(total^2) if brute force.
    # total = 729, total^2 = 531441. This is fast enough in Python if optimized or using numpy.
    # Let's use a vectorized approach or simple loops.
    # dist(i, j) == 2 means they differ in exactly 2 positions.
    for i in range(total):
        w_i = idx_to_tuple(i)
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                # Change pos1 and pos2
                for v1 in range(q):
                    if v1 == w_i[pos1]: continue
                    for v2 in range(q):
                        if v2 == w_i[pos2]: continue
                        new_w = list(w_i)
                        new_w[pos1] = v1
                        new_w[pos2] = v2
                        j = tuple_to_idx(tuple(new_w))
                        conflicts[i].add(j)
                        # Symmetry
                        conflicts[j].add(i)

    # Phase 1: Conflict-Aware Greedy Construction
    covered = np.zeros(total, dtype=bool)
    code_set = set()
    
    # Precompute gain for each word? No, gain changes as covered changes.
    # We will calculate gain on the fly.
    
    while not covered.all() and time.time() < deadline - 20.0:
        best_idx = -1
        best_gain = -1
        best_conflicts = total + 1
        
        # Optimization: Only check words that might add coverage?
        # All words add at least 1 coverage (themselves) if not covered, but if covered, they might add 0.
        # Actually, if a word is already covered, it might still add coverage to others.
        
        for idx in range(total):
            if idx in code_set:
                continue
            
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            
            # If gain is 0, skip
            if gain == 0:
                continue
            
            # Count conflicts with existing code
            conf_count = 0
            for c in code_set:
                if c in conflicts[idx]:
                    conf_count += 1
            
            # Selection criterion:
            # 1. Maximize gain
            # 2. Minimize conflicts
            if gain > best_gain or (gain == best_gain and conf_count < best_conflicts):
                best_gain = gain
                best_conflicts = conf_count
                best_idx = idx
        
        if best_idx < 0:
            break
            
        code_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True

    # Phase 2: Local Search to reduce size
    # We have a valid cover. We want to minimize |code_set|.
    # Strategy: Simulated Annealing or Greedy Removal + Swaps.
    # Let's use a robust Local Search:
    # 1. Greedy Removal: Remove any codeword that doesn't break coverage.
    # 2. Randomized Swap: Try to replace a codeword with another to enable removal of a different one.
    
    # Precompute coverage counts for fast removal checks
    # coverage_count[i] = number of codewords in ball(i)
    coverage_count = np.zeros(total, dtype=np.int32)
    for idx in code_set:
        for b_idx in balls[idx]:
            coverage_count[b_idx] += 1

    def is_redundant(idx):
        # A codeword is redundant if all words in its ball are covered by other codewords.
        # i.e., coverage_count[b] >= 2 for all b in ball(idx)?
        # No, coverage_count[b] includes idx. So if coverage_count[b] >= 2 for all b in ball(idx),
        # then removing idx leaves coverage_count[b] >= 1 for all b.
        # So yes, if min(coverage_count[ball(idx)]) >= 2, it is redundant.
        for b_idx in balls[idx]:
            if coverage_count[b_idx] == 1:
                return False
        return True

    def remove_codeword(idx):
        for b_idx in balls[idx]:
            coverage_count[b_idx] -= 1
        code_set.remove(idx)

    def add_codeword(idx):
        for b_idx in balls[idx]:
            coverage_count[b_idx] += 1
        code_set.add(idx)

    # Iterative improvement
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # 1. Greedy Removal Pass
        for idx in list(code_set):
            if is_redundant(idx):
                remove_codeword(idx)
                improved = True
        
        # 2. Randomized Local Search (Swaps)
        # Try to swap a codeword u with v to see if it allows a reduction.
        # Or just try to find a valid set of size K-1 by random restarts?
        # Let's try a simple hill climbing on the number of codewords.
        # We will try to remove a codeword u, and then repair the uncovered words.
        
        if time.time() > deadline - 15.0:
            break

        # Try 100 random removal-repair attempts
        for _ in range(100):
            if not code_set:
                break
            u = random.choice(list(code_set))
            
            # Temporarily remove u
            # Find words that become uncovered
            uncovered = []
            for b_idx in balls[u]:
                if coverage_count[b_idx] == 1:
                    uncovered.append(b_idx)
            
            if not uncovered:
                # u was redundant, should have been caught by greedy removal
                remove_codeword(u)
                improved = True
                break
            
            # We need to cover 'uncovered' with new codewords.
            # 'uncovered' is a small set.
            # Try to find a single codeword v that covers all 'uncovered'.
            # If found, swap u with v. Size unchanged.
            # But maybe v allows removing another codeword w?
            
            # Let's try to find a set of 1 or 2 codewords that cover 'uncovered'.
            # Since 'uncovered' is small, we can brute force candidates.
            
            # Candidate v must be in the intersection of balls of all words in 'uncovered'?
            # No, v must cover each word in 'uncovered'.
            # So v must be in ball(w) for all w in 'uncovered'.
            
            candidates = set(balls[uncovered[0]])
            for w in uncovered[1:]:
                candidates.intersection_update(balls[w])
            
            # Filter candidates: must not be in code_set (or if it is, it's a no-op swap)
            # Actually, if v is in code_set, it's already there.
            
            found_swap = False
            for v in candidates:
                if v == u:
                    continue
                if v in code_set:
                    continue
                
                # If we add v, does it allow removal of some other codeword?
                # Or simply: does {code_set - {u} + {v}} cover everything?
                # Yes, by construction, v covers 'uncovered', and others cover the rest.
                # So the new set is valid.
                
                # Does this help reduce size?
                # Only if v allows removing another codeword w.
                # Let's check if any w in code_set - {u} becomes redundant after adding v.
                
                # Temporarily apply swap
                # Remove u
                for b_idx in balls[u]:
                    coverage_count[b_idx] -= 1
                code_set.remove(u)
                
                # Add v
                for b_idx in balls[v]:
                    coverage_count[b_idx] += 1
                code_set.add(v)
                
                # Check for redundancy in the new set
                can_remove_more = False
                for w in list(code_set):
                    if is_redundant(w):
                        can_remove_more = True
                        break
                
                if can_remove_more:
                    # Accept this swap and then do a greedy removal pass
                    # The 'improved' flag will be set by the outer loop's greedy pass
                    # But we need to signal that we changed the set
                    improved = True
                    found_swap = True
                    break
                else:
                    # Revert swap
                    for b_idx in balls[v]:
                        coverage_count[b_idx] -= 1
                    code_set.remove(v)
                    
                    for b_idx in balls[u]:
                        coverage_count[b_idx] += 1
                    code_set.add(u)
            
            if found_swap:
                break

    # Final Greedy Removal Pass
    for idx in list(code_set):
        if is_redundant(idx):
            remove_codeword(idx)

    # Convert back to lists
    result = []
    for idx in code_set:
        result.append(list(idx_to_tuple(idx)))
    
    return result

def _greedy_set_cover(q, n, radius, deadline):
    total = q ** n

    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        return tuple(digits)

    balls = []
    for i in range(total):
        w = idx_to_tuple(i)
        ball = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(tuple(new_w))
        idxs = [tuple_to_idx(b) for b in ball]
        balls.append(idxs)

    covered = np.zeros(total, dtype=bool)
    codeword_set = set()

    while not covered.all() and time.time() < deadline - 5.0:
        best_idx = -1
        best_gain = -1
        for idx in range(total):
            if idx in codeword_set:
                continue
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0:
            break
        codeword_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True

    result = []
    for idx in codeword_set:
        result.append(list(idx_to_tuple(idx)))
    return result
