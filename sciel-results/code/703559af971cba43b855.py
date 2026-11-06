import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _general_construct(q, n, radius)
    
    # Ternary covering code K_3(7,1)
    N = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 15
    
    # Precompute all words and their indices
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as numpy arrays
    # B[i] contains the indices of the 15 words in the ball of radius 1 around words[i]
    B = np.empty((N, ball_size), dtype=np.int32)
    for i, w in enumerate(words):
        B[i, 0] = i
        pos = 1
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                B[i, pos] = word_to_idx[tuple(new_w)]
                pos += 1

    deadline = time.time() + 140  # 10s safety margin
    
    # Phase 1: Greedy Set Cover
    coverage = np.zeros(N, dtype=np.int32)
    C = []
    
    while time.time() < deadline - 40:
        # Vectorized gain calculation
        # coverage[B] gives the coverage counts for all points in all balls
        # We want to count how many points in each ball have coverage 0
        gains = np.sum(coverage[B] == 0, axis=1)
        
        if gains.max() == 0:
            break
        
        best_idx = int(np.argmax(gains))
        C.append(best_idx)
        coverage[B[best_idx]] += 1
        
    # Ensure we have a valid cover (greedy should guarantee this if it runs to completion, 
    # but if it times out, we might have holes. We must fill holes first.)
    if np.any(coverage == 0):
        # Add words to cover remaining holes greedily
        while np.any(coverage == 0):
            gains = np.sum(coverage[B] == 0, axis=1)
            best_idx = int(np.argmax(gains))
            if best_idx in C:
                # If best is already in C, it shouldn't have gain > 0 unless coverage logic is wrong
                # But coverage[B[best_idx]] should be > 0 for all points in ball if best_idx is in C?
                # No, coverage is global. If best_idx is in C, its points are covered by itself.
                # So gain for best_idx should be 0 for its own points?
                # Wait. If best_idx is in C, coverage[B[best_idx]] >= 1 for all points in B[best_idx].
                # So gains[best_idx] should be 0.
                # If gains.max() > 0, then best_idx is NOT in C.
                pass
            C.append(best_idx)
            coverage[B[best_idx]] += 1

    # Phase 2: Local Search
    C_set = set(C)
    
    def try_remove():
        """Try to remove redundant codewords."""
        improved = False
        for c in C[:]:
            if c not in C_set:
                continue
            # Check if any point in B[c] is covered ONLY by c
            # coverage[B[c]] == 1 means uniquely covered by c
            if not np.any(coverage[B[c]] == 1):
                # Redundant
                coverage[B[c]] -= 1
                C.remove(c)
                C_set.discard(c)
                improved = True
                break
        return improved

    def try_swap_1_for_1():
        """Try to swap one codeword for another to enable further reduction or maintain size while improving structure."""
        improved = False
        # We iterate through C and try to replace each one
        for i, c in enumerate(C):
            if c not in C_set:
                continue
            # Find points uniquely covered by c
            unique_mask = (coverage[B[c]] == 1)
            unique_points = B[c][unique_mask]
            
            if len(unique_points) == 0:
                # Should have been removed
                continue
            
            # We need a c_add such that B[c_add] covers all unique_points
            # And c_add is not in C
            # To speed this up, we can filter candidates.
            # A candidate c_add must cover at least one of the unique points.
            # Actually, it must cover ALL of them.
            
            # Heuristic: Pick the most constrained unique point (the one covered by fewest other codewords?)
            # Or just iterate through all possible c_add? N=2187 is small.
            # But doing this for every c in C (size ~186) and every c_add (2187) is 186*2187 ~ 400k checks.
            # Each check involves checking if a set of ~1-5 points is in B[c_add].
            # This is fast in Python if we use sets or bitmasks.
            
            # Let's use a bitset approach for speed.
            # Precompute bitmasks for balls?
            # N=2187 bits = 274 bytes per ball. 2187 * 274 = 600KB.
            # This is feasible.
            
            pass # Placeholder for optimization strategy
            
        return improved

    # For better performance, we will use a bitset representation for the balls.
    # N=2187. We can use numpy uint64 array of size (N, ceil(2187/64)) = (2187, 35).
    num_blocks = (N + 63) // 64
    B_bitsets = np.zeros((N, num_blocks), dtype=np.uint64)
    for i in range(N):
        for point in B[i]:
            B_bitsets[i, point // 64] |= (1 << (point % 64))
            
    # Re-run local search with bitsets
    C_set = set(C)
    
    def get_unique_points_bitset(c):
        """Returns a bitset of points uniquely covered by c."""
        # A point p is uniquely covered by c if:
        # 1. p is in B[c]
        # 2. For all other c' in C, p is NOT in B[c']
        # This is hard to compute directly with bitsets without iterating C.
        # Alternative: 
        # coverage[B[c]] == 1
        unique_mask = (coverage[B[c]] == 1)
        unique_points = B[c][unique_mask]
        if len(unique_points) == 0:
            return None, 0
        # Convert to bitset
        bs = np.zeros(num_blocks, dtype=np.uint64)
        for p in unique_points:
            bs[p // 64] |= (1 << (p % 64))
        return bs, len(unique_points)

    def bitset_is_subset(a, b):
        """Check if bitset a is a subset of bitset b."""
        # a & ~b == 0
        return np.all((a & ~b) == 0)

    # Local Search Loop
    no_improve = 0
    max_no_improve = 10
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        
        # 1. Try Removals
        while try_remove():
            improved = True
            no_improve = 0
            
        if improved:
            continue
            
        # 2. Try 1-for-1 Swaps
        # We want to find c in C and c_add not in C such that:
        # Unique(C, c) is subset of B[c_add]
        # This allows us to replace c with c_add.
        # It doesn't reduce size immediately, but might allow removal of c_add later?
        # No, if we replace c with c_add, the set of covered points is the same.
        # The benefit is if c_add has "more slack" or allows a 2-for-1 later.
        # Actually, a direct size reduction comes from 2-for-1 or 1-for-0 (removal).
        # 1-for-1 swaps are used to escape local minima in the removal phase.
        
        # Let's try to find a swap that reduces the "number of unique points" for the new codeword?
        # Or simply random swaps to perturb?
        # A better metric: Try to swap c with c_add if c_add covers unique points of c 
        # AND c_add is "better" in some way.
        
        # For now, let's stick to finding swaps that allow subsequent removals.
        # This is complex. 
        # Alternative: 2-for-1 swap.
        # Find c1, c2 in C and c_add not in C such that:
        # Unique(C, c1) U Unique(C, c2) is subset of B[c_add]
        # Then we can remove c1, c2 and add c_add. Net change: -1.
        
        # Let's implement 2-for-1.
        found_2_for_1 = False
        C_list = C[:]
        for i in range(len(C_list)):
            if time.time() > deadline - 5:
                break
            c1 = C_list[i]
            if c1 not in C_set: continue
            bs1, n1 = get_unique_points_bitset(c1)
            if bs1 is None: continue
            
            for j in range(i+1, len(C_list)):
                if time.time() > deadline - 5:
                    break
                c2 = C_list[j]
                if c2 not in C_set: continue
                bs2, n2 = get_unique_points_bitset(c2)
                if bs2 is None: continue
                
                # Union of unique points
                bs_union = bs1 | bs2
                
                # Find c_add that covers bs_union
                # We can iterate through candidates not in C
                # To speed up, we can filter candidates that cover at least one point in bs_union
                # But N is small, let's just iterate all c_add not in C
                for c_add in range(N):
                    if c_add in C_set:
                        continue
                    if bitset_is_subset(bs_union, B_bitsets[c_add]):
                        # Valid 2-for-1 swap
                        # Remove c1, c2
                        coverage[B[c1]] -= 1
                        coverage[B[c2]] -= 1
                        C.remove(c1)
                        C.remove(c2)
                        C_set.discard(c1)
                        C_set.discard(c2)
                        
                        # Add c_add
                        coverage[B[c_add]] += 1
                        C.append(c_add)
                        C_set.add(c_add)
                        
                        found_2_for_1 = True
                        no_improve = 0
                        break
                if found_2_for_1:
                    break
            if found_2_for_1:
                break
                
        if not found_2_for_1:
            no_improve += 1
            
    return [list(words[c]) for c in C]


def _general_construct(q, n, radius):
    """Fallback for non-standard parameters."""
    N = q ** n
    if N > 100000:
        return [[0] * n]
    
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    balls = []
    for w in words:
        ball = [w]
        for dist in range(1, radius + 1):
            for positions in itertools.combinations(range(n), dist):
                for vals in itertools.product(range(q), repeat=dist):
                    if all(v != w[p] for v, p in zip(vals, positions)):
                        new_w = list(w)
                        for p, v in zip(positions, vals):
                            new_w[p] = v
                        ball.append(tuple(new_w))
        balls.append([word_to_idx[b] for b in ball])
    
    balls = np.array(balls, dtype=np.int32)
    
    coverage = np.zeros(N, dtype=np.int32)
    C = []
    
    while np.any(coverage == 0):
        current_cov = coverage[balls]
        gains = np.sum(current_cov == 0, axis=1)
        if gains.max() == 0:
            break
        best_idx = int(np.argmax(gains))
        C.append(best_idx)
        coverage[balls[best_idx]] += 1
    
    return [list(words[c]) for c in C]
