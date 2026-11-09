"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product


def _to_int(w, q):
    v = 0
    for x in w:
        v = v * q + x
    return v


def _to_word(i, q, n):
    w = [0] * n
    for j in range(n - 1, -1, -1):
        w[j] = i % q
        i //= q
    return w


def _build_incidence(q, n, radius):
    """
    Build a boolean incidence matrix M of shape (total_space, total_space)
    where M[cw][w] = True iff Hamming distance(cw, w) <= radius.
    Also return the neighbor lists as a list of arrays for faster iteration.
    """
    total_space = q ** n
    # Precompute all offset patterns for Hamming distance <= radius
    # An offset pattern is a tuple of (pos, delta) pairs where delta is in 1..q-1
    # For radius r, we have C(n, r) position subsets, and for each, (q-1)^r value changes
    
    # We'll represent each neighbor as an integer index
    # Build list of all possible "delta vectors" as sparse representations
    # Actually, let's just build the neighbor list for each codeword directly
    # using the precomputed patterns
    
    # Precompute all patterns: list of (positions_tuple, deltas_tuple)
    patterns = [([], ())]  # radius 0
    for r in range(1, radius + 1):
        for pos in combinations(range(n), r):
            for deltas in product(range(1, q), repeat=r):
                patterns.append((pos, deltas))
    
    num_patterns = len(patterns)
    
    # For each codeword cw (as integer), compute all neighbor integers
    # neighbor_map[cw] = array of neighbor integers
    neighbor_map = [None] * total_space
    for cw in range(total_space):
        w = _to_word(cw, q, n)
        nbrs = set()
        for pos, deltas in patterns:
            nw = list(w)
            for p, d in zip(pos, deltas):
                nw[p] = (w[p] + d) % q
            ni = _to_int(nw, q)
            nbrs.add(ni)
        neighbor_map[cw] = np.array(sorted(nbrs), dtype=np.int32)
    
    return neighbor_map


def _direct_sum_code(q, n, radius):
    """
    Construct the direct-sum covering code.
    For q=3, n=8, radius=1: use two blocks of length 4.
    The (4,1) ternary covering code has 9 words (the complete code of length 4 over 3 symbols
    with radius 1 is actually not 9... let me think).
    
    Actually, K_3(4,1) = 9? Let's verify: 3^4 = 81 words. Each codeword covers 
    1 + 4*2 = 9 words. So lower bound is 81/9 = 9. And indeed the complete code works:
    all words of the form (a, a, a, a) for a in {0,1,2} plus... no that's only 3.
    
    Actually for K_3(4,1): the code {0000, 1111, 2222, 0120, 1021, 2102, ...} 
    Hmm, let me just use a known construction.
    
    The direct sum: if C1 is a (n1, r1) code and C2 is a (n2, r2) code with r1+r2=r,
    then C1 x C2 is a (n1+n2, r) code with |C1|*|C2| words.
    
    For K_3(8,1): we can use K_3(4,0) x K_3(4,1) or K_3(4,1) x K_3(4,1).
    K_3(4,0) = 3^4 = 81 (trivial, radius 0 means each word covers only itself)
    K_3(4,1) = 9 (I believe this is correct: 81/9=9 and the code exists)
    
    So K_3(8,1) <= 81 * 9 = 729? That's worse than 486.
    
    Better: K_3(4,1) x K_3(4,1) gives 9*9=81 words for radius 2, not radius 1.
    
    For radius 1: we need r1+r2=1, so one is 0 and one is 1.
    K_3(4,0) * K_3(4,1) = 81 * 9 = 729.
    
    Hmm, that's not 486. Let me reconsider.
    
    Actually, the 486-word code by van Laarhoven et al. might not be a simple direct sum.
    Let me just use a greedy approach with a good initial solution.
    
    For now, let me just return an empty list and rely on greedy.
    """
    return []


def _greedy_cover(neighbor_map, total_space, deadline, seed=42):
    """Greedy set cover with efficient updates."""
    rng = random.Random(seed)
    counts = np.zeros(total_space, dtype=np.int32)
    uncovered_count = np.zeros(total_space, dtype=np.int32)
    for cw in range(total_space):
        uncovered_count[cw] = len(neighbor_map[cw])
    
    codewords = []
    
    while True:
        if time.time() >= deadline - 5.0:
            break
        
        best_cw = int(np.argmax(uncovered_count))
        best_count = uncovered_count[best_cw]
        if best_count == 0:
            break
        
        codewords.append(best_cw)
        
        for w in neighbor_map[best_cw]:
            if counts[w] == 0:
                counts[w] += 1
                for cw2 in neighbor_map[w]:
                    uncovered_count[cw2] -= 1
    
    return codewords


def _local_search_reduce(codewords, neighbor_map, total_space, deadline):
    """Try to remove redundant codewords."""
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        counts = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            for w in neighbor_map[cw]:
                counts[w] += 1
        
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() >= deadline - 2.0:
                break
            cw = codewords[i]
            removable = True
            for w in neighbor_map[cw]:
                if counts[w] <= 1:
                    removable = False
                    break
            if removable:
                codewords.pop(i)
                changed = True
                break
    return codewords


def _local_search_swap(codewords, neighbor_map, total_space, deadline, max_iterations=200):
    """
    1-exchange local search: try to replace a codeword with another that
    covers at least as many critical (singly-covered) words.
    """
    rng = random.Random(12345)
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 3.0 and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Compute coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for cw in codewords:
            for w in neighbor_map[cw]:
                counts[w] += 1
        
        # Find critical words (covered exactly once)
        critical_words = np.where(counts == 1)[0]
        
        if len(critical_words) == 0:
            # No critical words, try removal
            for i in range(len(codewords) - 1, -1, -1):
                cw = codewords[i]
                removable = True
                for w in neighbor_map[cw]:
                    if counts[w] <= 1:
                        removable = False
                        break
                if removable:
                    codewords.pop(i)
                    improved = True
                    break
            continue
        
        # For each codeword, find which critical words it covers
        # Try to find a swap that improves things
        for i in range(len(codewords)):
            if time.time() >= deadline - 3.0:
                break
            cw = codewords[i]
            # Critical words covered by this codeword
            cw_critical = [w for w in neighbor_map[cw] if counts[w] == 1]
            
            if len(cw_critical) == 0:
                # This codeword doesn't cover any critical words, try to remove it
                removable = True
                for w in neighbor_map[cw]:
                    if counts[w] <= 1:
                        removable = False
                        break
                if removable:
                    codewords.pop(i)
                    improved = True
                    break
                continue
            
            # Try to find a replacement codeword that covers all critical words
            # of this codeword and doesn't make things worse
            # For efficiency, pick a random critical word and find codewords covering it
            target_w = rng.choice(cw_critical)
            # Find all codewords that cover target_w
            candidates = [c for c in range(total_space) if c != cw and target_w in neighbor_map[c]]
            
            if not candidates:
                continue
            
            # Score each candidate: how many of cw's critical words it covers,
            # minus how many new critical words it would create
            best_candidate = None
            best_score = 0
            for cand in candidates[:50]:  # Limit to 50 candidates for speed
                # Count how many of cw's critical words cand covers
                covers_critical = sum(1 for w in cw_critical if w in neighbor_map[cand])
                if covers_critical < len(cw_critical):
                    continue
                # Check if removing cw and adding cand keeps coverage
                # A word w is still covered if counts[w] >= 2 or cand covers w
                # Words that become uncovered: those with counts[w]==1 and only covered by cw
                # and not covered by cand
                newly_uncovered = 0
                for w in neighbor_map[cw]:
                    if counts[w] == 1 and w not in neighbor_map[cand]:
                        newly_uncovered += 1
                if newly_uncovered == 0:
                    # Also check if cand introduces any issues
                    # Words that cand covers but were uncovered before? No, all are covered.
                    # The gain is: how many words go from count 1 to count 2 (good)
                    # vs words that go from count 2 to count 1 (bad)
                    # Net effect on critical count:
                    # Remove cw: words in neighbor_map[cw] lose 1 count
                    # Add cand: words in neighbor_map[cand] gain 1 count
                    # Words in both: no net change
                    # Words only in cw: count decreases by 1
                    # Words only in cand: count increases by 1
                    # New critical words: words only in cw with count 2 -> become 1
                    # Removed critical words: words only in cand with count 1 -> become 2
                    # But we already ensured no newly uncovered.
                    score = covers_critical  # Higher is better
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
            
            if best_candidate is not None:
                codewords[i] = best_candidate
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    total_space = q ** n
    
    # Build neighbor map
    neighbor_map = _build_incidence(q, n, radius)
    
    # Phase 1: Multiple greedy covers with different seeds
    best_codewords = None
    best_len = float('inf')
    
    for seed in [42, 1, 2, 3, 7, 13, 100, 999, 123, 456]:
        if time.time() >= deadline - 15.0:
            break
        cw = _greedy_cover(neighbor_map, total_space, deadline, seed=seed)
        cw = _local_search_reduce(cw, neighbor_map, total_space, deadline)
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
    
    # Phase 2: Local search swap on the best solution
    if time.time() < deadline - 10.0:
        best_codewords = _local_search_swap(best_codewords, neighbor_map, total_space, deadline, max_iterations=100)
    
    # Phase 3: Final reduction pass
    best_codewords = _local_search_reduce(best_codewords, neighbor_map, total_space, deadline)
    
    # Phase 4: Another round of swaps if time permits
    if time.time() < deadline - 5.0:
        best_codewords = _local_search_swap(best_codewords, neighbor_map, total_space, deadline, max_iterations=50)
        best_codewords = _local_search_reduce(best_codewords, neighbor_map, total_space, deadline)
    
    # Convert back to lists
    result = [_to_word(cw, q, n) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
