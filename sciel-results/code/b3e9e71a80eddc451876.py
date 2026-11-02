"""Improved ternary covering code K_3(7,1) via randomized greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(q, n, radius):
    # Only the ternary length-7 radius-1 case is the hard one; handle others simply.
    if not (q == 3 and n == 7 and radius == 1):
        # Fallback: direct-sum style construction for small cases
        # For general q,n,radius, use a simple greedy
        return _general_greedy(q, n, radius)
    
    # Deadline guard: evaluator kills at 150s; keep safety margin
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin
    
    # Precompute all 2187 words as tuples for hashing
    all_words = list(itertools.product(range(3), repeat=7))
    all_words_set = set(all_words)
    
    # Precompute neighbors: for each word, the set of words within Hamming distance 1
    # A word w has 1 + 7*2 = 15 neighbors (itself + 14 single-position changes)
    # We'll represent coverage using a set of covered words
    
    def neighbors_of(word):
        """Yield all words within Hamming distance 1 of word."""
        yield word
        for i in range(7):
            for d in range(3):
                if d != word[i]:
                    new_word = list(word)
                    new_word[i] = d
                    yield tuple(new_word)
    
    # Precompute neighbor lists for all words (memory: 2187 * 15 = ~32805 entries)
    neighbor_cache = {}
    for w in all_words:
        neighbor_cache[w] = list(neighbors_of(w))
    
    # Greedy construction with randomization
    def greedy_randomized(seed):
        rng = random.Random(seed)
        covered = set()
        codewords = []
        remaining = set(all_words)
        
        while remaining:
            # Score each candidate: number of newly covered words
            best_candidate = None
            best_count = -1
            best_tie = 0
            
            # For speed, only consider a random subset if remaining is large
            if len(remaining) > 500:
                # Sample candidates from remaining
                candidates = random.sample(list(remaining), min(200, len(remaining)))
            else:
                candidates = list(remaining)
            
            for cand in candidates:
                new_cover = 0
                for nb in neighbor_cache[cand]:
                    if nb in remaining:
                        new_cover += 1
                # Add small random perturbation to break ties
                score = new_cover + rng.random() * 0.01
                if score > best_count + 1e-9:
                    best_count = score
                    best_candidate = cand
                    best_tie = new_cover
            
            if best_candidate is None:
                # Fallback: pick random from remaining
                best_candidate = rng.choice(list(remaining))
            
            codewords.append(list(best_candidate))
            # Remove all neighbors from remaining
            for nb in neighbor_cache[best_candidate]:
                remaining.discard(nb)
            covered.add(best_candidate)
        
        return codewords
    
    # Local search: try to remove redundant codewords
    def try_remove_redundant(codewords):
        """Try to remove codewords that are redundant (all their unique coverage is covered by others)."""
        covered_by = {}  # word -> set of indices in codewords that cover it
        for i, cw in enumerate(codewords):
            for nb in neighbor_cache[tuple(cw)]:
                if nb not in covered_by:
                    covered_by[nb] = set()
                covered_by[nb].add(i)
        
        # A codeword i is redundant if for every word w covered by i, there exists j != i with w covered by j
        removed = set()
        for i in range(len(codewords)):
            if i in removed:
                continue
            redundant = True
            for nb in neighbor_cache[tuple(codewords[i])]:
                others = covered_by[nb] - {i}
                if not others:
                    redundant = False
                    break
            if redundant:
                removed.add(i)
                # Update covered_by: remove i from all words it covers
                for nb in neighbor_cache[tuple(codewords[i])]:
                    covered_by[nb].discard(i)
        
        if removed:
            return [cw for i, cw in enumerate(codewords) if i not in removed]
        return codewords
    
    # Local search: try to swap a codeword with a better one
    def local_search_swap(codewords, max_iters=50):
        """Try to replace codewords to reduce total count or improve coverage."""
        codewords = list(codewords)
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Try removing each codeword and see if coverage is maintained
            for i in range(len(codewords)):
                if time.time() - start_time > DEADLINE:
                    break
                # Check if codeword i is redundant
                test_cw = tuple(codewords[i])
                # Find words only covered by this codeword
                only_covered = []
                for nb in neighbor_cache[test_cw]:
                    count = 0
                    for j, cw in enumerate(codewords):
                        if j == i:
                            continue
                        # Check if nb is within distance 1 of cw
                        if _hamming_dist_leq(tuple(cw), nb, 1):
                            count += 1
                            if count >= 1:
                                break
                    if count == 0:
                        only_covered.append(nb)
                if not only_covered:
                    codewords.pop(i)
                    improved = True
                    break
            if not improved:
                break
        return codewords
    
    def _hamming_dist_leq(a, b, max_dist):
        """Check if Hamming distance between a and b is <= max_dist."""
        dist = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                dist += 1
                if dist > max_dist:
                    return False
        return True
    
    # Run multiple randomized greedy + local search iterations
    best_code = None
    best_len = float('inf')
    
    # First, run a deterministic greedy to get a baseline
    baseline = greedy_randomized(0)
    baseline = try_remove_redundant(baseline)
    baseline = local_search_swap(baseline, max_iters=30)
    best_code = baseline
    best_len = len(baseline)
    
    # Try multiple random seeds
    for seed in range(1, 1000):
        if time.time() - start_time > DEADLINE:
            break
        
        code = greedy_randomized(seed)
        code = try_remove_redundant(code)
        code = local_search_swap(code, max_iters=20)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
            # Try further refinement on the best
            if time.time() - start_time < DEADLINE - 5:
                refined = local_search_swap(best_code, max_iters=50)
                if len(refined) < best_len:
                    best_code = refined
                    best_len = best_len  # already updated
    
    # Final cleanup pass
    if best_code is not None and time.time() - start_time < DEADLINE - 2:
        best_code = try_remove_redundant(best_code)
        best_code = local_search_swap(best_code, max_iters=10)
    
    # Verify coverage (safety check)
    if best_code is None:
        # Should not happen, but fallback
        best_code = baseline
    
    return best_code


def _general_greedy(q, n, radius):
    """General greedy for non-special cases."""
    all_words = list(itertools.product(range(q), repeat=n))
    
    def hamming_dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    def neighbors_within_radius(word, r):
        """Generate all words within Hamming distance r of word."""
        if r == 0:
            yield word
            return
        if r == 1:
            yield word
            for i in range(n):
                for d in range(q):
                    if d != word[i]:
                        new_word = list(word)
                        new_word[i] = d
                        yield tuple(new_word)
            return
        # General case: recursive
        def _gen(pos, remaining_dist, current):
            if remaining_dist == 0:
                yield tuple(current)
                return
            if pos == n:
                return
            # Don't change position pos
            for val in range(q):
                current[pos] = val
                yield from _gen(pos + 1, remaining_dist, current)
                # Change position pos
                if remaining_dist > 0:
                    for val in range(q):
                        if val != word[pos]:
                            current[pos] = val
                            yield from _gen(pos + 1, remaining_dist - 1, current)
        
        yield from _gen(0, radius, [0] * n)
    
    covered = set()
    codewords = []
    remaining = set(all_words)
    
    while remaining:
        best_candidate = None
        best_count = -1
        for cand in list(remaining):
            new_cover = sum(1 for nb in neighbors_within_radius(cand, radius) if nb in remaining)
            if new_cover > best_count:
                best_count = new_cover
                best_candidate = cand
        codewords.append(list(best_candidate))
        for nb in neighbors_within_radius(best_candidate, radius):
            remaining.discard(nb)
    
    return codewords
# EVOLVE-BLOCK-END
