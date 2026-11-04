"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return list(itertools.product(range(3), repeat=n))
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Encode words as integers 0..728
    # Word w = [w0,w1,w2,w3,w4,w5] -> int = w0 + 3*w1 + 9*w2 + 27*w3 + 81*w4 + 243*w5
    def word_to_int(w):
        return w[0] + 3*w[1] + 9*w[2] + 27*w[3] + 81*w[4] + 243*w[5]
    
    def int_to_word(x):
        return [(x // 3**i) % 3 for i in range(6)]
    
    # Precompute neighbors: for each word, the set of words within Hamming distance 1
    # A word w covers itself and all words differing in exactly one position
    # Total neighbors per word: 1 + 6*2 = 13
    
    # For symmetry breaking, we use the automorphism group of H(6,3):
    # - Permute coordinates (S_6)
    # - Permute alphabet symbols in each coordinate (S_3^6)
    # - Translate coordinates (Z_3^6) - actually for q=3, adding a constant mod 3
    
    # Strategy: Use branch-and-bound with symmetry breaking
    # Key symmetry-breaking constraint: enforce that the first codeword is in a
    # canonical form. By the automorphism group, we can assume:
    # 1. First codeword has first coordinate = 0 (translate)
    # 2. First codeword has second coordinate = 0 (translate)
    # 3. More constraints via coordinate permutations and alphabet permutations
    
    # Actually, let's use a greedy + local search approach with symmetry-aware initialization
    # and a sophisticated local search that exploits structure.
    
    # Generate all 729 words
    all_words = list(itertools.product(range(3), repeat=6))
    all_ints = [word_to_int(w) for w in all_words]
    int_to_idx = {x: i for i, x in enumerate(all_ints)}
    
    # Precompute coverage: for each codeword, which words it covers
    # Use numpy for speed
    all_words_arr = np.array(all_words)  # shape (729, 6)
    
    # For each position i and symbol s, find words that differ only at position i with value s
    # A word w is covered by codeword c if they differ in at most 1 position
    
    # Build coverage matrix efficiently
    # cover[c_idx, w_idx] = 1 if w is within distance 1 of c
    # This is a 729x729 boolean matrix
    
    # For each word w, it's covered by c if:
    # - c == w, or
    # - c differs from w in exactly one position
    
    # Let's precompute for each word, the list of codewords that cover it
    # Actually, for the search, we need: given a set of selected codewords, which words are covered
    
    # Use a different approach: represent coverage as bitsets for efficiency
    # Each codeword covers 13 words. Represent as a set of integers.
    
    # Precompute: for each codeword index, the set of word indices it covers
    covers = []
    for ci, c in enumerate(all_words):
        covered = {ci}  # covers itself
        for pos in range(6):
            for sym in range(3):
                if sym == c[pos]:
                    continue
                w = list(c)
                w[pos] = sym
                wi = word_to_int(w)
                covered.add(wi)
        covers.append(frozenset(covered))
    
    # Total words to cover: 729
    # We need minimum set of codewords such that union of their covers = all 729 words
    
    # Use a greedy algorithm with multiple restarts and local search
    
    def greedy_set_cover(coverage_sets, num_elements, time_limit=10):
        """Greedy set cover with randomization."""
        t0 = time.time()
        remaining = set(range(num_elements))
        selected = []
        # coverage_sets[i] is the set of elements covered by set i
        # We need to track which elements are still uncovered
        
        # For efficiency, track for each element, which sets cover it
        # But that's expensive. Instead, use the standard greedy: at each step,
        # pick the set that covers the most uncovered elements.
        
        # Precompute: for each element, which sets cover it
        elem_to_sets = [set() for _ in range(num_elements)]
        for si, s in enumerate(coverage_sets):
            for e in s:
                elem_to_sets[e].add(si)
        
        # Greedy with random tie-breaking
        while remaining:
            if time.time() - t0 > time_limit:
                break
            best_set = -1
            best_count = -1
            best_sets = []
            for si in range(len(coverage_sets)):
                if si in selected:
                    continue
                count = len(coverage_sets[si] & remaining)
                if count > best_count:
                    best_count = count
                    best_sets = [si]
                elif count == best_count and count > 0:
                    best_sets.append(si)
            if not best_sets:
                break
            # Random selection among best
            chosen = random.choice(best_sets)
            selected.append(chosen)
            remaining -= coverage_sets[chosen]
        
        return selected
    
    # Local search: try to remove codewords, then swap
    def try_remove(selected, covers, num_elements):
        """Try to remove codewords from selected while maintaining coverage."""
        selected = list(selected)
        # Compute coverage count for each element
        cov_count = np.zeros(num_elements, dtype=np.int32)
        for si in selected:
            for e in covers[si]:
                cov_count[e] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                si = selected[i]
                # Check if removing si still covers all elements
                can_remove = True
                for e in covers[si]:
                    if cov_count[e] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for e in covers[si]:
                        cov_count[e] -= 1
                    selected.pop(i)
                    improved = True
                    break
        return selected
    
    def local_search(selected, covers, num_elements, time_limit=20):
        """Local search with add-then-remove heuristic."""
        t0 = time.time()
        best = list(selected)
        best_len = len(best)
        
        # Compute coverage for best
        def compute_cov_count(sel):
            cc = np.zeros(num_elements, dtype=np.int32)
            for si in sel:
                for e in covers[si]:
                    cc[e] += 1
            return cc
        
        while time.time() - t0 < time_limit:
            # Try adding a random codeword, then removing
            if len(best) >= num_elements:
                break
            cc = compute_cov_count(best)
            
            # Find uncovered elements
            uncovered = np.where(cc == 0)[0]
            if len(uncovered) > 0:
                # Need to add codewords to cover uncovered
                # Pick a codeword that covers many uncovered
                best_add = -1
                best_add_count = -1
                for si in range(num_elements):
                    if si in best:
                        continue
                    count = sum(1 for e in covers[si] if cc[e] == 0)
                    if count > best_add_count:
                        best_add_count = count
                        best_add = si
                if best_add >= 0 and best_add_count > 0:
                    candidate = best + [best_add]
                    candidate = try_remove(candidate, covers, num_elements)
                    if len(candidate) < best_len:
                        best = candidate
                        best_len = len(candidate)
                        continue
            
            # Try swap: replace one codeword with another
            if len(best) > 0:
                # Try removing one and see if we can cover with a different one
                for i in range(len(best)):
                    if time.time() - t0 > time_limit:
                        break
                    si = best[i]
                    temp = best[:i] + best[i+1:]
                    cc = compute_cov_count(temp)
                    uncovered = np.where(cc == 0)[0]
                    if len(uncovered) == 0:
                        # Can remove
                        best = temp
                        best_len = len(best)
                    elif len(uncovered) <= 3:
                        # Try to find a single codeword that covers all uncovered
                        for sj in range(num_elements):
                            if sj in best:
                                continue
                            if all(cc[e] == 0 and e in covers[sj] for e in uncovered):
                                candidate = temp + [sj]
                                candidate = try_remove(candidate, covers, num_elements)
                                if len(candidate) < best_len:
                                    best = candidate
                                    best_len = len(candidate)
                                break
        
        return best
    
    # Multiple restarts with different random seeds
    best_solution = None
    best_len = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization: pick random codewords
        init_size = random.randint(60, 90)
        init = random.sample(range(729), init_size)
        
        # Greedy completion
        remaining = set(range(729))
        selected = list(init)
        for si in selected:
            remaining -= covers[si]
        
        # Greedy to complete
        t0 = time.time()
        while remaining and time.time() - t0 < 5:
            best_set = -1
            best_count = -1
            best_sets = []
            for si in range(729):
                if si in selected:
                    continue
                count = len(covers[si] & remaining)
                if count > best_count:
                    best_count = count
                    best_sets = [si]
                elif count == best_count and count > 0:
                    best_sets.append(si)
            if not best_sets:
                break
            chosen = random.choice(best_sets)
            selected.append(chosen)
            remaining -= covers[chosen]
        
        if remaining:
            # Fill in remaining with brute force
            for si in range(729):
                if not remaining:
                    break
                if si not in selected:
                    selected.append(si)
                    remaining -= covers[si]
        
        # Local search
        selected = local_search(selected, covers, 729, time_limit=15)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_solution = list(selected)
            print(f"Restart {restart}: found solution of size {best_len}", flush=True)
        
        if best_len <= 73:
            break
    
    if best_solution is None:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert back to words
    return [int_to_word(x) for x in best_solution]
# EVOLVE-BLOCK-END
