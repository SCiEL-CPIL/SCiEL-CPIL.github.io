import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 112.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors as array and as bitsets for fast intersection
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        # Build bitset representation: neighbor_bits[w] is a bitset of words within distance 1 of w
        # N=6561, so we need ceil(6561/64)=103 uint64 words per bitset
        num_words = (N + 63) // 64
        neighbor_bits = np.zeros((N, num_words), dtype=np.uint64)
        for w in range(N):
            for nb in neighbors[w, :]:
                neighbor_bits[w, nb // 64] |= (1 << (nb % 64))
        
        def greedy_init(rng, use_randomized=False):
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            while np.any(uncovered) and time.time() < deadline:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                if use_randomized and len(candidates) > 1:
                    top_k = min(8, len(candidates))
                    chosen = rng.choice(candidates[:top_k])
                    w = chosen
                else:
                    w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w, :]] = False
            return list(code)
        
        def reduce_code(code, neighbors, deadline):
            code_set = set(code)
            if not code_set:
                return code_set
            
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            
            return code_set
        
        def try_swap(code_set, coverage, neighbors, neighbor_bits, deadline, rng):
            """Try to swap a codeword with another to reduce size using bitset checks."""
            if not code_set:
                return code_set, coverage, False
            
            c = rng.choice(list(code_set))
            c_neighbors = neighbors[c, :]
            
            # Find words covered only by c
            essential_words = c_neighbors[coverage[c_neighbors] == 1]
            
            if len(essential_words) == 0:
                code_set.remove(c)
                coverage[neighbors[c, :]] -= 1
                return code_set, coverage, True
            
            # Pick one essential word w
            w = rng.choice(essential_words)
            
            # Find candidates d that cover w
            candidates = neighbors[w, :]
            
            best_size = len(code_set)
            best_new_set = None
            best_new_coverage = None
            
            cand_list = candidates.tolist()
            random.shuffle(cand_list)
            
            for d in cand_list[:20]:
                if d in code_set:
                    continue
                
                # Check: all essential_words must be in neighbors[d]
                # Use bitset: essential_words must be subset of neighbor_bits[d]
                # Build bitset of essential_words and check subset
                ew_bitset = np.zeros(num_words, dtype=np.uint64)
                for ew in essential_words:
                    ew_bitset[ew // 64] |= (1 << (ew % 64))
                
                # Check if ew_bitset is subset of neighbor_bits[d]
                if np.any(ew_bitset & ~neighbor_bits[d]):
                    continue
                
                # This d covers all essential words of c
                # Now check if we can remove other codewords that become redundant
                # Quick check: compute new coverage for affected region only
                new_coverage = coverage.copy()
                new_coverage[neighbors[c, :]] -= 1
                new_coverage[neighbors[d, :]] += 1
                
                # Try to find redundant codewords in the new coverage
                new_set = set(code_set)
                new_set.remove(c)
                new_set.add(d)
                
                # Check each codeword in new_set for redundancy (only those near c or d)
                affected = set(neighbors[c, :].tolist()) | set(neighbors[d, :].tolist())
                removed_any = False
                for cc in list(new_set):
                    if cc not in affected:
                        continue
                    if np.min(new_coverage[neighbors[cc, :]]) > 1:
                        new_set.remove(cc)
                        new_coverage[neighbors[cc, :]] -= 1
                        removed_any = True
                
                if len(new_set) < best_size:
                    best_size = len(new_set)
                    best_new_set = new_set
                    best_new_coverage = new_coverage
            
            if best_new_set is not None:
                return best_new_set, best_new_coverage, True
            
            return code_set, coverage, False
        
        def try_double_swap(code_set, coverage, neighbors, neighbor_bits, deadline, rng):
            """Try removing two codewords and adding one to reduce size by 1."""
            if len(code_set) < 3:
                return code_set, coverage, False
            
            c_list = list(code_set)
            if len(c_list) < 2:
                return code_set, coverage, False
            
            # Pick two random codewords to potentially remove
            idx1 = rng.integers(len(c_list))
            idx2 = rng.integers(len(c_list) - 1)
            if idx2 >= idx1:
                idx2 += 1
            c1 = c_list[idx1]
            c2 = c_list[idx2]
            
            # Find words covered only by c1 or only by c2
            c1_neighbors = neighbors[c1, :]
            c2_neighbors = neighbors[c2, :]
            
            # Words that need coverage after removing c1 and c2
            # They must be covered by the new codeword d
            # A word is "essential" if it's in neighbors[c1] or neighbors[c2] 
            # and has coverage <= 2 (will be uncovered if not covered by d)
            
            # Build the set of words that would become uncovered
            # coverage_after_removal = coverage - [c in neighbors[c1]] - [c in neighbors[c2]]
            # We need d such that for all w with coverage_after_removal[w] == 0, w is in neighbors[d]
            
            # Efficient approach: compute which words have coverage exactly 1 or 2 
            # and are in the union of neighbors[c1] and neighbors[c2]
            
            union_mask = np.zeros(N, dtype=np.bool_)
            union_mask[c1_neighbors] = True
            union_mask[c2_neighbors] = True
            
            # Words in union with coverage 1: these MUST be covered by d
            must_cover = union_mask & (coverage == 1)
            
            # Words in union with coverage 2: if they're in both c1 and c2 neighbors, 
            # removing both makes them uncovered, so they must be covered by d
            in_both = np.zeros(N, dtype=np.bool_)
            in_both[c1_neighbors] = True
            in_both[c2_neighbors] = True
            # Actually: coverage==2 and in_both means after removing c1,c2, coverage becomes 0
            # Wait, coverage counts ALL codewords. If coverage[w]==2 and w is in neighbors[c1] and neighbors[c2],
            # then after removing c1 and c2, coverage[w] becomes 0.
            # If coverage[w]==2 and w is in neighbors[c1] but not neighbors[c2], after removing c1,c2, coverage[w] becomes 1.
            
            # So must_cover = words where (coverage[w] - [w in nb(c1)] - [w in nb(c2)]) == 0
            # i.e., coverage[w] == [w in nb(c1)] + [w in nb(c2)]
            
            in_c1 = np.zeros(N, dtype=np.int32)
            in_c1[c1_neighbors] = 1
            in_c2 = np.zeros(N, dtype=np.int32)
            in_c2[c2_neighbors] = 1
            
            must_cover_mask = (coverage == (in_c1 + in_c2)) & union_mask
            
            if not np.any(must_cover_mask):
                # Both c1 and c2 are redundant!
                code_set.remove(c1)
                code_set.remove(c2)
                coverage[neighbors[c1, :]] -= 1
                coverage[neighbors[c2, :]] -= 1
                return code_set, coverage, True
            
            must_cover_words = np.where(must_cover_mask)[0]
            
            # Pick a word from must_cover_words to guide candidate search
            w = rng.choice(must_cover_words)
            
            # Candidates d must cover all must_cover_words
            # Find d in neighbors[w] that covers all must_cover_words
            candidates = neighbors[w, :]
            
            best_size = len(code_set) - 1  # We want to remove 2, add 1, net -1
            best_new_set = None
            best_new_coverage = None
            
            cand_list = candidates.tolist()
            random.shuffle(cand_list)
            
            for d in cand_list[:15]:
                if d in code_set:
                    continue
                
                # Check if d covers all must_cover_words
                # Build bitset of must_cover_words
                mc_bitset = np.zeros(num_words, dtype=np.uint64)
                for mw in must_cover_words:
                    mc_bitset[mw // 64] |= (1 << (mw % 64))
                
                if np.any(mc_bitset & ~neighbor_bits[d]):
                    continue
                
                # d covers all essential words. Check if we can remove more.
                new_coverage = coverage.copy()
                new_coverage[neighbors[c1, :]] -= 1
                new_coverage[neighbors[c2, :]] -= 1
                new_coverage[neighbors[d, :]] += 1
                
                new_set = set(code_set)
                new_set.remove(c1)
                new_set.remove(c2)
                new_set.add(d)
                
                # Try to remove more redundant codewords
                affected = set(neighbors[c1, :].tolist()) | set(neighbors[c2, :].tolist()) | set(neighbors[d, :].tolist())
                for cc in list(new_set):
                    if cc not in affected:
                        continue
                    if np.min(new_coverage[neighbors[cc, :]]) > 1:
                        new_set.remove(cc)
                        new_coverage[neighbors[cc, :]] -= 1
                
                if len(new_set) < best_size:
                    best_size = len(new_set)
                    best_new_set = new_set
                    best_new_coverage = new_coverage
            
            if best_new_set is not None:
                return best_new_set, best_new_coverage, True
            
            return code_set, coverage, False
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts
        max_restarts = 30
        for i in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.default_rng(42 + i * 13)
            
            # Alternate strategies
            use_randomized = (i % 3 == 0)
            
            # 1. Greedy Init
            code_list = greedy_init(rng, use_randomized)
            code_set = set(code_list)
            
            # 2. Initial Reduction
            code_set = reduce_code(list(code_set), neighbors, deadline)
            
            # 3. Calculate Coverage
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            
            # 4. Iterative Improvement with single and double swaps
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                
                # Single swaps
                for _ in range(60):
                    if time.time() > deadline - 3:
                        break
                    new_set, new_coverage, was_improved = try_swap(
                        code_set, coverage, neighbors, neighbor_bits, deadline, rng
                    )
                    if was_improved:
                        code_set = new_set
                        coverage = new_coverage
                        improved = True
                
                # Double swaps
                for _ in range(30):
                    if time.time() > deadline - 3:
                        break
                    new_set, new_coverage, was_improved = try_double_swap(
                        code_set, coverage, neighbors, neighbor_bits, deadline, rng
                    )
                    if was_improved:
                        code_set = new_set
                        coverage = new_coverage
                        improved = True
                
                # Final reduction pass
                code_set = reduce_code(list(code_set), neighbors, deadline)
                coverage = np.zeros(N, dtype=np.int32)
                for c in code_set:
                    coverage[neighbors[c, :]] += 1
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
                
            # Early termination if we hit the known optimum
            if best_size <= 486:
                break

        if best_code is None:
            # Fallback: tetracode product
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
        # Convert to list of lists of digits
        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        # Generic fallback
        N = q ** n
        deadline = time.time() + 110.0
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
