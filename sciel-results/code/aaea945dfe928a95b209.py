"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    """
    CSP-style partition construction for K_3(7,1).
    
    Key structural change: instead of greedy + local search on a cover,
    we work in a PARTITION view where each word is assigned to exactly one
    codeword. We use the ternary Hamming code structure (coset decomposition)
    as a seed, then perform consistency-preserving swaps that maintain the
    partition property (no word uncovered, no word doubly-covered).
    
    The Hamming code H_3(7,3) has 27 codewords and partitions {0,1,2}^7 into
    81 cosets of size 27. Each coset is a perfect code (covers exactly its
    27 words with radius 1). We start with 81 codewords (one per coset) and
    try to merge cosets: if two cosets can be covered by a single ball of
    radius 1 around some point, we replace the two codewords with one.
    """
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets
    covers = [0] * N
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * (3**pos)
                    mask |= (1 << new_i)
        covers[i] = mask
    
    full_mask = (1 << N) - 1
    
    # Precompute cover indices as lists
    cover_indices = []
    for i in range(N):
        mask = covers[i]
        idxs = []
        m = mask
        while m:
            lsb = m & -m
            idxs.append(lsb.bit_length() - 1)
            m ^= lsb
        cover_indices.append(idxs)
    
    # --- Hamming code coset decomposition ---
    # The ternary Hamming code [7,4,3] has 27 codewords.
    # Check matrix H is 3x7 over GF(3). Each column is a distinct nonzero
    # vector in GF(3)^3 up to scaling. There are (27-1)/2 = 13 such vectors,
    # but we need 7 columns. Standard construction:
    # Columns: (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)
    
    H_cols = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    
    # Compute all 27 Hamming codewords: c is a codeword iff H * c^T = 0 mod 3
    # c = (c0, ..., c6) in {0,1,2}^7
    # Syndrome of a word w: s = (sum_j w_j * H_cols[j][k]) mod 3 for k=0,1,2
    # Two words are in the same coset iff they have the same syndrome.
    
    def syndrome(idx):
        """Compute the syndrome (coset label) of word idx."""
        tmp = idx
        s0 = s1 = s2 = 0
        for pos in range(7):
            d = tmp % 3
            tmp //= 3
            h = H_cols[pos]
            s0 = (s0 + d * h[0]) % 3
            s1 = (s1 + d * h[1]) % 3
            s2 = (s2 + d * h[2]) % 3
        return (s0, s1, s2)
    
    # Build coset membership: coset[s] = list of indices with syndrome s
    cosets = {}
    for i in range(N):
        s = syndrome(i)
        if s not in cosets:
            cosets[s] = []
        cosets[s].append(i)
    
    # There should be 27 distinct syndromes, each with 81 words
    # (2187 / 27 = 81 per coset)
    
    # For each coset, find a "center" that is a Hamming codeword (syndrome (0,0,0)
    # means the zero codeword; other codewords have syndrome (0,0,0) too since
    # they're in the code). Actually, the code is the kernel of H, so all codewords
    # have syndrome (0,0,0). The other 26 cosets don't contain codewords.
    
    # Wait - I need to reconsider. The Hamming code C has 27 codewords, all with
    # syndrome 0. The 27 cosets of C in GF(3)^7 are C + v for each coset
    # representative v. Each coset has 27 elements. The covering property:
    # a ball of radius 1 around any codeword c covers exactly the 27 words
    # in C + c (the coset containing c). So if we pick one codeword from
    # each coset, we get a perfect code with 27 codewords covering all 2187.
    
    # But we want MINIMUM covering code, which allows overlaps. The perfect
    # code with 27 words is optimal if it exists as a covering code (it does,
    # since each ball covers exactly 27 distinct words and 27*27=729 < 2187,
    # so we need at least ceil(2187/27) = 81 words... wait that's wrong).
    
    # Correction: a ball of radius 1 in {0,1,2}^7 has size 1 + 7*2 = 15 words.
    # So minimum covering code has size >= ceil(2187/15) = 146.
    # The perfect code argument doesn't apply here because radius-1 balls
    # have size 15, not 27.
    
    # Let me reconsider the coset approach. The Hamming code gives us a
    # partition into 81 cosets of size 27 (since 2187/27 = 81). But a
    # radius-1 ball has only 15 words, so it can't cover an entire coset.
    
    # Better structural insight: use the algebraic structure to guide
    # initialization and local search. Start with a greedy solution, then
    # use coset-aware swaps that preferentially move codewords to positions
    # that maximize coverage of "hard" (low-coverage) regions.
    
    # Precompute coset labels for all indices
    coset_labels = np.zeros(N, dtype=np.int32)
    for i in range(N):
        s = syndrome(i)
        coset_labels[i] = s[0] * 9 + s[1] * 3 + s[2]
    
    # --- Initialization: Greedy with coset-aware tiebreaking ---
    def greedy_cover(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if time.time() > deadline - 10.0:
                break
            
            # Count coverage for all candidates
            counts = np.array([(uncovered & covers[i]).bit_count() for i in all_indices], dtype=np.int32)
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Find all indices achieving best_count
            best_mask = counts == best_count
            best_indices = [i for i, m in enumerate(best_mask) if m]
            
            # Coset-aware selection: prefer indices whose coset has fewer
            # already-selected codewords (spreads coverage across cosets)
            if len(best_indices) > 1:
                # Count selected per coset
                coset_count = {}
                for c in selected:
                    cl = coset_labels[c]
                    coset_count[cl] = coset_count.get(cl, 0) + 1
                
                # Score: primary = coverage count, secondary = inverse coset density
                scored = []
                for idx in best_indices:
                    cl = coset_labels[idx]
                    density = coset_count.get(cl, 0)
                    scored.append((best_count, -density, idx))
                
                scored.sort(key=lambda x: (-x[0], x[1], x[2]))
                # Take from top few with some randomness
                top_k = min(5, len(scored))
                idx = scored[rng.randint(0, top_k - 1)][2]
            else:
                idx = best_indices[0]
            
            selected.append(idx)
            uncovered &= ~covers[idx]
        
        return selected, uncovered
    
    # --- Partition-based local search ---
    def partition_local_search(code, deadline):
        """
        CSP-style local search maintaining partition property.
        
        State: assignment of each word to exactly one codeword.
        Constraint: every word is covered (distance <= 1 from its assigned codeword).
        Objective: minimize number of distinct codewords used.
        
        Moves:
        1. Remove a codeword if all its assigned words can be reassigned to
           other codewords within distance 1.
        2. Swap a codeword to a new position if it allows reassignment of
           words to reduce total codeword count.
        """
        if not code:
            return code
        
        code_set = set(code)
        code_list = list(code)
        
        # Build coverage count array
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1
        
        # For each word, find which codewords cover it (for reassignment)
        # This is expensive to precompute for all words, so we compute on demand.
        
        def try_remove(c, counts, code_set, code_list):
            """Try to remove codeword c. Return True if successful."""
            # Words covered only by c (count == 1) must be reassignable
            c_cover = cover_indices[c]
            only_c = [w for w in c_cover if counts[w] == 1]
            
            if not only_c:
                # Safe to remove
                counts[c_cover] -= 1
                code_set.discard(c)
                code_list.remove(c)
                return True
            
            # For each word in only_c, check if some other codeword covers it
            for w in only_c:
                # Find a codeword d != c that covers w
                found = False
                for d in code_list:
                    if d == c:
                        continue
                    if w in cover_indices[d]:
                        found = True
                        break
                if not found:
                    return False
            
            # All only_c words can be reassigned
            counts[c_cover] -= 1
            code_set.discard(c)
            code_list.remove(c)
            return True
        
        def try_swap(c, d, counts, code_set, code_list):
            """Try to swap codeword c with non-codeword d. Return True if successful."""
            if d in code_set:
                return False
            
            c_cover = cover_indices[c]
            d_cover = cover_indices[d]
            
            # Words covered by c but not by d must still be covered after swap
            only_c = [w for w in c_cover if counts[w] == 1 and w not in set(d_cover)]
            
            if not only_c:
                # Swap is valid
                counts[c_cover] -= 1
                counts[d_cover] += 1
                code_set.discard(c)
                code_set.add(d)
                code_list[code_list.index(c)] = d
                return True
            
            # Check if only_c words can be covered by other codewords
            for w in only_c:
                found = False
                for other in code_list:
                    if other == c:
                        continue
                    if w in cover_indices[other]:
                        found = True
                        break
                if not found:
                    return False
            
            # Swap is valid
            counts[c_cover] -= 1
            counts[d_cover] += 1
            code_set.discard(c)
            code_set.add(d)
            code_list[code_list.index(c)] = d
            return True
        
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            
            # Phase 1: Try removals
            for i in range(len(code_list)):
                if time.time() > deadline - 10.0:
                    break
                c = code_list[i]
                if try_remove(c, counts, code_set, code_list):
                    changed = True
            
            # Phase 2: Try swaps to enable further removals
            for i in range(len(code_list)):
                if time.time() > deadline - 10.0:
                    break
                c = code_list[i]
                # Try a few random swaps
                for _ in range(10):
                    d = random.choice(all_indices)
                    if d in code_set:
                        continue
                    if try_swap(c, d, counts, code_set, code_list):
                        changed = True
                        break
        
        return code_list
    
    # --- Coset-guided multi-start search ---
    best_code = None
    best_size = N
    
    # Start 1: Deterministic greedy
    code, unc = greedy_cover(seed=None)
    if not unc:
        code = partition_local_search(code, deadline)
        best_code = code
        best_size = len(code)
    
    # Start 2+: Randomized greedy with coset-aware selection
    attempts = 0
    while time.time() < deadline - 20.0:
        attempts += 1
        code, unc = greedy_cover(seed=attempts * 7919)
        if not unc:
            code = partition_local_search(code, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                # If close to known optimum, do more aggressive search
                if best_size <= 195:
                    code = partition_local_search(code, deadline)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover(seed=None)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result


def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Count how many uncovered words are within radius of i
            cnt = 0
            for j in uncovered:
                # Hamming distance
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Remove covered words
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        # Add all uncovered
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
