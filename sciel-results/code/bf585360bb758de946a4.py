import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # This problem is specifically for q=3, n=7, radius=1.
    # The best known covering code has size 186.
    # It is constructed as the union of 18 cosets of the ternary [7,3,3] Hamming code.
    # The Hamming code H is a linear code of length 7, dimension 3, minimum distance 3.
    # Size of H is 3^3 = 27.
    # 18 * 27 = 486? No.
    # Wait, the covering code C(7,1) over ternary alphabet has size 186.
    # 186 is not divisible by 27. So it's not a union of cosets of the [7,3,3] code?
    # Let's re-read the literature in my head.
    # K_3(7,1) = 186.
    # The ternary Golay code is [11,6,5].
    # The ternary Hamming code is [7,5,3]? No, [7,5,3] has size 3^5=243.
    # [7,3,3] has size 27.
    # Maybe the construction is different.
    
    # Actually, let's look at the "inspiration 3" which used Lagrangian relaxation.
    # But the directive says: "Use a deterministic algebraic construction based on the ternary Golay code structure: generate the 186 codewords by taking the union of specific cosets of the [7,3,3] ternary Hamming code..."
    
    # If the directive is specific, I should follow it, even if my internal knowledge is fuzzy on the exact cosets.
    # However, if I don't know the exact 186 words, I can't hardcode them.
    # Can I generate them?
    # The ternary [7,3,3] code is the dual of the [7,5,3] code? No.
    # The parity check matrix for the ternary Hamming code [7,5,3] has 2 rows? No, 3 rows.
    # H = [1 0 0 1 1 1 1; 0 1 0 1 1 2 2; 0 0 1 1 2 1 2] ? Something like that.
    # The code is the null space of H.
    # If I can generate the [7,3,3] code, I can pick 186 words? No, 186 is not a multiple of 27.
    
    # Wait, is it possible the "186" code is not a union of cosets?
    # Or maybe I should just use a strong heuristic that is faster?
    # The current score is 1.0848. The seed is 1.0.
    # 1.0848 means value = seed / 1.0848. If seed=200, value=184.3?
    # If best known is 186, and seed is likely around 200 or 204.
    # If I can get 186, score = seed/186.
    # If seed is 204, 204/186 = 1.096.
    # If seed is 200, 200/186 = 1.075.
    # Current score 1.0848 suggests value is slightly worse than optimal or close.
    
    # Let's try to generate the [7,3,3] ternary Hamming code cosets.
    # But since 186 is not divisible by 27, this specific algebraic structure might not be a simple union of cosets.
    # Perhaps the directive implies a *search* over cosets? Or maybe I should just implement a very fast greedy + local search that finds 186?
    
    # Let's stick to the directive's *spirit*: "deterministic algebraic construction".
    # If I can't guarantee 186 via algebra, I should use the fastest possible heuristic.
    # The current program uses a boolean cover matrix which is 2187x2187. That's ~4.7M bools. Fast.
    # The bottleneck in the current program is likely the ILS loop or the initial greedy.
    
    # Let's try a different approach: **Integer Linear Programming (ILP)** is too slow.
    # **Branch and Bound**? Too slow.
    # **Fast Greedy with Bitsets**:
    # Represent the space as a bitset of 2187 bits.
    # Each codeword covers 15 positions.
    # Use numpy arrays of uint64 or similar?
    # 2187 bits = 35 uint64s.
    # This makes the "count uncovered" step extremely fast.
    
    # Let's implement a highly optimized Greedy Set Cover using bitsets, followed by a tight local search.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (though task specifies this one)
        return _generic_construct(q, n, radius)
        
    space_size = 3**7 # 2187
    
    # Precompute neighbors as indices
    neighbors = []
    for idx in range(space_size):
        w = [0]*7
        tmp = idx
        for i in range(6, -1, -1):
            w[i] = tmp % 3
            tmp //= 3
        nbs = [idx]
        for pos in range(7):
            d = w[pos]
            for nd in range(3):
                if nd == d: continue
                diff = (nd - d) * (3 ** (6-pos))
                nbs.append(idx + diff)
        neighbors.append(nbs)
        
    # Bitset representation
    # num_words = 2187
    # bits_per_word = 64
    # num_blocks = ceil(2187/64) = 35
    num_blocks = (space_size + 63) // 64
    
    # cover_masks: list of lists of uint64?
    # cover_masks[i] is a list of num_blocks uint64s, where bit j in block b is set if (b*64+j) is in neighbors[i]
    
    cover_masks = []
    for i in range(space_size):
        blocks = [0] * num_blocks
        for nb in neighbors[i]:
            blocks[nb // 64] |= (1 << (nb % 64))
        cover_masks.append(blocks)
        
    # Uncovered mask: list of uint64s
    uncovered = [(1 << min(64, space_size - b*64)) - 1 for b in range(num_blocks)]
    # Mask out bits beyond space_size in the last block
    last_block_bits = space_size % 64
    if last_block_bits:
        uncovered[-1] = (1 << last_block_bits) - 1
        
    code = []
    
    # Greedy Set Cover
    # To find the best candidate, we need to count how many bits are set in (cover_mask[i] & uncovered)
    # This is a bit expensive if done for all 2187 candidates every step.
    # 2187 * 35 operations per step.
    # 200 steps * 2187 * 35 ~ 15M ops. Very fast in Python? Maybe.
    # We can optimize by using numpy if we store as int64 array?
    # But Python lists of ints might be faster for bitwise ops on small numbers?
    
    # Let's try to optimize the "find best" step.
    # We can maintain a "count" array? No, because uncovered changes.
    
    # Alternative: Precompute for each position j, the list of codewords covering it.
    # Then use a priority queue?
    # When we pick a codeword, we decrement the count of all uncovered words it covers.
    # If count becomes 0, it's covered.
    # This is the standard "Maximal Marginal Coverage" approach.
    
    # Let's use the count array approach. It's O(K * Space) initialization and O(K * Deg) per step.
    # K ~ 200, Space = 2187, Deg = 15.
    # Init: 200 * 2187 ~ 400k.
    # Step: 200 * 15 ~ 3000.
    # Total ~ 1M ops. Very fast.
    
    # But we need to find the max count efficiently.
    # We can just scan the array of size 2187.
    # 200 steps * 2187 scans = 437k ops.
    
    # Let's implement this.
    
    # counts[i] = number of uncovered words covered by i
    # uncovered_words: boolean array or set?
    # Let's use a boolean array `is_uncovered` and an integer array `coverage_count`.
    
    is_uncovered = np.ones(space_size, dtype=bool)
    coverage_count = np.zeros(space_size, dtype=np.int32)
    
    # Initialize coverage_count
    # For each word w, count how many codewords cover it? No, we want for each codeword c, how many *uncovered* words it covers.
    # Initially all are uncovered.
    # So coverage_count[c] = len(neighbors[c]) = 15.
    # Wait, this doesn't help distinguish between codewords if all cover 15.
    # We need to update this as we cover words.
    
    # Better:
    # Maintain `uncovered_mask` (boolean array).
    # For each candidate c, score = sum(uncovered_mask[neighbors[c]]).
    # This is what the current program does with `cover_masks @ uncovered`.
    
    # Let's stick to the bitset approach for the greedy phase, it's cleaner for "fast".
    
    # Actually, the current program's greedy is:
    # counts = cover_masks @ uncovered.astype(np.int8)
    # This is a matrix-vector product. 2187x2187 @ 2187.
    # This is ~4.7M multiplications/additions.
    # 200 iterations * 4.7M = 940M ops.
    # In NumPy, this is vectorized and fast, but maybe 0.1-0.5s per iteration?
    # If it takes 0.1s, 200 iterations = 20s.
    # Plus ILS.
    
    # If I use the "count updates" method:
    # 1. Start with `uncovered` set of indices.
    # 2. `score[c]` = number of neighbors of c in `uncovered`.
    # 3. Pick c with max score.
    # 4. Remove neighbors of c from `uncovered`.
    # 5. For each neighbor w of c, for each codeword d that covers w, decrement `score[d]`.
    #    This requires precomputing `covering_codewords[w]` = list of c such that w in neighbors[c].
    #    Size of covering_codewords[w] is 15.
    #    So step 5 is O(15 * 15) = 225 ops.
    #    Step 3 is O(2187) scan.
    #    Total per iteration ~ 2400 ops.
    #    200 iterations ~ 500k ops.
    #    This is MUCH faster than the matrix product.
    
    # Precompute covering_codewords and neighbor sets for O(1) membership
    covering_codewords = [[] for _ in range(space_size)]
    neighbor_sets = [set(n) for n in neighbors]  # For O(1) membership in swap validation
    for c in range(space_size):
        for w in neighbors[c]:
            covering_codewords[w].append(c)
    
    # Precompute all words as lists for fast conversion
    all_words = []
    for idx in range(space_size):
        w = [0]*7
        tmp = idx
        for i in range(6, -1, -1):
            w[i] = tmp % 3
            tmp //= 3
        all_words.append(w)
    
    # Greedy with tie-breaking by index for determinism
    score = np.full(space_size, 15, dtype=np.int32)
    uncovered = np.ones(space_size, dtype=bool)
    uncovered_count = space_size
    
    while uncovered_count > 0:
        # Find max score with tie-breaking by lower index
        max_score = -1
        best_c = -1
        for c in range(space_size):
            if score[c] > max_score:
                max_score = score[c]
                best_c = c
                
        if max_score == 0:
            break
            
        code.append(best_c)
        
        # Remove neighbors of best_c from uncovered
        for w in neighbors[best_c]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1
                # Update scores of all codewords covering w
                for d in covering_codewords[w]:
                    score[d] -= 1
                    
        if time.time() - t_start > deadline - 10:
            break

    # Ensure validity: add any uncovered words
    if uncovered_count > 0:
        for w in range(space_size):
            if uncovered[w]:
                code.append(w)
                uncovered[w] = False
                uncovered_count -= 1
    
    # Deduplicate
    code = list(set(code))
    
    # Local Search to minimize size
    coverage_count = np.zeros(space_size, dtype=np.int32)
    in_code = np.zeros(space_size, dtype=bool)
    for c in code:
        in_code[c] = True
        for w in neighbors[c]:
            coverage_count[w] += 1
    
    # Aggressive pruning: repeatedly remove any redundant codeword
    def prune_code():
        nonlocal code, coverage_count, in_code
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                if np.min(coverage_count[neighbors[c]]) > 1:
                    code.pop(i)
                    in_code[c] = False
                    for w in neighbors[c]:
                        coverage_count[w] -= 1
                    changed = True
                    break
    
    prune_code()
    
    best_code = code[:]
    best_size = len(code)
    
    rng = random.Random(42)
    
    # Local search with restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - t_start > deadline - 5:
            break
        
        # Random perturbation: swap in a few random codewords
        for _ in range(min(3, len(code))):
            if code:
                i = rng.randint(0, len(code)-1)
                c = code[i]
                new_c = rng.randint(0, space_size-1)
                if not in_code[new_c] and new_c != c:
                    # Check validity
                    valid = True
                    for w in neighbors[c]:
                        if coverage_count[w] == 1 and w not in neighbor_sets[new_c]:
                            valid = False
                            break
                    if valid:
                        code[i] = new_c
                        in_code[c] = False
                        in_code[new_c] = True
                        for w in neighbors[c]:
                            coverage_count[w] -= 1
                        for w in neighbors[new_c]:
                            coverage_count[w] += 1
        
        # After perturbation, prune and try to improve
        prune_code()
        
        # Hill climbing: try removals and swaps
        no_improve = 0
        while no_improve < 100 and time.time() - t_start < deadline - 3:
            improved = False
            
            # Try random removal
            if code and rng.random() < 0.5:
                i = rng.randint(0, len(code)-1)
                c = code[i]
                if np.min(coverage_count[neighbors[c]]) > 1:
                    code.pop(i)
                    in_code[c] = False
                    for w in neighbors[c]:
                        coverage_count[w] -= 1
                    improved = True
                    no_improve = 0
                    continue
            
            # Try swap
            if code:
                i = rng.randint(0, len(code)-1)
                c = code[i]
                new_c = rng.randint(0, space_size-1)
                if not in_code[new_c] and new_c != c:
                    valid = True
                    for w in neighbors[c]:
                        if coverage_count[w] == 1 and w not in neighbor_sets[new_c]:
                            valid = False
                            break
                    if valid:
                        code[i] = new_c
                        in_code[c] = False
                        in_code[new_c] = True
                        for w in neighbors[c]:
                            coverage_count[w] -= 1
                        for w in neighbors[new_c]:
                            coverage_count[w] += 1
                        # Try to prune after swap
                        prune_code()
                        if len(code) < best_size:
                            best_code = code[:]
                            best_size = len(code)
                            no_improve = 0
                        else:
                            no_improve += 1
                        improved = True
                        continue
            
            if not improved:
                no_improve += 1
        
        # Update best
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Final verification using precomputed all_words
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        for w in neighbors[c]:
            final_cov[w] += 1
            
    if np.any(final_cov == 0):
        # Fallback to greedy code (should be valid)
        return [all_words[c] for c in code]
    else:
        return [all_words[c] for c in best_code]

def _generic_construct(q, n, radius):
    # Simple fallback
    space_size = q ** n
    neighbors = []
    all_words = []
    for idx in range(space_size):
        w = [0]*n
        tmp = idx
        for i in range(n-1, -1, -1):
            w[i] = tmp % q
            tmp //= q
        all_words.append(w)
        nbs = [idx]
        for pos in range(n):
            d = w[pos]
            for nd in range(q):
                if nd == d: continue
                diff = (nd - d) * (q ** (n-1-pos))
                nbs.append(idx + diff)
        neighbors.append(nbs)
        
    uncovered = np.ones(space_size, dtype=bool)
    code = []
    while uncovered.any():
        counts = np.zeros(space_size, dtype=np.int32)
        for c in range(space_size):
            counts[c] = np.sum(uncovered[neighbors[c]])
        best_c = np.argmax(counts)
        if counts[best_c] == 0: break
        code.append(best_c)
        uncovered[neighbors[best_c]] = False
    return [all_words[c] for c in code]
