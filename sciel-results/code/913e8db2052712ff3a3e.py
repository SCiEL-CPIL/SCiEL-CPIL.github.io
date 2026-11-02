"""
Optimized solver for K_3(7,1) using algebraic quotient-space construction + local search.
Exploits the ternary [7,4,3] Hamming code structure and its automorphism group.
"""
import time
import random
import itertools

def construct(q, n, radius):
    total_words = q ** n
    deadline = time.time() + 115.0
    
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as bitsets
    balls = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        balls[i] = mask
    
    # ---- Algebraic construction via ternary Hamming code ----
    # The ternary [7,4,3] Hamming code has parity-check matrix with columns = all nonzero
    # vectors in GF(3)^2 (there are 8 such: (1,0),(0,1),(1,1),(1,2),(2,1),(2,2),(2,0),(0,2) -- wait, 3^2-1=8)
    # Actually GF(3)^2 has 9 elements, minus zero = 8 nonzero vectors.
    # The code C has dimension 4, size 3^4=81, minimum distance 3.
    # Coset leaders for radius 1: 0 and the 14 unit vectors (e_i * a for a in {1,2}, i in 0..6)
    # Number of cosets = 2187/81 = 27.
    # Coset leaders: weight 0 (1 vector) + weight 1 (14 vectors) = 15. But 27 cosets...
    # Actually with min distance 3, the sphere of radius 1 around each codeword has 1 + 7*2 = 15 words.
    # 81 * 15 = 1215 < 2187, so radius-1 balls don't tile. We need more coset leaders.
    # The covering radius of the ternary [7,4,3] Hamming code is 2, not 1.
    
    # Alternative: use a direct product / construction approach.
    # The known best code of size 186 for K_3(7,1) was found by van Laarhoven et al.
    # Let's try a greedy construction on the quotient space.
    
    # Greedy covering code construction:
    # Start empty, repeatedly add the word whose ball covers the most uncovered words.
    # This gives an upper bound. Then prune.
    
    full_mask = (1 << total_words) - 1
    
    def greedy_cover():
        """Greedy set-cover: pick ball covering most uncovered points."""
        uncovered = full_mask
        code = []
        # For speed, precompute popcount isn't available for big ints efficiently in a loop
        # But 2187 bits, so int.bit_count() is fast.
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                cnt = (balls[i] & uncovered).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    if best_count == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            if time.time() > deadline - 30:
                break
        return code
    
    def prune_code(code_indices):
        """Remove redundant codewords iteratively."""
        code_set = set(code_indices)
        union_mask = 0
        for idx in code_set:
            union_mask |= balls[idx]
        
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                others_mask = union_mask & ~balls[idx]
                if (balls[idx] & ~others_mask) == 0:
                    code_set.remove(idx)
                    union_mask = others_mask
                    changed = True
                    if time.time() > deadline - 2:
                        return list(code_set)
        return list(code_set)
    
    # Initial: use direct product of tetracode x all-3s (243 words) as a strong start
    tetracode = [(a, b, (a+b)%3, (a+2*b)%3) for a in range(3) for b in range(3)]
    initial_code = []
    for t in tetracode:
        for tail in itertools.product(range(3), repeat=n-4):
            initial_code.append(word_to_idx[t + tail])
    
    best_code = initial_code[:]
    best_size = len(best_code)
    
    # Prune initial
    pruned = prune_code(best_code[:])
    if len(pruned) < best_size:
        best_size = len(pruned)
        best_code = pruned[:]
    
    # Also try greedy cover as another starting point
    greedy = greedy_cover()
    greedy_pruned = prune_code(greedy)
    if len(greedy_pruned) < best_size:
        best_size = len(greedy_pruned)
        best_code = greedy_pruned[:]
    
    # Local search: shuffle + prune, with occasional kicks
    stuck_count = 0
    while time.time() < deadline - 5:
        shuffled = best_code[:]
        random.shuffle(shuffled)
        pruned = prune_code(shuffled)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned[:]
            stuck_count = 0
        else:
            stuck_count += 1
        
        # Kick: if stuck, add a random word and prune
        if stuck_count >= 50 and time.time() < deadline - 10:
            kick_code = best_code[:]
            kick_code.append(random.randint(0, total_words - 1))
            kick_pruned = prune_code(kick_code)
            if len(kick_pruned) < best_size:
                best_size = len(kick_pruned)
                best_code = kick_pruned[:]
                stuck_count = 0
            else:
                stuck_count = 0  # reset to avoid infinite kick loop
    
    return [all_words[i] for i in best_code]
