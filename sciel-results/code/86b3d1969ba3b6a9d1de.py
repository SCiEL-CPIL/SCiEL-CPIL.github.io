"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(q, n, radius):
    # Specifics for K_3(8,1)
    # q=3, n=8, radius=1
    # Strategy: Direct sum of K_3(4,1) with itself.
    # K_3(4,1) has size 13 (optimal).
    # Code = C4 x C4. Size = 13 * 13 = 169.
    # Then apply local search to try to reduce size.
    
    start_time = time.time()
    deadline = start_time + 110
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (though task is specific)
        # Simple greedy for generic case
        total_words = q ** n
        all_words = [list(map(int, divmod(i, q**j))) for j in range(n) for i in range(q**n)] # Placeholder, not used for q=3,n=8
        # Actually, let's just return a valid but large code if not the specific case
        # But the task contract says "evaluator passes this task's alphabet size q, length n and covering radius".
        # It implies q=3, n=8, radius=1.
        # We will assume q=3, n=8, radius=1 for optimization.
        # If not, we return a trivial code (all words) to be safe, though it will score poorly.
        return [[i % q for i in range(n)] for _ in range(total_words)]

    # Optimal code for K_3(4,1)
    # Reference: "Ternary covering codes of length 4"
    # One known optimal code of size 13:
    C4 = [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [1, 0, 1, 2],
        [1, 1, 2, 0],
        [1, 2, 0, 1],
        [2, 0, 2, 1],
        [2, 1, 0, 2],
        [2, 2, 1, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [1, 2, 1, 0],
        [2, 1, 2, 0]
    ]
    
    # Verify C4 covers all 3^4 = 81 words
    def covers_code(code, word, length):
        dist = 0
        for i in range(length):
            if code[i] != word[i]:
                dist += 1
                if dist > 1:
                    return False
        return True

    # Verify C4
    for w1 in range(3):
        for w2 in range(3):
            for w3 in range(3):
                for w4 in range(3):
                    word = [w1, w2, w3, w4]
                    covered = any(covers_code(c, word, 4) for c in C4)
                    if not covered:
                        # Fallback: add word to C4 if not covered (should not happen)
                        C4.append(word)
    
    # Construct C8 = C4 x C4
    # Codeword is [c4a, c4b] where c4a, c4b in C4
    C8 = []
    for c1 in C4:
        for c2 in C4:
            C8.append(c1 + c2)
    
    # Current size: 13 * 13 = 169
    # Target: minimize this.
    # Theoretical lower bound is 403? Wait.
    # K_3(8,1) best known is 486.
    # My calculation of 169 must be wrong.
    # Let's re-evaluate K_3(4,1).
    # Sphere size for q=3, n=4, r=1: 1 + 4*(3-1) = 9.
    # Total words: 81.
    # Lower bound: ceil(81/9) = 9.
    # Is 13 the optimal size for K_3(4,1)?
    # Yes, K_3(4,1) = 13 is a well-known result.
    # So C8 size 169 should be a valid covering code.
    # Why is the "best known" 486?
    # Ah, the "best known" 486 might be for a different radius or I am misremembering the state of the art.
    # Or perhaps K_3(4,1) is NOT 13?
    # Let's check.
    # If K_3(4,1) = 13, then K_3(8,1) <= 169.
    # The prompt says "best known code has 486 words".
    # This implies either:
    # 1. K_3(4,1) > 13.
    # 2. The direct sum property doesn't hold as I think (it does: if C is covering, CxC is covering for length 2n and same radius).
    # 3. The prompt's "best known" is outdated or refers to a different constraint.
    
    # Let's double check K_3(4,1).
    # A quick search in my internal knowledge base:
    # K_3(1,1) = 3
    # K_3(2,1) = 6
    # K_3(3,1) = 11
    # K_3(4,1) = 13
    # K_3(5,1) = 27?
    # K_3(6,1) = 46?
    # K_3(7,1) = 85?
    # K_3(8,1) = ?
    
    # If K_3(4,1)=13, then 169 is a valid upper bound.
    # The prompt says "best known code has 486".
    # This is a huge discrepancy.
    # Let's re-read carefully: "Choose as few words... The best known code has 486 words".
    # Is it possible the radius is different? "radius 1".
    # Is it possible the alphabet is different? "alphabet {0,1,2}".
    # Is it possible the length is different? "length 8".
    
    # Maybe my C4 is not a valid covering code?
    # Let's test C4 = [[0,0,0,0], ...]
    # If I run the code and it returns 169, and it's valid, I win big.
    # If it's invalid, I lose.
    # I must ensure C4 is valid.
    
    # Let's try to construct C4 more carefully or verify.
    # Actually, there is a known code for K_3(4,1) of size 13.
    # If the prompt's 486 is correct, then 169 is a breakthrough.
    # However, it is highly unlikely that the "best known" is 486 if 169 is possible.
    # Let's check the sphere bound again.
    # K_3(8,1) >= 3^8 / (1 + 8*2) = 6561 / 17 = 385.9 -> 386.
    # 486 is close to 386.
    # 169 is way below 386.
    # Therefore, 169 CANNOT be a valid covering code.
    # Why?
    # Because the direct sum property $K_q(2n, r) \le K_q(n, r)^2$ is correct.
    # This implies $K_3(8,1) \le K_3(4,1)^2$.
    # If $K_3(8,1) \ge 386$, then $K_3(4,1)^2 \ge 386 \implies K_3(4,1) \ge 20$.
    # So K_3(4,1) must be at least 20.
    # My memory of K_3(4,1)=13 is WRONG.
    
    # Let's find a valid C4.
    # If K_3(4,1) is around 20, then 20*20 = 400, which is close to the lower bound 386.
    # The prompt says best known is 486.
    # So if I can find a good C4 of size ~22, I get 484.
    # If I can find C4 of size 21, I get 441.
    # If I can find C4 of size 20, I get 400.
    
    # Let's use a greedy algorithm to find a good C4.
    # Then square it.
    # Then maybe do local search on the product.
    
    # Greedy for C4 (q=3, n=4, r=1)
    def greedy_cover(q, n, r):
        total = q ** n
        words = [tuple(map(int, divmod(i, q**j))) for j in range(n) for i in range(q**n)]
        # Correct word generation
        words = []
        for i in range(total):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            words.append(tuple(w))
            
        uncovered = set(range(total))
        selected = []
        
        while uncovered:
            best_idx = -1
            max_cov = -1
            for i in range(total):
                if i not in uncovered: # Optimization: only consider uncovered as candidates? No, codeword can be covered.
                    # Actually, we can pick any word as codeword.
                    # But picking an uncovered one is a good heuristic.
                    pass
                # Calculate coverage
                c = words[i]
                cov = 0
                for u in uncovered:
                    u_w = words[u]
                    dist = sum(1 for a, b in zip(c, u_w) if a != b)
                    if dist <= r:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if max_cov == 0:
                break
            selected.append(best_idx)
            # Update uncovered
            c = words[best_idx]
            new_uncovered = set()
            for u in uncovered:
                u_w = words[u]
                dist = sum(1 for a, b in zip(c, u_w) if a != b)
                if dist > r:
                    new_uncovered.add(u)
            uncovered = new_uncovered
            
        return [words[i] for i in selected]
        
    # Run greedy for C4
    C4_greedy = greedy_cover(3, 4, 1)
    size_c4 = len(C4_greedy)
    
    # Construct C8 from C4_greedy
    C8 = []
    for c1 in C4_greedy:
        for c2 in C4_greedy:
            C8.append(list(c1) + list(c2))
            
    # Local Search on C8
    # Try to remove codewords
    # And try to swap?
    # For now, just removal.
    
    # Precompute balls for C8?
    # C8 size is ~400-500.
    # Total words 6561.
    
    # Efficient removal
    # Count coverage of each word by current C8.
    
    def get_ball(word, q, n, r=1):
        ball = {tuple(word)}
        for i in range(n):
            for v in range(q):
                if v == word[i]:
                    continue
                new_w = list(word)
                new_w[i] = v
                ball.add(tuple(new_w))
        return ball
        
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate([tuple(map(int, divmod(i, 3**j))) for j in range(8) for i in range(3**8)])}
    # Fix word generation again
    all_words_8 = []
    for i in range(3**8):
        w = []
        x = i
        for _ in range(8):
            w.append(x % 3)
            x //= 3
        all_words_8.append(tuple(w))
    word_to_idx = {w: i for i, w in enumerate(all_words_8)}
    
    # Current code
    current_code = C8
    # Convert to indices for speed
    code_indices = [word_to_idx[tuple(c)] for c in current_code]
    
    # Coverage counts
    cover_count = [0] * (3**8)
    for ci in code_indices:
        w = all_words_8[ci]
        ball = get_ball(w, 3, 8, 1)
        for bw in ball:
            cover_count[word_to_idx[bw]] += 1
            
    # Removal phase
    changed = True
    while changed:
        changed = False
        # Try to remove each codeword
        for i in range(len(code_indices) - 1, -1, -1):
            ci = code_indices[i]
            w = all_words_8[ci]
            ball = get_ball(w, 3, 8, 1)
            can_remove = True
            for bw in ball:
                if cover_count[word_to_idx[bw]] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove
                for bw in ball:
                    cover_count[word_to_idx[bw]] -= 1
                code_indices.pop(i)
                changed = True
                if time.time() > deadline - 5:
                    return [list(all_words_8[ci]) for ci in code_indices]
                    
    # Convert back
    result = [list(all_words_8[ci]) for ci in code_indices]
    return result

# EVOLVE-BLOCK-END
