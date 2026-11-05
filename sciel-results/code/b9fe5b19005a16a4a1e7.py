import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Map word tuple to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls B(i, radius) for all i
    # Since radius is 1, B(i, 1) = {i} U {j : dist(i, j) = 1}
    balls = []
    for i in range(total_words):
        word = idx_to_word(i)
        ball = [i]
        # Generate all neighbors at distance 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = word.copy()
                new_word[pos] = new_val
                ball.append(word_to_idx(new_word))
        balls.append(ball)

    # State
    covered = [False] * total_words
    code = []
    # Set of indices in code for fast lookup? 
    # Actually, we just need to iterate through code.
    
    # Helper to add a codeword
    def add_codeword(idx):
        code.append(idx)
        for j in balls[idx]:
            covered[j] = True

    # Helper to check if a codeword is redundant
    # A codeword c is redundant if for every word w in B(c, 1), 
    # there exists another codeword c' in code \ {c} such that w in B(c', 1).
    # This is equivalent to: count_covered[w] > 1 for all w in B(c, 1)?
    # No, we need to know if w is covered by SOME OTHER codeword.
    
    # To make local search efficient, we can maintain a count of how many codewords cover each word.
    cover_count = [0] * total_words
    
    def add_codeword_with_count(idx):
        code.append(idx)
        for j in balls[idx]:
            cover_count[j] += 1
            covered[j] = True

    def remove_codeword(idx):
        # Find index of idx in code
        # This is O(K). K is ~200. Fine.
        try:
            pos = code.index(idx)
        except ValueError:
            return
        code.pop(pos)
        for j in balls[idx]:
            cover_count[j] -= 1
            if cover_count[j] == 0:
                covered[j] = False

    # Greedy Phase
    # We want to pick c that maximizes number of new words covered.
    # Since the space is small, we can just iterate all candidates.
    while not all(covered):
        best_idx = -1
        best_gain = -1
        for c in range(total_words):
            gain = 0
            for j in balls[c]:
                if not covered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = c
                if gain == 15: # Max possible for q=3, n=7, r=1 is 1 + 7*2 = 15
                    break
        if best_gain == 0:
            break # Should not happen if not all covered
        add_codeword_with_count(best_idx)
        
        if time.time() - start_time > time_limit * 0.1:
            break

    # Local Search Phase
    # 1. Try to remove redundant codewords
    # 2. Try to swap/kick
    
    def is_redundant(idx):
        for j in balls[idx]:
            if cover_count[j] == 1:
                return False
        return True

    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try removals
        for i in range(len(code)):
            if is_redundant(code[i]):
                remove_codeword(code[i])
                improved = True
                # Restart loop as indices changed
                break
        
        if not improved:
            # Try a kick: remove a random codeword, then greedily fix
            if len(code) > 0:
                kick_idx = random.choice(code)
                old_code = code.copy()
                old_cover_count = cover_count.copy()
                old_covered = covered.copy()
                
                remove_codeword(kick_idx)
                
                # Greedily add words to cover uncovered ones
                while not all(covered):
                    best_idx = -1
                    best_gain = -1
                    for c in range(total_words):
                        if c in code: # Avoid adding existing? 
                            # Actually, adding existing is useless for gain if we count unique.
                            # But if we just look for gain, existing ones have gain 0 for already covered.
                            # We can just check all.
                            pass
                        gain = 0
                        for j in balls[c]:
                            if not covered[j]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = c
                    if best_gain == 0:
                        break
                    add_codeword_with_count(best_idx)
                    
                    if time.time() - start_time > time_limit:
                        break
                
                if len(code) < len(old_code):
                    improved = True
                    # Keep new state
                else:
                    # Revert
                    code = old_code
                    cover_count = old_cover_count
                    covered = old_covered

    # Final cleanup: try removals again
    while (time.time() - start_time < time_limit):
        removed = False
        for i in range(len(code)):
            if is_redundant(code[i]):
                remove_codeword(code[i])
                removed = True
                break
        if not removed:
            break

    # Convert indices back to words
    return [idx_to_word(idx) for idx in code]
