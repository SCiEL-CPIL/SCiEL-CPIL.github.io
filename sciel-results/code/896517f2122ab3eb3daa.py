import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But handle generally if possible, or assume these.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # We assume q=3, n=7, radius=1 based on the description.
    # If parameters differ, we might need a fallback, but let's optimize for the given case.
    
    if q**n > 50000:
        # Fallback to simple greedy if space is too large for ILS to be effective in time
        # But 3^7=2187 is small.
        pass

    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (neighbors within radius)
    # balls[i] is a list of indices of words within distance <= radius from all_words[i]
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # Initial Greedy Code
    def get_initial_code():
        uncovered = set(range(num_words))
        code = []
        while uncovered:
            best_c = -1
            best_cov = -1
            # Sample a subset if too slow, but 2187 is small
            for c in range(num_words):
                cov = 0
                for u in balls[c]:
                    if u in uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_c = c
            code.append(best_c)
            for u in balls[best_c]:
                if u in uncovered:
                    uncovered.remove(u)
        return code

    best_code = get_initial_code()
    best_size = len(best_code)
    
    # Setup state for ILS
    # We will operate on indices.
    # State: code_set (set of indices), cnt (list of counts)
    
    def calculate_state(code_list):
        cnt = [0] * num_words
        for c in code_list:
            for u in balls[c]:
                cnt[u] += 1
        num_uncovered = sum(1 for c in cnt if c == 0)
        return set(code_list), cnt, num_uncovered

    current_code, current_cnt, current_num_unc = calculate_state(best_code)
    
    import time
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        # 1. Local Optimization: Deletion
        improved = True
        while improved:
            improved = False
            # Iterate over a copy of the list to allow removal
            for c in list(current_code):
                # Check if c can be removed
                # Condition: all u in balls[c] have cnt[u] > 1
                can_remove = True
                for u in balls[c]:
                    if current_cnt[u] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    current_code.remove(c)
                    for u in balls[c]:
                        current_cnt[u] -= 1
                    # num_uncovered doesn't change because we only removed if cnt > 1
                    improved = True
        
        # Check if we found a better code
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            
        # 2. Perturbation
        # Randomly remove k codewords
        k = random.randint(2, 5)
        if len(current_code) > k:
            to_remove = random.sample(list(current_code), k)
            for c in to_remove:
                current_code.remove(c)
                for u in balls[c]:
                    current_cnt[u] -= 1
                    if current_cnt[u] == 0:
                        current_num_unc += 1
            # Note: current_num_unc needs to be tracked carefully.
            # Let's recompute num_uncovered after perturbation to be safe, or update incrementally.
            # Incremental:
            # We subtracted 1 from cnt[u]. If it became 0, unc increased.
            # But wait, if it was 1, it becomes 0. If it was 0, it becomes -1 (impossible if valid before).
            # Since code was valid before perturbation, cnt[u] >= 1.
            # So if cnt[u] becomes 0, it was 1 before. So num_unc increases by 1 for each such u.
            
        # 3. Repair
        while current_num_unc > 0:
            # Find an uncovered word
            # Pick a random one to avoid bias? Or just scan.
            # Scanning 2187 is fast.
            uncovered_word = -1
            for i in range(num_words):
                if current_cnt[i] == 0:
                    uncovered_word = i
                    break
            
            if uncovered_word == -1:
                break # Should not happen if num_unc > 0
                
            # Find best candidate to add
            best_cand = -1
            best_score = -1
            # Candidates are words c NOT in current_code such that uncovered_word is in balls[c]
            # We can precompute "reverse balls" or just scan all words.
            # Scanning all 2187 words and checking membership in current_code (set) is O(2187).
            # For each candidate, score is count of u in balls[c] with cnt[u]==0.
            
            for c in range(num_words):
                if c in current_code:
                    continue
                if uncovered_word not in balls[c]:
                    continue
                
                score = 0
                for u in balls[c]:
                    if current_cnt[u] == 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand == -1:
                # No candidate found? Should not happen if coverage is possible.
                # But if we are stuck, break.
                break
                
            # Add best_cand
            current_code.add(best_cand)
            for u in balls[best_cand]:
                if current_cnt[u] == 0:
                    current_num_unc -= 1
                current_cnt[u] += 1
                
        # 4. Optional: Local Optimization again after repair?
        # The deletion loop at the start of the next iteration handles this.
        
    return [list(all_words[c]) for c in best_code]
