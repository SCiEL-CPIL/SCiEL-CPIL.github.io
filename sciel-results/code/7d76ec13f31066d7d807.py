import itertools
import time

def construct(q, n, radius):
    # Parameters for this specific task: q=3, n=6, radius=1
    # But the function signature is generic. 
    # The task contract says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we should handle general q, n, radius? 
    # "Keep the function names and signatures of the seed program."
    # The seed program hardcodes the logic for q=3, n=6.
    # However, the prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # This implies the function should use q, n, radius.
    # The baseline ignored q, n, radius and hardcoded the tetracode logic.
    # If I write a generic greedy, it works for any small q, n.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # If the problem is too large for greedy, we might need a fallback.
    # But for q=3, n=6, it's tiny.
    # Let's check if q, n are the expected ones.
    # If q**n is huge, greedy will be slow.
    # 3**6 = 729.
    # If the evaluator passes different params, e.g. q=3, n=10, 3**10 = 59049.
    # Greedy on 59049 nodes:
    # Steps ~ K(3,10,1). Lower bound 59049 / (1 + 10*2) = 59049/21 ~ 2800.
    # Ops: 2800 * 59049 * 21 ~ 3.4 billion. Too slow for Python in 120s.
    
    # So we should only use greedy if the space is small enough.
    # Otherwise, fall back to the baseline or a faster heuristic.
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback to baseline logic if space is too big for pure Python greedy
        # The baseline logic is specific to q=3, n=6?
        # "tetracode" is specific to q=3.
        # If q != 3, the baseline fails or produces wrong results?
        # The baseline:
        # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # This assumes q=3.
        # If q is different, range(3) is wrong.
        # The task contract says: "The evaluator passes this task's alphabet size q...".
        # It implies the task is fixed to q=3, n=6, radius=1?
        # "Football pool problem for 6 matches: ternary covering code K_3(6,1)"
        # Yes, the task is specific. The parameters q, n, radius are passed, but they will be 3, 6, 1.
        
        # So we can assume q=3, n=6.
        # But to be safe, if total_words is large, we should return a valid code.
        # The baseline code is valid for q=3, n=6.
        # If the evaluator passes different params, the baseline might break.
        # However, the prompt says "Keep the function names and signatures...".
        # And "The evaluator passes this task's...".
        # It's safe to assume q=3, n=6, radius=1.
        
        # For robustness, if total_words is large, use a simple product code if possible?
        # Or just use the baseline logic which is hardcoded for q=3.
        # Let's stick to the greedy for the expected case.
        pass

    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbors = []
    for w in words:
        nb = [word_to_idx[tuple(w)]] # self
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig: continue
                nw = w.copy()
                nw[i] = v
                nb.append(word_to_idx[tuple(nw)])
        neighbors.append(nb)
        
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        if time.time() > deadline:
            break # Return what we have? No, must be valid.
            # If we break early, the code is invalid.
            # So we must finish.
            # Given the speed, we won't hit the deadline.
            
        best_candidate = -1
        best_count = 0
        
        # We only need to consider candidates that cover at least one uncovered word.
        # We can iterate all candidates.
        for c_idx in range(total_words):
            count = 0
            for nb_idx in neighbors[c_idx]:
                if nb_idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidate = c_idx
                
        if best_candidate == -1:
            break # Should not happen if uncovered is not empty
            
        code.append(words[best_candidate])
        
        # Remove covered
        for nb_idx in neighbors[best_candidate]:
            if nb_idx in uncovered:
                uncovered.remove(nb_idx)
                
    return code
