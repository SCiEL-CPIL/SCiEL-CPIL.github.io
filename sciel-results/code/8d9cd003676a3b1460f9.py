import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases (not expected in this specific task instance but good for robustness)
    import itertools
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute inverse balls: for each word, which codewords cover it
    inv_balls = []
    for w in range(total):
        covering = set()
        for c in range(total):
            if w in balls[c]:
                covering.add(c)
        inv_balls.append(covering)
    
    best_size = total
    best_set = None
    
    rng = np.random.RandomState(42)
    
    def dual_greedy_construct():
        """Dual greedy set cover with shadow prices based on scarcity."""
        selected = []
        uncovered = np.ones(total, dtype=bool)
        # Shadow prices: how many remaining candidates can cover each uncovered word
        # Start with full candidate set
        candidate_count = np.ones(total, dtype=np.int32)
        
        # For each word, track how many candidates can cover it
        for w in range(total):
            candidate_count[w] = len(inv_balls[w])
        
        while np.any(uncovered):
            if time.time() > deadline - 8.0:
                return None
            
            # For each candidate, compute weighted coverage: sum of shadow prices of uncovered words it covers
            # Words with fewer remaining candidates have higher "urgency"
            # Use inverse of candidate_count as weight (higher weight for scarce words)
            
            # Compute urgency weight for each uncovered word
            urgency = np.zeros(total, dtype=np.float64)
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            # Weight = 1 / (candidate_count + 1) to avoid division by zero
            urgency[unc_idx] = 1.0 / (candidate_count[unc_idx].astype(np.float64) + 1.0)
            
            # For each candidate, compute total urgency of uncovered words in its ball
            candidate_scores = np.zeros(total, dtype=np.float64)
            for c in range(total):
                ball_words = balls[c]
                candidate_scores[c] = np.sum(urgency[ball_words])
            
            # Pick candidate with highest score
            max_score = np.max(candidate_scores)
            if max_score == 0:
                break
            
            candidates = np.where(candidate_scores >= max_score * 0.999)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update: mark words in ball as covered, update candidate counts
            ball_words = balls[best_idx]
            newly_covered = ball_words[uncovered[ball_words]]
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
            
            # Update candidate counts: for each newly covered word, 
            # we need to decrement the count for all candidates that cover it
            # But we only care about uncovered words, so this is just bookkeeping
            # Actually, we need to update candidate_count for future urgency computation
            # But since we're removing candidates implicitly (by covering words), 
            # let's just recompute or approximate
            
            # Simpler approach: just track which words are uncovered and use 
            # the original candidate counts but only for uncovered words
            # The key insight is that as we select codewords, the "effective" 
            # number of remaining candidates for each word decreases
        
        return selected
    
    def prune_set(selected, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 4.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected
    
    # Multiple restarts with different seeds
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        rng_seed = 42 + num_restarts * 1000
        rng = np.random.RandomState(rng_seed)
        
        selected = dual_greedy_construct()
        if selected is None:
            continue
        
        # Compute cover counts for pruning
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        # Prune
        selected = prune_set(selected, cover_count)
        
        # Local search: try exchanges
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i, c in enumerate(selected):
                ball_words = balls[c]
                critical = ball_words[cover_count[ball_words] == 1]
                
                if len(critical) == 0:
                    # c is redundant, remove it
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    improved = True
                    break
            
            if not improved:
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
    
    # Final polish
    if best_set is not None and time.time() < deadline - 3.0:
        cover_count = np.zeros(total, dtype=np.int32)
        for c in best_set:
            cover_count[balls[c]] += 1
        
        for _ in range(5):
            changed = False
            new_best = []
            for c in best_set:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_best.append(c)
            best_set = new_best
            if not changed:
                break
        
        best_size = len(best_set)
    
    # Convert best_set to list of digit lists
    if best_set is None:
        result = []
        for w in range(total):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            digits.reverse()
            result.append(digits)
        return result
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
