import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # 1. Generate initial seed: Direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Convert to base-3 integers for efficiency
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    # Precompute digits for all words
    total_words = q ** n
    all_words = np.arange(total_words)
    
    # digits[w, i] = digit at position i of word w
    digits = np.zeros((total_words, n), dtype=np.int32)
    for i in range(n):
        div = q ** (n - 1 - i)
        digits[:, i] = (all_words // div) % q
        
    # Precompute balls: for each word, store its neighbors as indices
    balls = []
    for w in range(total_words):
        b = [w]
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.append(list_to_int(new_word))
        balls.append(b)
    
    # Convert balls to numpy arrays for faster operations
    balls_np = [np.array(b, dtype=np.int32) for b in balls]
    
    # Helper to compute coverage and unique coverers
    def compute_coverage(code):
        coverage = np.zeros(total_words, dtype=np.int32)
        unique_coverer = np.full(total_words, -1, dtype=np.int32)
        
        for c in code:
            for w in balls_np[c]:
                if coverage[w] == 0:
                    unique_coverer[w] = c
                elif coverage[w] == 1:
                    unique_coverer[w] = -1
                coverage[w] += 1
        return coverage, unique_coverer

    def get_unique_groups(unique_coverer):
        unique_groups = {}
        for w in range(total_words):
            uc = unique_coverer[w]
            if uc >= 0:
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        return unique_groups

    def batch_remove(code):
        if time.time() > deadline:
            return code, False
        
        coverage, unique_coverer = compute_coverage(code)
        essential = set()
        for w in range(total_words):
            if unique_coverer[w] >= 0:
                essential.add(unique_coverer[w])
        
        removable = code - essential
        if not removable:
            return code, False
            
        # Remove all non-essential codewords
        code -= removable
        return code, True

    def try_swap_1_for_1(code):
        if time.time() > deadline:
            return code, False
            
        coverage, unique_coverer = compute_coverage(code)
        unique_groups = get_unique_groups(unique_coverer)
        
        # Sort by size of unique group to find easy swaps first
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if len(U_c) == 0:
                continue
            
            # Find a word w_new that covers all words in U_c
            # w_new must be in the intersection of balls[w] for all w in U_c
            # Start with the ball of the first word in U_c
            
            candidates_w = balls_np[U_c[0]]
            
            # Sort candidates by how many U_c words they cover (prioritize better ones)
            scored_candidates = []
            for w_new in candidates_w:
                if w_new in code:
                    continue
                # Count how many words in U_c are covered by w_new
                ball_new = balls_np[w_new]
                count = sum(1 for w in U_c if w in ball_new)
                if count == len(U_c):
                    scored_candidates.append((0, w_new))  # Perfect cover, highest priority
                elif count > 0:
                    scored_candidates.append((count, w_new))
            
            # Try perfect covers first
            for _, w_new in scored_candidates:
                ball_new = balls_np[w_new]
                valid = all(w in ball_new for w in U_c)
                if valid:
                    code.remove(c)
                    code.add(w_new)
                    return code, True
                    
        return code, False

    def try_swap_2_for_1(code):
        if time.time() > deadline:
            return code, False
            
        coverage, unique_coverer = compute_coverage(code)
        unique_groups = get_unique_groups(unique_coverer)
        
        # We look for two codewords c1, c2 such that their unique words U_c1 ∪ U_c2
        # can be covered by a single word w_new.
        
        # Sort groups by size
        groups_list = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        checked_pairs = set()
        
        for i in range(len(groups_list)):
            if time.time() > deadline:
                return code, False
            c1, U1 = groups_list[i]
            if len(U1) == 0: continue
            
            for j in range(i + 1, len(groups_list)):
                c2, U2 = groups_list[j]
                if len(U2) == 0: continue
                
                # If sum of sizes is too large, intersection of balls is likely empty
                if len(U1) + len(U2) > 12:
                    continue
                    
                pair_key = (min(c1, c2), max(c1, c2))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                U_union = U1 + U2
                
                # Find w_new covering U_union
                candidates_w = balls_np[U_union[0]]
                
                for w_new in candidates_w:
                    if w_new in code:
                        continue
                    
                    ball_new = balls_np[w_new]
                    valid = all(w in ball_new for w in U_union)
                    
                    if valid:
                        code.remove(c1)
                        code.remove(c2)
                        code.add(w_new)
                        return code, True
                        
        return code, False

    def repair(code):
        if time.time() > deadline:
            return code, False
        
        # Find uncovered words
        covered = np.zeros(total_words, dtype=bool)
        for c in code:
            covered[balls_np[c]] = True
        
        uncovered = np.where(~covered)[0]
        
        if len(uncovered) == 0:
            return code, False
        
        # Add a codeword that covers the most uncovered words
        best_c = None
        best_count = 0
        
        # Optimization: Only check words that are neighbors of uncovered words?
        # A word w can only cover uncovered words if it is in the ball of one of them.
        # So we only need to check candidates from the union of balls of uncovered words.
        
        candidate_set = set()
        for w in uncovered:
            candidate_set.update(balls_np[w].tolist())
        
        # Remove words already in code
        candidate_set -= code
        
        for w_new in candidate_set:
            count = np.sum(covered[balls_np[w_new]] == False)
            if count > best_count:
                best_count = count
                best_c = w_new
        
        if best_c is not None and best_count > 0:
            code.add(best_c)
            return code, True
        
        return code, False

    def greedy_init():
        code = set()
        covered = np.zeros(total_words, dtype=bool)
        
        # Randomize order of words to break symmetry
        order = list(range(total_words))
        random.shuffle(order)
        
        # Greedy: repeatedly pick word covering most uncovered
        # To speed up, we can just pick the first word in random order that covers something new?
        # No, standard greedy picks max.
        # With 729 words, checking all 729 candidates each step is 729*73 ~ 50k ops, which is fast.
        
        while True:
            if time.time() > deadline:
                break
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            
            best_c = None
            best_count = 0
            
            # Check all words? Or just a subset?
            # Checking all 729 words is fine.
            for w in range(total_words):
                if w in code:
                    continue
                count = np.sum(covered[balls_np[w]] == False)
                if count > best_count:
                    best_count = count
                    best_c = w
            
            if best_c is None or best_count == 0:
                break # Should not happen if uncovered > 0
            
            code.add(best_c)
            covered[balls_np[best_c]] = True
            
        return code

    # Main Loop
    # Try multiple initializations
    best_code = None
    best_size = float('inf')
    
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 20: # Leave time for final polish
            break
            
        if r == 0:
            # Tetracode sum
            initial_codewords = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=n - 4):
                    word = h + list(tail)
                    initial_codewords.append(list_to_int(word))
            code = set(initial_codewords)
        elif r == 1:
            # Greedy init
            code = greedy_init()
        elif r == 2:
            # Greedy init with different random seed
            random.seed(r * 12345 + 678)
            code = greedy_init()
        else:
            # Perturbed Greedy: Remove random few from greedy init
            code = greedy_init()
            # Remove 2-4 random
            k = random.randint(2, 4)
            to_remove = random.sample(list(code), min(k, len(code)))
            for c in to_remove:
                code.remove(c)
            # Repair
            while True:
                code, changed = repair(code)
                if not changed:
                    break

        # Local Search
        made_progress = True
        while made_progress and time.time() < deadline - 10:
            made_progress = False
            
            # Batch remove
            code, changed = batch_remove(code)
            if changed:
                made_progress = True
                continue
                
            # 1-for-1 Swap
            code, changed = try_swap_1_for_1(code)
            if changed:
                made_progress = True
                continue
                
            # 2-for-1 Swap
            code, changed = try_swap_2_for_1(code)
            if changed:
                made_progress = True
                continue
                
            # Perturbation: Remove 2-4 random, repair
            if len(code) > 10 and random.random() < 0.4:
                k = random.randint(2, 5)
                to_remove = random.sample(list(code), min(k, len(code)))
                for c in to_remove:
                    code.remove(c)
                # Repair
                while True:
                    code, changed = repair(code)
                    if not changed:
                        break
                made_progress = True # Force another loop iteration to re-evaluate
            
        # Final polish
        while time.time() < deadline - 5:
            code, changed = batch_remove(code)
            if not changed:
                break
            code, changed = try_swap_1_for_1(code)
            if not changed:
                break
            code, changed = try_swap_2_for_1(code)
            if not changed:
                break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = set(code)
            
    # Convert back to list of lists
    if best_code is None:
        # Fallback
        best_code = set()
        
    result = [int_to_list(c, n) for c in best_code]
    return result

# EVOLVE-BLOCK-END
