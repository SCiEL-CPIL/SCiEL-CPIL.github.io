import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    total_words = q ** n
    
    # Precompute all words as numpy array for speed
    all_words = np.zeros((total_words, n), dtype=np.int8)
    for i in range(total_words):
        x = i
        for pos in range(n):
            all_words[i, pos] = x % q
            x //= q
    
    # Precompute balls for radius=1 using direct index manipulation
    if radius == 1:
        # For each word, ball = {word} + {word with one position changed}
        # Total ball size = 1 + n*(q-1)
        ball_size = 1 + n * (q - 1)
        balls = np.zeros((total_words, ball_size), dtype=np.int32)
        
        for i in range(total_words):
            balls[i, 0] = i
            count = 1
            for pos in range(n):
                orig_val = all_words[i, pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Compute index of word that differs only at pos
                    new_idx = i
                    # In base-q representation, position pos has weight q^pos
                    # (least significant digit is position 0)
                    weight = q ** pos
                    new_idx = i - orig_val * weight + val * weight
                    balls[i, count] = new_idx
                    count += 1
        
        # Convert to list of lists for easier manipulation
        balls_list = [balls[i, :ball_size].tolist() for i in range(total_words)]
    else:
        # For general radius, compute balls via Hamming distance
        balls_list = []
        for i in range(total_words):
            w = all_words[i]
            dists = np.sum(w != all_words, axis=1)
            ball = np.where(dists <= radius)[0].tolist()
            balls_list.append(ball)
    
    # Coverage tracking using numpy for speed
    def compute_coverage(code_indices):
        cov = np.zeros(total_words, dtype=np.int8)
        for idx in code_indices:
            cov[balls_list[idx]] += 1
        return cov
    
    def is_covered(cov):
        return np.all(cov > 0)
    
    def count_uncovered(cov):
        return int(np.sum(cov == 0))
    
    # Greedy set cover with numpy acceleration
    def greedy_cover():
        uncovered = np.ones(total_words, dtype=np.int8)
        code_indices = []
        
        while np.any(uncovered):
            # Compute coverage count for each candidate
            counts = np.zeros(total_words, dtype=np.int32)
            for i in range(total_words):
                counts[i] = np.sum(uncovered[balls_list[i]])
            
            best_idx = int(np.argmax(counts))
            if counts[best_idx] == 0:
                break
            
            code_indices.append(best_idx)
            uncovered[balls_list[best_idx]] = 0
        
        return code_indices
    
    # Randomized greedy
    def randomized_greedy():
        uncovered = np.ones(total_words, dtype=np.int8)
        code_indices = []
        
        while np.any(uncovered):
            counts = np.zeros(total_words, dtype=np.int32)
            for i in range(total_words):
                counts[i] = np.sum(uncovered[balls_list[i]])
            
            max_count = int(np.max(counts))
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = int(random.choice(candidates))
            
            code_indices.append(best_idx)
            uncovered[balls_list[best_idx]] = 0
        
        return code_indices
    
    # Minimize by removing redundant codewords
    def minimize(code_indices):
        changed = True
        while changed:
            changed = False
            cov = compute_coverage(code_indices)
            for i in range(len(code_indices)):
                test_cov = cov.copy()
                test_cov[balls_list[code_indices[i]]] -= 1
                if is_covered(test_cov):
                    code_indices = code_indices[:i] + code_indices[i+1:]
                    cov = test_cov
                    changed = True
                    break
        return code_indices
    
    # Local search: swap-based with better candidate generation
    def local_search(initial_code, time_limit):
        current = minimize(initial_code[:])
        cov = compute_coverage(current)
        current_set = set(current)
        best = current[:]
        best_size = len(best)
        
        while time.time() < time_limit:
            if len(current) == 0:
                break
            
            i = random.randint(0, len(current) - 1)
            old = current[i]
            
            # Generate candidates: prefer words that cover many currently uncovered or singly-covered positions
            # When we remove 'old', positions in balls[old] that had cov==1 become uncovered
            # So we want 'new' to cover as many of those as possible
            critical_positions = set()
            for nb in balls_list[old]:
                if cov[nb] == 1:
                    critical_positions.add(nb)
            
            if critical_positions:
                # Score candidates by how many critical positions they cover
                best_new = -1
                best_score = -1
                # Sample a subset of candidates for speed
                sample_size = min(200, total_words)
                candidates = random.sample(range(total_words), sample_size)
                for new in candidates:
                    if new in current_set:
                        continue
                    score = 0
                    for nb in balls_list[new]:
                        if nb in critical_positions:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_new = new
                
                if best_new == -1:
                    continue
                new = best_new
            else:
                new = random.randint(0, total_words - 1)
                if new in current_set:
                    continue
            
            # Compute tentative coverage
            test_cov = cov.copy()
            test_cov[balls_list[old]] -= 1
            test_cov[balls_list[new]] += 1
            
            if not is_covered(test_cov):
                continue
            
            tent_code = current[:]
            tent_code[i] = new
            tent_code = minimize(tent_code)
            
            if len(tent_code) < len(current):
                current = tent_code
                cov = compute_coverage(current)
                current_set = set(current)
                if len(current) < best_size:
                    best = current[:]
                    best_size = len(best)
            elif len(tent_code) == len(current) and random.random() < 0.05:
                current = tent_code
                cov = compute_coverage(current)
                current_set = set(current)
        
        return best
    
    # Strategy 1: Greedy
    code1 = greedy_cover()
    code1 = minimize(code1)
    best_code = code1
    best_size = len(best_code)
    
    # Strategy 2: Randomized greedy (multiple runs)
    remaining_time = deadline - time.time()
    num_runs = max(1, int(remaining_time / 2.0))
    
    for run in range(num_runs):
        if time.time() > deadline - 5.0:
            break
        code = randomized_greedy()
        code = minimize(code)
        if len(code) < best_size:
            best_code = code
            best_size = len(best_code)
    
    # Strategy 3: Local search from best
    remaining = deadline - time.time()
    if remaining > 3.0:
        ls_result = local_search(best_code, deadline - 1.0)
        if len(ls_result) < best_size:
            best_code = ls_result
            best_size = len(best_code)
    
    # Final minimize
    best_code = minimize(best_code)
    
    # Convert to words
    return [all_words[idx].tolist() for idx in best_code]
