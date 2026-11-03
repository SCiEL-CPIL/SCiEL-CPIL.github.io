import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code using Greedy Set Cover with bitset acceleration
    and Local Search (Redundancy Removal + Swap).
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7, r=1

    def int_to_digits(x):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits

    def digits_to_int(d):
        x = 0
        multiplier = 1
        for i in range(n):
            x += d[i] * multiplier
            multiplier *= q
        return x

    # Precompute neighbors (ball of radius 1)
    balls = []
    for i in range(total_words):
        digits = int_to_digits(i)
        ball_set = {i}
        for pos in range(n):
            original_val = digits[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                ball_set.add(digits_to_int(new_digits))
        balls.append(sorted(ball_set))

    # Precompute ball as numpy array for fast operations
    balls_np = [np.array(b, dtype=np.int32) for b in balls]

    # Also precompute: for each word, which balls cover it (dual view)
    # This allows fast candidate generation during local search
    # dual[i] = list of ball indices c such that i in balls_np[c]
    dual = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls_np[c]:
            dual[w].append(c)

    def greedy_cover(randomize=False):
        remaining = np.ones(total_words, dtype=np.int8)
        selected = []
        
        while np.any(remaining):
            best_score = -1
            best_candidate = -1
            max_possible = ball_size
            
            if randomize:
                candidates = list(range(total_words))
                random.shuffle(candidates)
            else:
                candidates = range(total_words)
            
            for c in candidates:
                score = np.sum(remaining[balls_np[c]])
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == max_possible:
                        break
                elif score == best_score and score > 0 and randomize:
                    if random.random() < 0.3:
                        best_candidate = c

            if best_candidate == -1:
                break

            selected.append(best_candidate)
            remaining[balls_np[best_candidate]] = 0
            
        return selected

    def remove_redundants(selected):
        if not selected:
            return selected
        
        coverage = np.zeros(total_words, dtype=np.int8)
        for c in selected:
            coverage[balls_np[c]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                if np.min(coverage[balls_np[c]]) > 1:
                    coverage[balls_np[c]] -= 1
                    selected.pop(i)
                    changed = True
        return selected

    def local_search_swap(selected, max_iterations=200, deadline=None):
        """Try to replace one codeword with another to reduce total size."""
        if not selected:
            return selected
        
        selected = list(selected)
        coverage = np.zeros(total_words, dtype=np.int8)
        for c in selected:
            coverage[balls_np[c]] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            for i in range(len(selected)):
                c = selected[i]
                coverage[balls_np[c]] -= 1
                
                uncovered = np.where(coverage == 0)[0]
                
                if len(uncovered) == 0:
                    selected.pop(i)
                    improved = True
                    break
                else:
                    best_new = -1
                    best_gain = 0
                    
                    # Use dual view to find candidates that cover uncovered words
                    candidate_set = set()
                    for u in uncovered:
                        for cand in dual[u]:
                            candidate_set.add(cand)
                    
                    for cand in candidate_set:
                        gain = np.sum(coverage[balls_np[cand]] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_new = cand
                        if best_gain == len(uncovered):
                            break
                    
                    if best_new != -1 and best_gain == len(uncovered):
                        selected[i] = best_new
                        coverage[balls_np[best_new]] += 1
                        improved = True
                        break
                    else:
                        coverage[balls_np[c]] += 1
        
        return selected

    def local_search_multi_swap(selected, max_iterations=50, deadline=None):
        """Try 2-for-1 or 1-for-2 swaps to reduce size."""
        if not selected:
            return selected
        
        selected = list(selected)
        coverage = np.zeros(total_words, dtype=np.int8)
        for c in selected:
            coverage[balls_np[c]] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # Try removing 2 codewords and replacing with 1
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    c1, c2 = selected[i], selected[j]
                    
                    # Temporarily remove both
                    coverage[balls_np[c1]] -= 1
                    coverage[balls_np[c2]] -= 1
                    
                    uncovered = np.where(coverage == 0)[0]
                    
                    if len(uncovered) == 0:
                        # Both were redundant
                        selected.pop(j)
                        selected.pop(i)
                        improved = True
                        break
                    else:
                        # Find a single codeword that covers all uncovered
                        candidate_set = set()
                        for u in uncovered:
                            for cand in dual[u]:
                                candidate_set.add(cand)
                        
                        best_new = -1
                        best_gain = 0
                        for cand in candidate_set:
                            gain = np.sum(coverage[balls_np[cand]] == 0)
                            if gain > best_gain:
                                best_gain = gain
                                best_new = cand
                            if best_gain == len(uncovered):
                                break
                        
                        if best_new != -1 and best_gain == len(uncovered):
                            # Replace 2 with 1
                            selected.pop(j)
                            selected[i] = best_new
                            coverage[balls_np[best_new]] += 1
                            improved = True
                            break
                        else:
                            # Restore
                            coverage[balls_np[c1]] += 1
                            coverage[balls_np[c2]] += 1
                
                if improved:
                    break
            
            if not improved:
                # Try adding 1 codeword to enable removing 2
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        c1, c2 = selected[i], selected[j]
                        
                        # Remove both temporarily
                        coverage[balls_np[c1]] -= 1
                        coverage[balls_np[c2]] -= 1
                        
                        uncovered = np.where(coverage == 0)[0]
                        
                        if len(uncovered) <= 2:
                            # Check if we can find a single codeword covering uncovered
                            candidate_set = set()
                            for u in uncovered:
                                for cand in dual[u]:
                                    candidate_set.add(cand)
                            
                            for cand in candidate_set:
                                gain = np.sum(coverage[balls_np[cand]] == 0)
                                if gain == len(uncovered):
                                    # Add cand, keep the 2 removed -> net -1
                                    selected.pop(j)
                                    selected[i] = cand
                                    coverage[balls_np[cand]] += 1
                                    improved = True
                                    break
                                else:
                                    coverage[balls_np[c1]] += 1
                                    coverage[balls_np[c2]] += 1
                        
                        if improved:
                            break
                        else:
                            coverage[balls_np[c1]] += 1
                            coverage[balls_np[c2]] += 1
                    
                    if improved:
                        break
        
        return selected

    best_solution = None
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # First run: Deterministic greedy + redundancy removal + local search
    sol = greedy_cover(randomize=False)
    sol = remove_redundants(sol)
    sol = local_search_swap(sol, max_iterations=200, deadline=deadline)
    sol = local_search_multi_swap(sol, max_iterations=30, deadline=deadline)
    if best_solution is None or len(sol) < len(best_solution):
        best_solution = list(sol)
    
    # Subsequent runs: Randomized greedy + redundancy removal + local search
    run_id = 0
    while time.time() < deadline:
        run_id += 1
        sol = greedy_cover(randomize=True)
        sol = remove_redundants(sol)
        sol = local_search_swap(sol, max_iterations=100, deadline=deadline)
        sol = local_search_multi_swap(sol, max_iterations=20, deadline=deadline)
        
        if best_solution is None or len(sol) < len(best_solution):
            best_solution = list(sol)
            if len(best_solution) <= 186:
                break
    
    # Final polish on best solution
    if best_solution is not None and time.time() < deadline:
        best_solution = remove_redundants(best_solution)
        best_solution = local_search_swap(best_solution, max_iterations=150, deadline=deadline)
        best_solution = local_search_multi_swap(best_solution, max_iterations=30, deadline=deadline)
        best_solution = remove_redundants(best_solution)

    # Convert indices back to lists of digits
    return [int_to_digits(c) for c in best_solution]
