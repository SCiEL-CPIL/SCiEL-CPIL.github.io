import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build balls array: balls[w] = [w, neighbors...]
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
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse mapping: for each word w, which ball centers cover it
    inv = [defaultdict(int) for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w][c] += 1
            
    # Convert to lists for faster iteration
    inv_lists = [list(inv[w].keys()) for w in range(total)]
    
    # Precompute balls as a list of arrays for faster access
    balls_list = [balls[c] for c in range(total)]
    
    best_size = total
    best_set = None
    
    # Use multiple random restarts with greedy + pruning + local search
    np.random.seed(42)
    random.seed(42)
    
    def compute_cover_count(selected):
        """Compute cover count vector for a set of codewords."""
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        return cover_count
    
    def greedy_construct(seed=None):
        """Randomized greedy set cover construction with vectorized updates"""
        if seed is not None:
            np.random.seed(seed)
        
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        cand_count = np.full(total, ball_size, dtype=np.int32)
        
        # Track which centers are still "active" (not yet selected)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            # Find the best candidate among active centers
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            
            if max_count <= 0:
                break
            
            # Pick randomly among max-count active candidates for diversity
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            active[best_idx] = False
            
            # Update cover counts and candidate counts
            ball_words = balls[best_idx]
            newly_covered = ball_words[cover_count[ball_words] == 0]
            if len(newly_covered) > 0:
                uncovered_count -= len(newly_covered)
                # Decrement cand_count for all centers that cover newly covered words
                for w in newly_covered:
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
            cover_count[ball_words] += 1
            
            if time.time() > deadline - 8.0:
                break
        
        return selected, uncovered_count
    
    def prune_redundant(selected, time_limit=None):
        """Remove codewords that are not needed - vectorized version"""
        if not selected:
            return selected
        
        if time_limit is None:
            time_limit = deadline - 2.0
        
        cover_count = compute_cover_count(selected)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            new_selected = []
            for c in selected:
                ball_words = balls[c]
                if np.min(cover_count[ball_words]) > 1:
                    cover_count[ball_words] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        return selected
    
    def local_search_swap(selected, time_limit=None):
        """Fast swap-based local search using vectorized operations"""
        if not selected:
            return selected
        
        if time_limit is None:
            time_limit = deadline - 2.0
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            cover_count = compute_cover_count(selected)
            selected_set = set(selected)
            n_sel = len(selected)
            
            # Find codewords with unique words (critical codewords)
            # For each codeword, check if any word in its ball has cover_count == 1
            has_unique = np.zeros(n_sel, dtype=bool)
            unique_words_per_cw = [None] * n_sel
            
            for i, c in enumerate(selected):
                ball_words = balls[c]
                mask = cover_count[ball_words] == 1
                if np.any(mask):
                    has_unique[i] = True
                    unique_words_per_cw[i] = ball_words[mask]
            
            # Try to find a swap for each critical codeword
            for i in range(n_sel):
                if not has_unique[i]:
                    continue
                c = selected[i]
                unique_words = unique_words_per_cw[i]
                
                # Find intersection of inv_lists for all unique words
                common_candidates = set(inv_lists[unique_words[0]])
                for w in unique_words[1:]:
                    common_candidates.intersection_update(inv_lists[w])
                
                # Check if any common candidate is NOT in selected
                for c_prime in common_candidates:
                    if c_prime not in selected_set:
                        # Perform swap
                        selected[i] = c_prime
                        selected_set.discard(c)
                        selected_set.add(c_prime)
                        improved = True
                        break
                if improved:
                    break
            
            # After swaps, try to remove redundant codewords again
            if time.time() < time_limit:
                cover_count = compute_cover_count(selected)
                new_selected = []
                for c in selected:
                    ball_words = balls[c]
                    if np.min(cover_count[ball_words]) > 1:
                        cover_count[ball_words] -= 1
                    else:
                        new_selected.append(c)
                
                if len(new_selected) < len(selected):
                    selected = new_selected
                    improved = True
        
        return selected
    
    def dual_guided_set_cover(seed=None):
        """
        Dual-guided set cover with vectorized updates
        """
        if seed is not None:
            np.random.seed(seed)
        
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        cand_count = np.full(total, ball_size, dtype=np.int32)
        
        # Track which centers are still "active" (not yet selected)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            # Find uncovered words
            uncovered_mask = cover_count == 0
            uncovered_words = np.where(uncovered_mask)[0]
            if len(uncovered_words) == 0:
                break
            
            # For each uncovered word, compute difficulty = 1 / (number of active candidates that cover it)
            # Use a faster approximation: just count active candidates
            difficulties = np.zeros(len(uncovered_words), dtype=np.float64)
            for i, w in enumerate(uncovered_words):
                num_active = 0
                for c in inv_lists[w]:
                    if active[c]:
                        num_active += 1
                difficulties[i] = 1.0 / max(num_active, 1)
            
            # Pick the hardest uncovered word (highest difficulty)
            hardest_idx = np.argmax(difficulties)
            hardest_word = uncovered_words[hardest_idx]
            
            # Among active candidates that cover the hardest word, pick the one with highest cand_count
            candidates_covering = [c for c in inv_lists[hardest_word] if active[c]]
            if not candidates_covering:
                break
            
            # Pick candidate with highest coverage
            best_candidate = max(candidates_covering, key=lambda c: cand_count[c])
            
            selected.append(best_candidate)
            active[best_candidate] = False
            
            # Update cover counts and candidate counts
            ball_words = balls[best_candidate]
            newly_covered = ball_words[cover_count[ball_words] == 0]
            if len(newly_covered) > 0:
                uncovered_count -= len(newly_covered)
                for w in newly_covered:
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
            cover_count[ball_words] += 1
            
            if time.time() > deadline - 8.0:
                break
        
        return selected, uncovered_count
    
    def simulated_annealing_polish(selected, time_limit=None):
        """
        Simulated annealing-inspired local search to escape local minima.
        Uses a penalty-based energy function.
        """
        if not selected:
            return selected
        
        if time_limit is None:
            time_limit = deadline - 2.0
        
        # Start temperature
        T = 50.0
        T_min = 0.1
        cooling = 0.95
        
        cover_count = compute_cover_count(selected)
        selected_set = set(selected)
        
        best_selected = selected.copy()
        best_energy = len(selected)
        
        while T > T_min and time.time() < time_limit:
            # Randomly pick a codeword to potentially remove or replace
            if len(selected) > 10 and random.random() < 0.3:
                # Try removing a codeword
                i = random.randint(0, len(selected) - 1)
                c_remove = selected[i]
                
                # Check if removal leaves any word uncovered
                ball_words = balls[c_remove]
                would_uncover = ball_words[cover_count[ball_words] == 1]
                
                if len(would_uncover) == 0:
                    # Safe to remove
                    selected.pop(i)
                    selected_set.discard(c_remove)
                    cover_count[ball_words] -= 1
                    best_selected = selected.copy()
                    best_energy = len(selected)
                elif len(would_uncover) <= 3:
                    # Try to find a replacement that covers the uncovered words
                    common_candidates = set(inv_lists[would_uncover[0]])
                    for w in would_uncover[1:]:
                        common_candidates.intersection_update(inv_lists[w])
                    
                    found_replacement = False
                    for c_prime in common_candidates:
                        if c_prime not in selected_set:
                            # Perform swap: remove c_remove, add c_prime
                            selected[i] = c_prime
                            selected_set.discard(c_remove)
                            selected_set.add(c_prime)
                            cover_count[ball_words] -= 1
                            cover_count[balls[c_prime]] += 1
                            found_replacement = True
                            break
                    
                    if not found_replacement:
                        # Revert
                        pass
                    else:
                        best_selected = selected.copy()
                        best_energy = len(selected)
            else:
                # Try swapping a codeword with a random other codeword
                if len(selected) < 2:
                    T *= cooling
                    continue
                    
                i = random.randint(0, len(selected) - 1)
                c_old = selected[i]
                
                # Find a random candidate to swap with
                ball_words = balls[c_old]
                unique_words = ball_words[cover_count[ball_words] == 1]
                
                if len(unique_words) > 0:
                    # Find candidates that cover all unique words
                    common_candidates = set(inv_lists[unique_words[0]])
                    for w in unique_words[1:]:
                        common_candidates.intersection_update(inv_lists[w])
                    
                    candidates = [c for c in common_candidates if c not in selected_set]
                    if candidates:
                        c_new = random.choice(candidates)
                        # Perform swap
                        selected[i] = c_new
                        selected_set.discard(c_old)
                        selected_set.add(c_new)
                        cover_count[balls[c_old]] -= 1
                        cover_count[balls[c_new]] += 1
                        best_selected = selected.copy()
                        best_energy = len(selected)
                else:
                    # No unique words, try removing
                    if random.random() < 0.5:
                        selected.pop(i)
                        selected_set.discard(c_old)
                        cover_count[ball_words] -= 1
                        best_selected = selected.copy()
                        best_energy = len(selected)
            
            T *= cooling
        
        return best_selected
    
    # Main loop: try multiple strategies with random restarts
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        seed = 42 + restart_count * 137
        
        # Alternate between greedy and dual-guided
        if random.random() < 0.5:
            selected, uncovered_count = greedy_construct(seed)
        else:
            selected, uncovered_count = dual_guided_set_cover(seed)
        
        # Check if valid
        if uncovered_count > 0:
            continue
        
        # Pruning: iteratively remove codewords that are not needed
        selected = prune_redundant(selected)
        
        # Local Search: Swap-based optimization
        selected = local_search_swap(selected)
        
        # Simulated annealing polish to escape local minima
        selected = simulated_annealing_polish(selected)
        
        # Final pruning
        selected = prune_redundant(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Early stop if we've found a very good solution and time is running low
        if best_size <= 486 and time.time() > deadline - 15.0:
            break
        if time.time() > deadline - 8.0:
            break
    
    # Convert to list of digit lists
    if best_set is None:
        # Fallback: return a large valid set
        best_set = list(range(total))
    
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
