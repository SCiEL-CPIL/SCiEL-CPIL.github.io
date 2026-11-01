import time
import numpy as np
import random
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
    
    # Precompute balls using positional weights
    weights = [3 ** (n - 1 - i) for i in range(n)]
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
    
    # Build inverse mapping: for each word, which codewords cover it
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    
    # Convert inv_lists to numpy arrays for faster access
    inv_arrays = [np.array(lst, dtype=np.int32) for lst in inv_lists]
    
    # Precompute overlap matrix: overlap[c1][c2] = number of words in common
    # But this is 6561x6561 = 43M entries, too large. Use on-the-fly instead.
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    def dual_guided_init():
        """
        Dual-guided set cover: maintain dual prices (shadow prices) for each word.
        Words that are 'hard' to cover (few remaining candidates) get higher prices.
        Pick codeword maximizing sum of dual prices over its ball.
        This is a different paradigm than pure greedy.
        """
        selected = []
        selected_set = set()
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_count = total
        
        # Dual prices: initialized to 1 for all words
        # Words with fewer covering candidates get higher priority
        num_coverers = np.array([len(inv_lists[w]) for w in range(total)], dtype=np.float64)
        dual_prices = 1.0 / num_coverers  # Harder words get higher price
        
        # Track active codewords
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0 and time.time() < deadline - 12.0:
            # Compute score for each active codeword: sum of dual prices of uncovered words in its ball
            scores = np.zeros(total, dtype=np.float64)
            
            # Vectorized computation
            # For each codeword c, score[c] = sum of dual_prices[w] for w in balls[c] where cover_count[w] == 0
            # We can compute this as: for each word w with cover_count[w]==0, add dual_prices[w] to all codewords covering w
            
            uncovered_mask = (cover_count == 0)
            if not np.any(uncovered_mask):
                break
            
            # For efficiency, process in batches
            uncovered_words = np.where(uncovered_mask)[0]
            
            # Reset scores
            scores.fill(0.0)
            
            # Add contributions from each uncovered word
            for w in uncovered_words:
                price = dual_prices[w]
                if price == 0:
                    continue
                # All codewords covering w get this price added
                covering = inv_arrays[w]
                scores[covering] += price
            
            # Only consider active codewords
            scores[~active] = -np.inf
            
            max_score = np.max(scores)
            if max_score <= 0:
                break
            
            # Get candidates with max score (with some randomness for diversity)
            top_candidates = np.where(np.isclose(scores, max_score, rtol=0, atol=1e-10))[0]
            if len(top_candidates) == 0:
                break
            
            # Pick randomly from top candidates for diversity
            best_idx = top_candidates[np.random.randint(len(top_candidates))]
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            active[best_idx] = False
            
            # Update coverage
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                cover_count[w] += 1
            
            # Update dual prices: words that are now covered get lower price
            # Words that still have few candidates get higher price
            # Re-compute num_coverers for uncovered words
            if uncovered_count > 0:
                for w in range(total):
                    if cover_count[w] == 0:
                        # Count how many active codewords still cover w
                        cnt = 0
                        for c in inv_arrays[w]:
                            if active[c]:
                                cnt += 1
                        if cnt > 0:
                            dual_prices[w] = 1.0 / cnt
                        else:
                            dual_prices[w] = 0.0
                    else:
                        dual_prices[w] = 0.0
        
        return selected, selected_set, cover_count, uncovered_count
    
    def prune_redundant(selected, selected_set, cover_count):
        """Remove all redundant codewords (those whose ball is fully covered by others)."""
        changed = True
        while changed and time.time() < deadline - 6.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected_set.discard(c)
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, selected_set, cover_count
    
    def two_exchange_local_search(selected, selected_set, cover_count):
        """
        Different local search: try to remove a codeword and add a different one.
        Focus on reducing the number of codewords while maintaining coverage.
        Use a different acceptance criterion: minimize max cover count (balance).
        """
        T = 3.0
        T_min = 0.01
        alpha = 0.9998
        stall_count = 0
        
        def get_objective(cover_count):
            """Objective: number of codewords + penalty for imbalance."""
            # Primary: minimize size (handled by removals)
            # Secondary: balance coverage
            cc = cover_count[cover_count > 0]
            if len(cc) == 0:
                return 0
            return float(len(selected)) + 0.1 * float(np.std(cc))
        
        current_obj = get_objective(cover_count)
        
        while time.time() < deadline - 3.0 and T > T_min:
            if len(selected) == 0:
                break
            
            # Try to remove a random codeword
            c_old = selected[random.randint(0, len(selected) - 1)]
            ball_c_old = balls[c_old]
            
            # Check if we can remove it (all words still covered)
            if np.min(cover_count[ball_c_old]) > 1:
                # Can remove directly
                cover_count[ball_c_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                stall_count = 0
                T *= alpha
                continue
            
            # Need to find replacement
            # Find words that would become uncovered
            critical_words = ball_c_old[cover_count[ball_c_old] == 1]
            
            if len(critical_words) == 0:
                cover_count[ball_c_old] -= 1
                selected.remove(c_old)
                selected_set.discard(c_old)
                stall_count = 0
                T *= alpha
                continue
            
            # Find candidate codewords that cover all critical words
            candidates = None
            if len(critical_words) <= 3:
                common = None
                for w in critical_words:
                    if common is None:
                        common = set(inv_arrays[w].tolist())
                    else:
                        common.intersection_update(inv_arrays[w].tolist())
                if common:
                    candidates = [c for c in common if c != c_old and c not in selected_set]
                else:
                    candidates = []
            else:
                # Too many critical words, skip this removal attempt
                candidates = []
            
            if not candidates:
                stall_count += 1
                if stall_count > 150:
                    break
                T *= alpha
                continue
            
            # Pick a candidate
            c_new = candidates[random.randint(0, len(candidates) - 1)]
            
            # Simulate swap
            old_counts = cover_count.copy()
            cover_count[ball_c_old] -= 1
            cover_count[balls[c_new]] += 1
            
            if np.any(cover_count < 0):
                cover_count = old_counts
                stall_count += 1
                if stall_count > 150:
                    break
                T *= alpha
                continue
            
            new_obj = get_objective(cover_count)
            delta = new_obj - current_obj
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                selected.remove(c_old)
                selected_set.discard(c_old)
                selected.append(c_new)
                selected_set.add(c_new)
                current_obj = new_obj
                stall_count = 0
            else:
                cover_count = old_counts
                stall_count += 1
                if stall_count > 150:
                    break
            
            # Periodic pruning
            if stall_count == 0 and len(selected) % 5 == 0:
                selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
                current_obj = get_objective(cover_count)
            
            T *= alpha
            if T < T_min:
                break
        
        return selected, selected_set, cover_count
    
    def add_remove_local_search(selected, selected_set, cover_count):
        """
        Different neighborhood: try adding a codeword and removing two, or vice versa.
        This explores a different part of the solution space.
        """
        T = 2.0
        T_min = 0.01
        alpha = 0.9999
        stall_count = 0
        
        while time.time() < deadline - 3.0 and T > T_min:
            if len(selected) < 2:
                break
            
            # Try to remove two codewords and add one
            idx1 = random.randint(0, len(selected) - 1)
            idx2 = random.randint(0, len(selected) - 1)
            if idx1 == idx2:
                continue
            
            c1 = selected[idx1]
            c2 = selected[idx2]
            
            # Find words that would become uncovered
            ball1 = balls[c1]
            ball2 = balls[c2]
            
            # Words covered only by c1 or c2
            combined = np.concatenate([ball1, ball2])
            unique_combined = np.unique(combined)
            
            # Count coverage without c1 and c2
            temp_cover = cover_count.copy()
            temp_cover[ball1] -= 1
            temp_cover[ball2] -= 1
            
            if np.any(temp_cover < 0):
                continue
            
            # Find words that would become uncovered
            uncovered_after_removal = unique_combined[temp_cover[unique_combined] == 0]
            
            if len(uncovered_after_removal) == 0:
                # Can remove both
                cover_count[ball1] -= 1
                cover_count[ball2] -= 1
                selected.remove(c1)
                selected.remove(c2)
                selected_set.discard(c1)
                selected_set.discard(c2)
                stall_count = 0
                T *= alpha
                continue
            
            # Find a codeword that covers all uncovered words
            if len(uncovered_after_removal) <= 3:
                common = None
                for w in uncovered_after_removal:
                    if common is None:
                        common = set(inv_arrays[w].tolist())
                    else:
                        common.intersection_update(inv_arrays[w].tolist())
                
                if common:
                    candidates = [c for c in common if c not in selected_set]
                    if candidates:
                        c_new = candidates[random.randint(0, len(candidates) - 1)]
                        
                        # Simulate
                        old_cover = cover_count.copy()
                        cover_count[ball1] -= 1
                        cover_count[ball2] -= 1
                        cover_count[balls[c_new]] += 1
                        
                        if np.any(cover_count < 0):
                            cover_count = old_cover
                            continue
                        
                        # Check if this reduces size
                        if len(selected) > 2:
                            selected.remove(c1)
                            selected.remove(c2)
                            selected_set.discard(c1)
                            selected_set.discard(c2)
                            selected.append(c_new)
                            selected_set.add(c_new)
                            stall_count = 0
                        T *= alpha
                        continue
            
            stall_count += 1
            if stall_count > 100:
                break
            T *= alpha
        
        return selected, selected_set, cover_count
    
    # Main search loop: multiple restarts with dual-guided init + different local searches
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        
        # Use dual-guided initialization
        selected, selected_set, cover_count, uncovered_count = dual_guided_init()
        
        if uncovered_count > 0:
            continue
        
        # Prune initial solution
        selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Apply two-exchange local search
        selected, selected_set, cover_count = two_exchange_local_search(selected, selected_set, cover_count)
        selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Apply add-remove local search if time permits
        if time.time() < deadline - 10.0:
            selected, selected_set, cover_count = add_remove_local_search(selected, selected_set, cover_count)
            selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected.copy()
        
        # If we're doing well, do another round
        if len(selected) <= best_size + 2 and time.time() < deadline - 12.0:
            selected, selected_set, cover_count = two_exchange_local_search(selected, selected_set, cover_count)
            selected, selected_set, cover_count = prune_redundant(selected, selected_set, cover_count)
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected.copy()
    
    # Convert to list of digit lists
    if best_set is None:
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
