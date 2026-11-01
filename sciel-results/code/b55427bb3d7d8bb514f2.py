import time
import numpy as np
import random
import itertools
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls as lists for faster intersection
    balls = [None] * total_words
    for i in range(total_words):
        w = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls[i] = ball

    # Convert balls to sets for intersection operations
    ball_sets = [set(b) for b in balls]

    def prune_cover(indices):
        indices = list(indices)
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                necessary = False
                for w in balls[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    for w in balls[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    def local_search(code_indices, max_time=5.0):
        """Try to reduce code size by removals, swaps, and 2-for-1 moves."""
        best = list(code_indices)
        best_size = len(best)
        local_start = time.time()
        
        def check_cover(code_list):
            """Check if code_list covers all words. Return True if yes."""
            covered = np.zeros(total_words, dtype=bool)
            for c in code_list:
                for w in balls[c]:
                    covered[w] = True
            return covered.all()
        
        def try_remove(best_list):
            """Try removing each codeword one at a time."""
            improved = True
            while improved:
                improved = False
                for i in range(len(best_list)):
                    if time.time() > local_start + max_time:
                        return best_list, False
                    remaining = best_list[:i] + best_list[i+1:]
                    if check_cover(remaining):
                        best_list = remaining
                        improved = True
                        break
            return best_list, improved
        
        def try_swap(best_list):
            """Try replacing one codeword with another to enable a removal."""
            improved = True
            while improved:
                improved = False
                if time.time() > local_start + max_time:
                    return best_list, False
                for i in range(len(best_list)):
                    if time.time() > local_start + max_time:
                        return best_list, False
                    # Find words only covered by best_list[i]
                    covered_by_others = np.zeros(total_words, dtype=bool)
                    for j, c in enumerate(best_list):
                        if j == i:
                            continue
                        for w in balls[c]:
                            covered_by_others[w] = True
                    unique_to_i = [w for w in balls[best_list[i]] if not covered_by_others[w]]
                    
                    # Try swapping best_list[i] with a candidate that covers unique_to_i
                    for candidate in range(total_words):
                        if candidate in best_list:
                            continue
                        if time.time() > local_start + max_time:
                            return best_list, False
                        # Check if candidate covers all words uniquely covered by best_list[i]
                        candidate_ball = set(balls[candidate])
                        if all(w in candidate_ball for w in unique_to_i):
                            new_list = best_list[:i] + [candidate] + best_list[i+1:]
                            if check_cover(new_list):
                                best_list = new_list
                                improved = True
                                break
                    if improved:
                        break
            return best_list, improved
        
        def try_2for1(best_list):
            """Try replacing two codewords with one (if possible)."""
            improved = True
            while improved:
                improved = False
                if time.time() > local_start + max_time:
                    return best_list, False
                for i in range(len(best_list)):
                    if time.time() > local_start + max_time:
                        return best_list, False
                    for j in range(i+1, len(best_list)):
                        if time.time() > local_start + max_time:
                            return best_list, False
                        # Find words covered only by best_list[i] or best_list[j]
                        covered_by_others = np.zeros(total_words, dtype=bool)
                        for k, c in enumerate(best_list):
                            if k == i or k == j:
                                continue
                            for w in balls[c]:
                                covered_by_others[w] = True
                        
                        unique_to_pair = []
                        for w in balls[best_list[i]]:
                            if not covered_by_others[w]:
                                unique_to_pair.append(w)
                        for w in balls[best_list[j]]:
                            if not covered_by_others[w] and w not in unique_to_pair:
                                unique_to_pair.append(w)
                        
                        # Try to find a single codeword that covers all unique_to_pair
                        if len(unique_to_pair) == 0:
                            # Both are redundant, just remove one
                            new_list = best_list[:i] + best_list[i+1:]
                            if check_cover(new_list):
                                best_list = new_list
                                improved = True
                                break
                        elif len(unique_to_pair) <= 19:  # Ball size is 1+6*2=19
                            for candidate in range(total_words):
                                if candidate in best_list:
                                    continue
                                if time.time() > local_start + max_time:
                                    return best_list, False
                                candidate_ball = set(balls[candidate])
                                if all(w in candidate_ball for w in unique_to_pair):
                                    new_list = best_list[:i] + [candidate] + best_list[i+1:j] + best_list[j+1:]
                                    if check_cover(new_list):
                                        best_list = new_list
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
            return best_list, improved
        
        # Iteratively apply improvements
        while time.time() < local_start + max_time:
            changed = False
            
            best_list, rem_improved = try_remove(best)
            if rem_improved:
                best = best_list
                best_size = len(best)
                changed = True
            
            if not changed and time.time() < local_start + max_time - 0.5:
                best_list, swap_improved = try_swap(best)
                if swap_improved:
                    best = best_list
                    best_size = len(best)
                    changed = True
            
            if not changed and time.time() < local_start + max_time - 0.5:
                best_list, pair_improved = try_2for1(best)
                if pair_improved:
                    best = best_list
                    best_size = len(best)
                    changed = True
            
            if not changed:
                break
        
        return best

    best_code = None
    best_size = float('inf')
    
    # 1. Try Direct Product Construction
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        dp_indices = prune_cover(dp_indices)
        if len(dp_indices) < best_size:
            best_size = len(dp_indices)
            best_code = dp_indices

    # 2. Randomized Greedy with improved efficiency
    num_restarts = 200  # More restarts since each is faster
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Use different randomization strategies for diversity
        strategy = r % 4
        
        if strategy == 0:
            # Standard greedy with random tie-breaking
            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                best_gain = 0
                best_candidates = []
                
                for c in range(total_words):
                    gain = len(ball_sets[c] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [c]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(c)
                
                if not best_candidates:
                    break
                
                if len(best_candidates) > 1 and random.random() < 0.3:
                    top_k = min(5, len(best_candidates))
                    best_c = random.choice(best_candidates[:top_k])
                else:
                    best_c = random.choice(best_candidates)
                
                code_indices.append(best_c)
                uncovered -= ball_sets[best_c]
                
                if time.time() > deadline - 20.0:
                    break
            
            if uncovered:
                continue
            
            pruned = prune_cover(code_indices)
            
            if time.time() < deadline - 10.0:
                pruned = local_search(pruned, max_time=min(3.0, deadline - time.time() - 5.0))
            
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
        
        elif strategy == 1:
            # Greedy with randomization: pick from top-K more aggressively
            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                gains = []
                for c in range(total_words):
                    gain = len(ball_sets[c] & uncovered)
                    gains.append((gain, c))
                
                gains.sort(reverse=True)
                
                if not gains or gains[0][0] == 0:
                    break
                
                # Pick from top 10 with weighted probability
                top_k = min(10, len(gains))
                top_gains = gains[:top_k]
                max_gain = top_gains[0][0]
                candidates = [c for g, c in top_gains if g >= max_gain * 0.8]
                best_c = random.choice(candidates)
                
                code_indices.append(best_c)
                uncovered -= ball_sets[best_c]
                
                if time.time() > deadline - 20.0:
                    break
            
            if uncovered:
                continue
            
            pruned = prune_cover(code_indices)
            
            if time.time() < deadline - 10.0:
                pruned = local_search(pruned, max_time=min(3.0, deadline - time.time() - 5.0))
            
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
        
        elif strategy == 2:
            # Start from tetracode product and improve via local search
            if q == 3 and n >= 4 and radius == 1:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                dp_indices = []
                for h in tetracode:
                    for tail in itertools.product(range(3), repeat=n - 4):
                        word = h + list(tail)
                        dp_indices.append(tuple_to_idx(tuple(word)))
                
                pruned = prune_cover(dp_indices)
                
                if time.time() < deadline - 10.0:
                    pruned = local_search(pruned, max_time=min(5.0, deadline - time.time() - 5.0))
                
                if len(pruned) < best_size:
                    best_size = len(pruned)
                    best_code = pruned
        
        else:
            # Greedy with random seed perturbation
            uncovered = set(range(total_words))
            code_indices = []
            
            # Add some random codewords first
            num_random = random.randint(5, 15)
            for _ in range(num_random):
                if not uncovered:
                    break
                c = random.randint(0, total_words - 1)
                code_indices.append(c)
                uncovered -= ball_sets[c]
            
            while uncovered:
                best_gain = 0
                best_candidates = []
                
                for c in range(total_words):
                    gain = len(ball_sets[c] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [c]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(c)
                
                if not best_candidates:
                    break
                
                best_c = random.choice(best_candidates)
                
                code_indices.append(best_c)
                uncovered -= ball_sets[best_c]
                
                if time.time() > deadline - 20.0:
                    break
            
            if uncovered:
                continue
            
            pruned = prune_cover(code_indices)
            
            if time.time() < deadline - 10.0:
                pruned = local_search(pruned, max_time=min(3.0, deadline - time.time() - 5.0))
            
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned

    if best_code is None:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            best_code = list(range(total_words))
    
    # Final verification and cleanup
    # Ensure all codewords are distinct and valid
    seen = set()
    valid_code = []
    for c in best_code:
        word_tuple = idx_to_tuple(c)
        if word_tuple not in seen:
            seen.add(word_tuple)
            valid_code.append(c)
    
    # Verify coverage
    covered = np.zeros(total_words, dtype=bool)
    for c in valid_code:
        for w in balls[c]:
            covered[w] = True
    
    if not covered.all():
        # Fallback: add missing coverage
        for i in range(total_words):
            if not covered[i]:
                valid_code.append(i)
                for w in balls[i]:
                    covered[w] = True
    
    result = [list(idx_to_tuple(c)) for c in valid_code]
    
    return result
