import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit
    
    # Precompute all words and their indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    total_words = q ** n
    
    # Precompute balls for all words
    ball_idx = [None] * total_words
    ball_set = [None] * total_words
    for w in all_words:
        idx = word_to_idx[w]
        b = [idx]
        for i in range(n):
            for val in range(q):
                if w[i] != val:
                    new_word = list(w)
                    new_word[i] = val
                    b.append(word_to_idx[tuple(new_word)])
        ball_idx[idx] = b
        ball_set[idx] = set(b)
    
    # Convert ball_idx to numpy arrays for faster operations
    ball_idx_np = [np.array(b, dtype=np.int32) for b in ball_idx]
    
    # Precompute reverse balls: for each word, which codewords cover it
    # This is the same as ball_idx since distance is symmetric
    # But we'll use it for coverage counting
    
    # 1. Generate initial codewords using Fast Greedy Set Cover
    covered_mask = np.zeros(total_words, dtype=bool)
    cw_idxs = set()
    
    # Symmetry-guided greedy: pick codewords that are "central" in some sense
    # Use a deterministic ordering based on word sum to break ties consistently
    word_sums = [sum(w) for w in all_words]
    
    # Initialize: pick the word with minimum sum (most "central" in a symmetric sense)
    first_idx = min(range(total_words), key=lambda i: (word_sums[i], i))
    cw_idxs.add(first_idx)
    covered_mask[ball_idx[first_idx]] = True
    
    # Fast greedy loop using precomputed gains
    while not covered_mask.all():
        if time.time() > deadline - 15:
            break
        
        best_idx = -1
        max_new = 0
        best_sum = 0
        
        # Scan all words, track gain and use symmetry for tie-breaking
        for idx in range(total_words):
            if idx in cw_idxs:
                continue
            b = ball_idx_np[idx]
            new_count = int(np.sum(~covered_mask[b]))
            if new_count > max_new:
                max_new = new_count
                best_idx = idx
                best_sum = word_sums[idx]
            elif new_count == max_new and new_count > 0:
                # Tie-breaking: prefer words with lower sum (more symmetric/central)
                if word_sums[idx] < best_sum:
                    best_idx = idx
                    best_sum = word_sums[idx]
                
        if best_idx == -1:
            break
            
        cw_idxs.add(best_idx)
        covered_mask[ball_idx[best_idx]] = True
        
    # Initialize cover_count
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in cw_idxs:
        cover_count[ball_idx[cw]] += 1
        
    # Helper functions
    def is_redundant(cw_idx):
        return np.all(cover_count[ball_idx[cw_idx]] > 1)
        
    def remove_cw(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            cover_count[w_idx] -= 1
        cw_idxs.remove(cw_idx)
        
    def add_cw(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            cover_count[w_idx] += 1
        cw_idxs.add(cw_idx)
    
    def try_2_for_1(c1, c2):
        """Try to replace c1 and c2 with a single codeword. Returns True if successful."""
        if c1 not in cw_idxs or c2 not in cw_idxs or c1 == c2:
            return False
        
        remove_cw(c1)
        remove_cw(c2)
        
        uncovered_mask = (cover_count == 0)
        
        if not np.any(uncovered_mask):
            # Both were jointly redundant
            return True
        
        num_uncovered = int(np.sum(uncovered_mask))
        max_ball_size = 1 + n * (q - 1)
        
        if num_uncovered > max_ball_size:
            add_cw(c1)
            add_cw(c2)
            return False
        
        uncovered_list = np.where(uncovered_mask)[0]
        
        # Candidate must be within distance 1 of all uncovered words
        # Intersection of balls of uncovered words
        # Start with ball of first uncovered word, intersect with others
        if len(uncovered_list) == 1:
            candidates = ball_idx[uncovered_list[0]]
        else:
            # Intersection of balls
            candidate_set = set(ball_idx[uncovered_list[0]])
            for u in uncovered_list[1:]:
                candidate_set &= ball_set[u]
                if not candidate_set:
                    break
            candidates = list(candidate_set)
        
        valid_candidate = -1
        for w_cand in candidates:
            if all(u in ball_set[w_cand] for u in uncovered_list):
                valid_candidate = w_cand
                break
        
        if valid_candidate != -1:
            add_cw(valid_candidate)
            return True
        else:
            add_cw(c1)
            add_cw(c2)
            return False
    
    def try_1_for_1(c_to_swap):
        """Try to swap c_to_swap with another codeword. Returns True if successful."""
        if c_to_swap not in cw_idxs:
            return False
        
        remove_cw(c_to_swap)
        
        uncovered_mask = (cover_count == 0)
        if not np.any(uncovered_mask):
            # Redundant
            return True
        
        num_uncovered = int(np.sum(uncovered_mask))
        max_ball_size = 1 + n * (q - 1)
        
        if num_uncovered > max_ball_size:
            add_cw(c_to_swap)
            return False
        
        uncovered_list = np.where(uncovered_mask)[0]
        
        # Find candidates that cover all uncovered words
        if len(uncovered_list) == 1:
            candidates = ball_idx[uncovered_list[0]]
        else:
            candidate_set = set(ball_idx[uncovered_list[0]])
            for u in uncovered_list[1:]:
                candidate_set &= ball_set[u]
                if not candidate_set:
                    break
            candidates = list(candidate_set)
        
        valid_candidate = -1
        for w_cand in candidates:
            if w_cand == c_to_swap:
                continue
            if all(u in ball_set[w_cand] for u in uncovered_list):
                valid_candidate = w_cand
                break
        
        if valid_candidate != -1:
            add_cw(valid_candidate)
            return True
        else:
            add_cw(c_to_swap)
            return False
    
    # Main Search Loop with multiple restarts
    best_size = len(cw_idxs)
    best_cw_idxs = set(cw_idxs)
    best_cover_count = cover_count.copy()
    
    max_restarts = 5
    restarts = 0
    
    while time.time() < deadline - 10:
        improved = True
        no_improvement_rounds = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Remove redundant codewords (multiple passes)
            redundant_found = True
            while redundant_found and time.time() < deadline - 10:
                redundant_found = False
                cws_to_check = list(cw_idxs)
                random.shuffle(cws_to_check)
                for cw in cws_to_check:
                    if is_redundant(cw):
                        remove_cw(cw)
                        improved = True
                        redundant_found = True
                        break
            
            # Phase 2: Systematic 2-for-1 replacements
            if not improved and time.time() < deadline - 10:
                # Try all pairs systematically, not just random
                cw_list = list(cw_idxs)
                random.shuffle(cw_list)
                for i in range(len(cw_list)):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i + 1, len(cw_list)):
                        if time.time() > deadline - 10:
                            break
                        if try_2_for_1(cw_list[i], cw_list[j]):
                            improved = True
                            break
                    if improved:
                        break
            
            # Phase 3: 1-for-1 swaps to escape local minima
            if not improved and time.time() < deadline - 10:
                attempts = 0
                max_attempts = 30
                while attempts < max_attempts and time.time() < deadline - 10:
                    attempts += 1
                    if not cw_idxs:
                        break
                    c_to_swap = random.choice(list(cw_idxs))
                    if try_1_for_1(c_to_swap):
                        improved = True
                        break
            
            # Track best solution
            if len(cw_idxs) < best_size:
                best_size = len(cw_idxs)
                best_cw_idxs = set(cw_idxs)
                best_cover_count = cover_count.copy()
            
            if not improved:
                no_improvement_rounds += 1
                if no_improvement_rounds >= 3:
                    break
        
        # Restart if stuck
        if time.time() < deadline - 15:
            restarts += 1
            if restarts >= max_restarts:
                break
            
            # Restore best solution and try a different random seed
            cw_idxs = set(best_cw_idxs)
            cover_count = best_cover_count.copy()
            
            # Add some random perturbation
            if cw_idxs:
                # Remove 1-2 random codewords and re-greedy
                to_remove = random.sample(list(cw_idxs), min(2, len(cw_idxs)))
                for cw in to_remove:
                    remove_cw(cw)
                
                # Re-greedy to cover any gaps
                while not np.all(cover_count > 0):
                    if time.time() > deadline - 10:
                        break
                    uncovered = np.where(cover_count == 0)[0]
                    if len(uncovered) == 0:
                        break
                    # Pick a random uncovered word and add its best covering codeword
                    u = random.choice(uncovered.tolist())
                    candidates = ball_idx[u]
                    best_cand = -1
                    best_gain = -1
                    for cand in candidates:
                        if cand in cw_idxs:
                            continue
                        gain = int(np.sum(cover_count[ball_idx[cand]] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand
                    if best_cand != -1:
                        add_cw(best_cand)
                    else:
                        break

    # Use best solution found
    cw_idxs = best_cw_idxs
    result = [list(idx_to_word[idx]) for idx in cw_idxs]
    
    return result
