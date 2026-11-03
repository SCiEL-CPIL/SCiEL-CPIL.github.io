import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin for 150s limit
    
    # Helper functions
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    total_words = q ** n
    
    # Precompute balls for all q^n words using numpy for speed
    # Ball of radius 1: word itself + all words at Hamming distance 1
    # For each position i, change digit to each of q-1 other values
    all_words = np.arange(total_words, dtype=np.int64)
    
    # Precompute ball neighbors as a list of arrays
    balls = []
    for w in all_words:
        w_list = int_to_list(int(w), n)
        neighbors = [w]
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    neighbors.append(list_to_int(new_word))
        balls.append(neighbors)
    
    # Convert to numpy arrays for faster iteration
    balls_np = [np.array(b, dtype=np.int64) for b in balls]
    
    # Generate initial seed using greedy independent set on distance-2 graph
    # This avoids pairs at Hamming distance 2, which are candidates for removal
    covered = np.zeros(total_words, dtype=np.uint8)
    code = set()
    
    # Precompute distance-2 neighbors for greedy selection
    # Two words are at distance 2 if they differ in exactly 2 positions
    # We want to select words such that no two selected words are at distance 2
    
    # Greedy: process words in order, add if not at distance 2 from any existing codeword
    # And if it covers new words
    for w in range(total_words):
        if covered[w]:
            continue
        # Check if w is at distance 2 from any existing codeword
        w_list = int_to_list(w, n)
        is_dist2 = False
        for c in code:
            c_list = int_to_list(c, n)
            diff_count = sum(1 for i in range(n) if w_list[i] != c_list[i])
            if diff_count == 2:
                is_dist2 = True
                break
        if not is_dist2:
            code.add(w)
            # Mark all words within distance 1 of w as covered
            for neighbor in balls_np[w]:
                covered[neighbor] = 1
    
    code_list = list(code)
    
    # Build coverage using numpy
    def compute_coverage(code_set):
        """Compute coverage counts for all words."""
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            coverage[balls_np[c]] += 1
        return coverage
    
    def get_removable(coverage):
        """Find codewords that can be removed without breaking coverage."""
        if time.time() > deadline - 5:
            return None
        
        # A word is uniquely covered if coverage count is 1
        # Find all words with coverage == 1
        unique_words = np.where(coverage == 1)[0]
        
        if len(unique_words) == 0:
            # All words have coverage >= 2, any codeword can be removed
            return code.copy()
        
        # Find which codewords are essential (uniquely cover some word)
        # Build a mapping: for each word, which codeword(s) cover it
        # More efficient: for each codeword, check if it uniquely covers any word
        essential = set()
        for c in code:
            for w in balls_np[c]:
                if coverage[w] == 1:
                    essential.add(c)
                    break
        
        removable = code - essential
        if not removable:
            return None
        return removable
    
    def try_swap(c_to_remove, coverage):
        """Try to find a replacement for c_to_remove."""
        if time.time() > deadline - 5:
            return None
        
        # Find words uniquely covered by c_to_remove
        unique_words = []
        for w in balls_np[c_to_remove]:
            if coverage[w] == 1:
                unique_words.append(w)
        
        if not unique_words:
            # Can just remove
            return None
        
        # Find candidate replacements that cover all unique_words
        # Candidates are in the intersection of balls of all unique_words
        # Use the smallest ball first for efficiency
        unique_words_sorted = sorted(unique_words, key=lambda w: len(balls_np[w]))
        candidate_set = set(balls_np[unique_words_sorted[0]])
        for w in unique_words_sorted[1:]:
            candidate_set &= set(balls_np[w])
            if not candidate_set:
                return None
        
        # Filter candidates: must not be in code
        valid_candidates = [w for w in candidate_set if w not in code]
        
        if not valid_candidates:
            return None
        
        return random.choice(valid_candidates)
    
    def greedy_remove():
        """Aggressively remove redundant codewords."""
        if time.time() > deadline - 10:
            return
        
        coverage = compute_coverage(code)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            removable = get_removable(coverage)
            if removable:
                # Try to remove multiple at once if possible
                removed_any = False
                for c in list(removable):
                    if time.time() > deadline - 10:
                        break
                    # Double-check this codeword is still removable
                    all_ok = True
                    for w in balls_np[c]:
                        if coverage[w] <= 1:
                            all_ok = False
                            break
                    if all_ok:
                        code.remove(c)
                        coverage[balls_np[c]] -= 1
                        improved = True
                        removed_any = True
                if not removed_any:
                    c = random.choice(list(removable))
                    code.remove(c)
                    coverage[balls_np[c]] -= 1
                    improved = True
    
    def greedy_swap():
        """Try to swap codewords to enable further removals."""
        if time.time() > deadline - 10:
            return False
        
        coverage = compute_coverage(code)
        for c in list(code):
            w_new = try_swap(c, coverage)
            if w_new is not None:
                code.remove(c)
                code.add(w_new)
                # Update coverage
                coverage[balls_np[c]] -= 1
                coverage[balls_np[w_new]] += 1
                return True
        return False
    
    def simulated_annealing():
        """Run simulated annealing to escape local minima."""
        if time.time() > deadline - 15:
            return
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        best_code = set(code)
        best_size = len(code)
        
        while time.time() < deadline - 15 and T > T_min:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try removal
                coverage = compute_coverage(code)
                removable = get_removable(coverage)
                if removable:
                    c = random.choice(list(removable))
                    code.remove(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                else:
                    # No removal possible, try swap
                    c = random.choice(list(code))
                    w_new = try_swap(c, coverage)
                    if w_new is not None:
                        code.remove(c)
                        code.add(w_new)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = set(code)
            else:
                # Try swap
                coverage = compute_coverage(code)
                c = random.choice(list(code))
                w_new = try_swap(c, coverage)
                if w_new is not None:
                    code.remove(c)
                    code.add(w_new)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                else:
                    # Try adding a word that covers the most uncovered words
                    # Find uncovered words
                    uncovered = np.where(coverage == 0)[0]
                    if len(uncovered) > 0:
                        # Pick a random uncovered word and find its ball
                        target = random.choice(uncovered.tolist())
                        candidates = [w for w in balls_np[target] if w not in code]
                        if candidates:
                            # Pick candidate that covers most uncovered words
                            best_cand = max(candidates, key=lambda c: np.sum(coverage[balls_np[c]] == 0))
                            code.add(best_cand)
                    else:
                        # All covered, try random addition
                        w_new = random.randint(0, total_words - 1)
                        if w_new not in code:
                            code.add(w_new)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Restore best code found
        code.clear()
        code.update(best_code)
    
    # Main Loop: First greedy removal, then simulated annealing
    no_improve_count = 0
    while time.time() < deadline - 20:
        size_before = len(code)
        
        # Greedy removal phase
        greedy_remove()
        
        # Try swaps
        if greedy_swap():
            continue
        
        size_after = len(code)
        if size_after < size_before:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If no more greedy improvement, run simulated annealing
        if time.time() > deadline - 15:
            break
        
        simulated_annealing()
        
        # After SA, try greedy removal again
        greedy_remove()
        
        # Check if we can do any more swaps
        if not greedy_swap():
            # Try a few more SA iterations
            if time.time() < deadline - 15:
                simulated_annealing()
                greedy_remove()
                if not greedy_swap():
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        break
    
    # Final greedy removal pass
    while time.time() < deadline - 2:
        coverage = compute_coverage(code)
        removable = get_removable(coverage)
        if removable:
            c = random.choice(list(removable))
            code.remove(c)
        else:
            break
    
    # Final verification and cleanup
    # Ensure all words are covered
    final_coverage = compute_coverage(code)
    if np.any(final_coverage == 0):
        # Add words to cover uncovered ones
        uncovered = np.where(final_coverage == 0)[0]
        for w in uncovered:
            if time.time() > deadline - 1:
                break
            # Find a codeword that covers w and add it
            candidates = [c for c in range(total_words) if w in balls_np[c]]
            if candidates:
                code.add(random.choice(candidates))
    
    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result
