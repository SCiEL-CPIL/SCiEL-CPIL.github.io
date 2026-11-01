import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But the function signature is general. We handle the specific case efficiently.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple construction for other cases
        # Direct sum with tetracode if possible
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all words (trivially valid, but large)
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Specific case: q=3, n=6, radius=1
    # Total words: 3^6 = 729
    # Index mapping: word (w0, w1, w2, w3, w4, w5) -> index = w0 + 3*w1 + 9*w2 + 27*w3 + 81*w4 + 243*w5
    
    # Precompute idx_to_word lookup
    idx_to_word = [[0]*6 for _ in range(729)]
    for i in range(729):
        x = i
        for pos in range(6):
            idx_to_word[i][pos] = x % 3
            x //= 3
    
    # Precompute masks using numpy for speed
    # mask[idx] is a Python int (bitset) where bit j is set if word j is within distance 1 of word idx
    masks = [0] * 729
    for w in itertools.product(range(3), repeat=6):
        idx = w[0] + 3*w[1] + 9*w[2] + 27*w[3] + 81*w[4] + 243*w[5]
        mask = 1 << idx
        for i in range(6):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                new_idx = new_w[0] + 3*new_w[1] + 9*new_w[2] + 27*new_w[3] + 81*new_w[4] + 243*new_w[5]
                mask |= (1 << new_idx)
        masks[idx] = mask
    
    all_mask = (1 << 729) - 1
    
    best_solution = None
    best_size = 999
    
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds to be safe (5s margin before 120s limit)
    
    # Greedy with random tie-breaking + local search
    for restart in range(100000):
        if time.time() > deadline:
            break
        
        uncovered = all_mask
        chosen = set()
        solution = []
        
        while uncovered:
            max_gain = -1
            candidates = []
            for i in range(729):
                if i in chosen:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                break
            
            # Pick randomly from candidates with max gain
            c = random.choice(candidates)
            chosen.add(c)
            solution.append(c)
            uncovered &= ~masks[c]
        
        # Reduction step using coverage count array for O(1) checks
        # coverage_count[j] = number of chosen codewords that cover word j
        coverage_count = [0] * 729
        for idx in solution:
            m = masks[idx]
            # Count bits in m
            # We need to increment coverage_count for each set bit in m
            # This is slow in pure Python, so let's use a different approach:
            # Instead, we'll recompute coverage by checking each word
            # Actually, let's just do the O(k*729/word_size) approach but smarter
            
            # For now, use the mask-based check but optimize the inner loop
            pass
        
        # Better reduction: try removing each codeword and check if coverage is still complete
        improved = True
        while improved:
            improved = False
            for i in range(len(solution)-1, -1, -1):
                # Check if solution without solution[i] still covers all
                # Build combined mask of all except i
                mask_sum = 0
                for j in range(len(solution)):
                    if j != i:
                        mask_sum |= masks[solution[j]]
                if mask_sum == all_mask:
                    solution.pop(i)
                    improved = True
                    break
        
        # Swap-based local search: try replacing one codeword with another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(solution)):
                current_idx = solution[i]
                # Try all other indices
                for new_idx in range(729):
                    if new_idx == current_idx:
                        continue
                    if new_idx in solution:
                        continue
                    # Check if replacing solution[i] with new_idx maintains coverage
                    # Build mask of all codewords except i, then add new_idx
                    mask_sum = 0
                    for j in range(len(solution)):
                        if j != i:
                            mask_sum |= masks[solution[j]]
                    mask_sum |= masks[new_idx]
                    if mask_sum == all_mask:
                        # Check if this reduces something or is neutral
                        # For now, accept if it's the same size (we can try to find smaller later)
                        # Actually, we want to find configurations that allow further reduction
                        # So we accept swaps that might enable future removals
                        # Simple heuristic: accept with some probability
                        if random.random() < 0.1:
                            solution[i] = new_idx
                            improved = True
                            break
                if improved:
                    break
        
        # Another reduction pass after swaps
        improved = True
        while improved:
            improved = False
            for i in range(len(solution)-1, -1, -1):
                mask_sum = 0
                for j in range(len(solution)):
                    if j != i:
                        mask_sum |= masks[solution[j]]
                if mask_sum == all_mask:
                    solution.pop(i)
                    improved = True
                    break
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
            if best_size <= 73:
                # Keep searching for better
                pass
    
    if best_solution is None:
        # Fallback: return all words
        return [idx_to_word[i][:] for i in range(729)]
        
    return [idx_to_word[i][:] for i in best_solution]
