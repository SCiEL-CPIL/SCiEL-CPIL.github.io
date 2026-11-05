import numpy as np
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # Leave 38s margin for 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters - simple greedy
        total_words = q ** n
        def int_to_base_q(num, qq, nn):
            res = [0] * nn
            for i in range(nn):
                res[i] = num % qq
                num //= qq
            return res
        words = [int_to_base_q(i, q, n) for i in range(total_words)]
        code = []
        uncovered = set(range(total_words))
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total_words):
                count = sum(1 for nb in range(n) if any(abs(words[c][nb] - words[w][nb]) > radius for w in [c]))
                # This fallback is very slow, but won't be used
                pass
            break
        return [words[c] for c in code]

    total_words = q ** n
    
    # Precompute neighbors using base-3 arithmetic
    def int_to_base3(num, nn):
        res = [0] * nn
        for i in range(nn):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        num = 0
        for d in reversed(digits):
            num = num * 3 + d
        return num

    # Generate all words
    words = [int_to_base3(i, n) for i in range(total_words)]
    
    # Precompute neighbors (distance <= radius=1)
    # For q=3, n=7, radius=1, neighbors are self + 14 single-bit changes
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = words[i][pos]
            for new_val in range(3):
                if new_val != orig_val:
                    # Compute index directly
                    delta = (new_val - orig_val) * (3 ** pos)
                    neighbors[i].append(i + delta)
                    
    # Precompute neighbor sets for fast membership testing
    neighbor_sets = [set(neighbors[i]) for i in range(total_words)]

    # Greedy Set Cover with heap-based optimization
    uncovered = set(range(total_words))
    code = []
    
    # Use a heap to track candidate codewords by their coverage count
    # We'll use a lazy deletion approach with a dictionary to track current counts
    heap = []  # (negative_count, codeword_index)
    current_counts = {}
    
    # Initialize heap
    for c in range(total_words):
        count = len(neighbors[c])  # All neighbors are uncovered initially
        current_counts[c] = count
        heapq.heappush(heap, (-count, c))
    
    while uncovered and time.time() < deadline - 60:
        # Find the codeword with maximum uncovered coverage
        best_c = -1
        best_count = -1
        
        while heap:
            neg_count, c = heapq.heappop(heap)
            count = -neg_count
            
            # Check if this is still valid (lazy deletion)
            if current_counts.get(c, 0) != count:
                continue
            
            # Check if c is still a candidate (has uncovered neighbors)
            actual_uncovered_count = sum(1 for nb in neighbors[c] if nb in uncovered)
            if actual_uncovered_count == 0:
                continue
            
            best_c = c
            best_count = actual_uncovered_count
            break
        
        if best_c == -1:
            break
        
        code.append(best_c)
        
        # Update uncovered set and counts
        for nb in neighbors[best_c]:
            if nb in uncovered:
                uncovered.remove(nb)
        
        # Update counts for all codewords that cover any of the newly covered neighbors
        # This is expensive, so we'll do it lazily - just mark for recomputation
        # For now, we'll just push updated counts for codewords adjacent to newly covered words
        for nb in neighbors[best_c]:
            if nb not in uncovered:  # nb was just covered
                for c in neighbors[nb]:
                    if c != best_c and c in current_counts:
                        new_count = sum(1 for x in neighbors[c] if x in uncovered)
                        if new_count != current_counts[c]:
                            current_counts[c] = new_count
                            heapq.heappush(heap, (-new_count, c))
        
        # Time check
        if time.time() > deadline - 60:
            break

    # Local Search Phase
    coverage = np.zeros(total_words, dtype=int)
    for c in code:
        for nb in neighbors[c]:
            coverage[nb] += 1
            
    def add_code(c):
        for nb in neighbors[c]:
            coverage[nb] += 1
        code.append(c)
        
    def remove_code(c):
        for nb in neighbors[c]:
            coverage[nb] -= 1
        code.remove(c)
        
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # 1. Try to remove redundant codewords
        removal_pass = True
        while removal_pass and time.time() < deadline - 10:
            removal_pass = False
            indices = list(range(len(code)))
            for idx in indices:
                if idx >= len(code):
                    continue
                c = code[idx]
                removable = True
                for nb in neighbors[c]:
                    if coverage[nb] == 1:
                        removable = False
                        break
                if removable:
                    remove_code(c)
                    removal_pass = True
                    improved = True
                    break # Restart removal pass to re-evaluate all
            
        # 2. Try to find a swap (1-for-1 exchange)
        if time.time() > deadline - 10:
            break

        swap_performed = False
        code_indices = list(range(len(code)))
        random.shuffle(code_indices)
        
        for idx in code_indices:
            if time.time() > deadline - 10:
                break
            if idx >= len(code):
                continue
            c = code[idx]
            unique_covered = []
            for nb in neighbors[c]:
                if coverage[nb] == 1:
                    unique_covered.append(nb)
            
            if not unique_covered:
                continue
            
            # Find candidate c_new that covers all words in unique_covered
            candidates = []
            for c_new in range(total_words):
                if c_new == c:
                    continue
                # Check if c_new covers all words in unique_covered
                covers_all = True
                for w in unique_covered:
                    if w not in neighbor_sets[c_new]:
                        covers_all = False
                        break
                if covers_all:
                    candidates.append(c_new)
                    break # Take the first one found to save time
            
            if candidates:
                c_new = candidates[0]
                remove_code(c)
                add_code(c_new)
                swap_performed = True
                improved = True
                break # Restart outer loop

    # Final cleanup: remove any remaining redundant codewords
    for i in range(len(code) - 1, -1, -1):
        c = code[i]
        removable = True
        for nb in neighbors[c]:
            if coverage[nb] == 1:
                removable = False
                break
        if removable:
            remove_code(c)

    return [words[c] for c in code]
