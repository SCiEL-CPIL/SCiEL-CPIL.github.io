import time
import random
import sys
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin from 120s limit
    
    total_words = q ** n
    
    # Precompute balls (neighborhoods) for each word
    # For radius=1, each ball has 1 + n*(q-1) words
    ball_size = 1 + n * (q - 1)
    
    # Precompute ball masks using bitsets
    # For 2187 words, a bitset is 2187 bits ~ 274 bytes, manageable
    # But Python integers as bitsets are efficient for bit operations
    
    # Convert word to index
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute all balls
    balls = []
    ball_masks = []
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
    
    all_mask = (1 << total_words) - 1
    
    def greedy_cover_lazy(seed):
        """
        Lazy greedy set cover: use a priority queue where entries are (gain, codeword).
        When popping, recompute gain; if it changed, skip.
        """
        random.seed(seed)
        covered_mask = 0
        code = []
        
        # Initialize heap with all codewords and their initial gains
        heap = []
        for c in range(total_words):
            gain = ball_masks[c].bit_count()
            heap.append((-gain, c))
        heapq.heapify(heap)
        
        # Track which codewords are in the heap (to avoid duplicates)
        in_heap = [True] * total_words
        
        while covered_mask != all_mask:
            if time.time() > deadline:
                return None
            
            # Pop until we find a valid entry
            valid_gain = -1
            valid_c = -1
            while heap:
                neg_gain, c = heapq.heappop(heap)
                in_heap[c] = False
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                if gain == -neg_gain:
                    valid_gain = gain
                    valid_c = c
                    break
            
            if valid_c == -1 or valid_gain <= 0:
                # Rebuild heap
                heap = []
                for c in range(total_words):
                    gain = (ball_masks[c] & ~covered_mask).bit_count()
                    if gain > 0:
                        heap.append((-gain, c))
                        in_heap[c] = True
                    else:
                        in_heap[c] = False
                heapq.heapify(heap)
                
                if not heap:
                    break
                
                # Pop again
                while heap:
                    neg_gain, c = heapq.heappop(heap)
                    in_heap[c] = False
                    gain = (ball_masks[c] & ~covered_mask).bit_count()
                    if gain == -neg_gain:
                        valid_gain = gain
                        valid_c = c
                        break
                
                if valid_c == -1 or valid_gain <= 0:
                    break
            
            # Select from top candidates with same gain for randomization
            # Collect all with same gain
            candidates = [valid_c]
            threshold = valid_gain
            while heap:
                neg_gain, c = heapq.heappop(heap)
                in_heap[c] = False
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                if gain == threshold:
                    candidates.append(c)
                elif gain < threshold:
                    # Put it back
                    if gain > 0:
                        heapq.heappush(heap, (-gain, c))
                        in_heap[c] = True
                    break
                else:
                    # Shouldn't happen since we popped in order
                    heapq.heappush(heap, (-gain, c))
                    in_heap[c] = True
            
            chosen = random.choice(candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            # Update gains for all remaining codewords in heap
            # This is expensive; instead, we'll rely on lazy re-evaluation
            # But we need to push updated entries
            # Actually, the lazy approach: we don't update, we just re-evaluate on pop
            # But we need to make sure we don't miss improvements
            # The standard lazy greedy: when a codeword's gain changes, we don't update the heap
            # We just recompute when popping. But we need to ensure we eventually pick the best.
            # This works because any codeword whose gain decreased will be skipped when popped,
            # and we'll continue popping until we find one whose gain is still valid.
            # But we might miss codewords whose gain increased (which can't happen in set cover).
            # In set cover, gains only decrease as we add more codewords.
            # So the lazy approach is correct: we pop, recompute, and if the gain is less than
            # what's in the heap, we discard and continue.
            
            # However, we also need to push back codewords that we popped but didn't select
            # from the candidates collection above
            for c in candidates:
                if c != chosen:
                    gain = (ball_masks[c] & ~covered_mask).bit_count()
                    if gain > 0:
                        heapq.heappush(heap, (-gain, c))
                        in_heap[c] = True
                    else:
                        in_heap[c] = False
        
        return code
    
    def reduce_cover(code):
        """Remove redundant codewords."""
        if not code:
            return code
        
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code
    
    def local_search(code):
        """
        Local search: try to remove codewords, and if that fails, try swaps.
        """
        if not code:
            return code
        
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove each codeword
            for i in range(len(code)):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: for each codeword, try to replace it with another that covers its unique words
            for i in range(len(code)):
                c = code[i]
                # Find words only covered by c
                unique_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    continue  # Should have been caught by removal
                
                # Find a replacement that covers all unique_words
                # And ideally adds some new coverage or allows further reduction
                best_swap = -1
                best_score = -1
                
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    
                    # Check if c_prime covers all unique_words
                    covers_all = True
                    for w in unique_words:
                        if not (ball_masks[c_prime] >> w) & 1:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Score: how many words does c_prime cover that are currently
                    # covered by multiple codewords (i.e., would become single-covered)?
                    # This is a heuristic to encourage further reductions
                    score = 0
                    for w in balls[c_prime]:
                        if cover_count[w] >= 2:
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_swap = c_prime
                    
                    if time.time() > deadline:
                        break
                
                if best_swap != -1:
                    # Perform swap
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code[i] = best_swap
                    
                    for w in balls[best_swap]:
                        cover_count[w] += 1
                    in_code[best_swap] = True
                    
                    # Check if this swap enabled a reduction
                    reduced = False
                    for j in range(len(code)):
                        cj = code[j]
                        redundant = True
                        for w in balls[cj]:
                            if cover_count[w] == 1:
                                redundant = False
                                break
                        if redundant:
                            for w in balls[cj]:
                                cover_count[w] -= 1
                            in_code[cj] = False
                            code.pop(j)
                            reduced = True
                            break
                    
                    if reduced:
                        improved = True
                        break
                    else:
                        # Revert swap
                        for w in balls[best_swap]:
                            cover_count[w] -= 1
                        in_code[best_swap] = False
                        code[i] = c
                        for w in balls[c]:
                            cover_count[w] += 1
                        in_code[c] = True
        
        return code
    
    # Main loop
    best_solution = None
    best_size = float('inf')
    
    # First run with seed 0
    initial_code = greedy_cover_lazy(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover_lazy(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        seed += 1
    
    if best_solution is None:
        # Fallback: direct sum construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
    
    return result
