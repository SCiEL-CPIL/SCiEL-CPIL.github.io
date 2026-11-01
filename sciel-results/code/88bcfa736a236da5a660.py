import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Specific case for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as tuples
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute spheres: list of indices for each word within distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
            
        # Precompute reverse spheres for faster update?
        # Not strictly necessary for greedy, but good for local search.
        
        best_code = None
        best_size = total_words
        
        # Number of restarts depends on time
        # Each restart is fast.
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Greedy Construction
            covered = np.zeros(total_words, dtype=bool)
            uncovered_count = total_words
            code = []
            picked = set()
            
            # Gain array: number of uncovered words covered by each candidate
            gain = np.zeros(total_words, dtype=int)
            for i in range(total_words):
                gain[i] = len(spheres[i])
                
            # We will use a simple scan for the best candidate each time,
            # but with a randomized tie-breaking to explore different structures.
            # To speed up, we can maintain a list of candidates with high gain,
            # but 729 is small, so O(N^2) is 729^2 * 73 ~ 40M ops, which is fast in Python?
            # 40M ops in Python might take a few seconds.
            # Let's optimize the inner loop.
            
            # Actually, let's use the heap approach from the current program but with better tie-breaking.
            # Or just simple scan with NumPy?
            # NumPy vectorization for gain update is tricky because of the dependency.
            
            # Let's stick to the heap approach but add randomness to tie-breakers.
            import heapq
            
            heap = []
            # Initial gain is 13 for all.
            # Tie-breaker: random value to diversify, or symmetry score.
            # Let's use a random value for tie-breaking to simulate temperature.
            tie_breakers = np.random.rand(total_words)
            
            for i in range(total_words):
                # Heap entry: (-gain, tie_breaker, word_idx)
                # We want max gain, so -gain.
                # For tie-breaking, we want to pick one of the max gain words.
                # If we use random tie-breakers, we pick a random max-gain word.
                heap.append((-gain[i], tie_breakers[i], i))
            
            heapq.heapify(heap)
            
            while uncovered_count > 0:
                if time.time() > deadline:
                    break
                
                best_candidate = -1
                best_gain = 0
                best_tie = 0
                
                # Pop until we find a valid candidate
                while heap:
                    neg_gain, tb, cand_idx = heapq.heappop(heap)
                    
                    # Check if stale
                    if -neg_gain != gain[cand_idx]:
                        continue
                    
                    # Check if picked
                    if cand_idx in picked:
                        continue
                    
                    best_candidate = cand_idx
                    best_gain = -neg_gain
                    best_tie = tb
                    break
                
                if best_candidate == -1:
                    break
                
                # Select candidate
                picked.add(best_candidate)
                code.append(list(all_words[best_candidate]))
                
                # Update coverage and gains
                for idx in spheres[best_candidate]:
                    if not covered[idx]:
                        covered[idx] = True
                        uncovered_count -= 1
                        
                        # Update gain for all candidates that cover idx
                        # We need reverse_spheres for this
                        # Let's precompute reverse_spheres outside the restart loop
                        pass
            
            # We need reverse_spheres for the update step above.
            # Let's precompute it once.
            
            # The loop above is missing the gain update logic.
            # Let's re-implement the restart loop properly.
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = [c[:] for c in code]
                
                # If we found 73, we can stop early?
                # 73 is the known upper bound. If we hit 73, it's likely optimal.
                if best_size <= 73:
                    break

        # If best_code is None or we didn't finish, we need to return something valid.
        # The logic above was incomplete. Let's write a clean version.
        
        # Precompute reverse spheres
        reverse_spheres = [[] for _ in range(total_words)]
        for i in range(total_words):
            for j in spheres[i]:
                reverse_spheres[j].append(i)
        
        best_code = None
        best_size = total_words
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            covered = [False] * total_words
            uncovered_count = total_words
            code = []
            picked = set()
            
            gain = [0] * total_words
            for i in range(total_words):
                gain[i] = len(spheres[i])
                
            heap = []
            # Use random tie-breakers to explore different greedy paths
            tie_breakers = [random.random() for _ in range(total_words)]
            
            for i in range(total_words):
                heap.append((-gain[i], tie_breakers[i], i))
            
            heapq.heapify(heap)
            
            while uncovered_count > 0:
                if time.time() > deadline:
                    break
                
                best_candidate = -1
                best_gain = 0
                
                while heap:
                    neg_gain, tb, cand_idx = heapq.heappop(heap)
                    
                    if -neg_gain != gain[cand_idx]:
                        continue
                    
                    if cand_idx in picked:
                        continue
                    
                    best_candidate = cand_idx
                    best_gain = -neg_gain
                    break
                
                if best_candidate == -1:
                    break
                
                picked.add(best_candidate)
                code.append(list(all_words[best_candidate]))
                
                for idx in spheres[best_candidate]:
                    if not covered[idx]:
                        covered[idx] = True
                        uncovered_count -= 1
                        
                        for cand in reverse_spheres[idx]:
                            if cand not in picked:
                                gain[cand] -= 1
                                heapq.heappush(heap, (-gain[cand], tie_breakers[cand], cand))
            
            # Local Search: Try to reduce size
            # If size > 73, try to remove a word and see if we can cover the gaps with fewer words?
            # Or just try to remove a word that is "redundant" (its sphere is covered by others).
            # Check redundancy:
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                for i in range(len(code) - 1, -1, -1):
                    # Check if removing code[i] leaves all its sphere covered by others
                    # This check is expensive.
                    # Simpler: Try to remove code[i]. If it creates uncovered words, 
                    # try to add one word that covers them. If the net change is <= 0, keep it.
                    # Actually, just checking redundancy is a good first step.
                    
                    # Check if every word in spheres[code_idx] is covered by some other codeword
                    # This is O(sphere_size * code_size * sphere_size)
                    # 13 * 73 * 13 ~ 12000 ops per check. 73 checks ~ 1M ops. Fast.
                    
                    cw_idx = word_to_idx[tuple(code[i])]
                    is_redundant = True
                    for w in spheres[cw_idx]:
                        # Is w covered by any other codeword?
                        covered_by_other = False
                        for j in range(len(code)):
                            if i == j:
                                continue
                            other_cw_idx = word_to_idx[tuple(code[j])]
                            if w in spheres[other_cw_idx]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        code.pop(i)
                        improved = True
                        if time.time() > deadline:
                            break
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = [c[:] for c in code]
                
                if best_size <= 73:
                    break

        if best_code is not None:
            return best_code
        else:
            # Fallback: Return the last code generated (should be valid)
            return code if 'code' in locals() and code else [list(t) for t in itertools.product(range(q), repeat=n)]
            
    else:
        # Fallback: Simple greedy for other params
        # This part is rarely used for this specific task
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
            
        reverse_spheres = [[] for _ in range(len(all_words))]
        for i in range(len(all_words)):
            for j in spheres[i]:
                reverse_spheres[j].append(i)
                
        covered = [False] * len(all_words)
        uncovered_count = len(all_words)
        code = []
        picked = set()
        
        gain = [0] * len(all_words)
        for i in range(len(all_words)):
            gain[i] = len(spheres[i])
            
        heap = []
        for i in range(len(all_words)):
            heap.append((-gain[i], random.random(), i))
        heapq.heapify(heap)
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            best_candidate = -1
            while heap:
                neg_gain, tb, cand_idx = heapq.heappop(heap)
                if -neg_gain != gain[cand_idx]:
                    continue
                if cand_idx in picked:
                    continue
                best_candidate = cand_idx
                break
            if best_candidate == -1:
                break
            picked.add(best_candidate)
            code.append(list(all_words[best_candidate]))
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    for cand in reverse_spheres[idx]:
                        if cand not in picked:
                            gain[cand] -= 1
                            heapq.heappush(heap, (-gain[cand], random.random(), cand))
                            
        return code
