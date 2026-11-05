import time
import itertools
import random
import heapq

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations
    all_perms = list(itertools.permutations(range(1, n+1)))
    perm_set = set(all_perms)
    counts = {p: 0 for p in perm_set}
    
    # Initial count
    for i in range(len(current_word) - n + 1):
        p = tuple(current_word[i:i+n])
        if p in counts:
            counts[p] += 1

    best_word = current_word[:]
    best_len = len(current_word)
    
    start_time = time.time()
    deadline = 115.0
    
    # Greedy Set-Cover Heuristic for Deletion
    # We prioritize deleting elements that are part of "constrained" windows (count == 1)
    # to force resolution of these constraints early, or rather, we prioritize deleting
    # elements that are NOT critical (count > 1) but maximize the "redundancy" removed.
    # However, the directive suggests: "maximizing the number of *new* permutations ... completed ... prioritizing symbols that complete permutations which have the fewest other possible completions".
    # This is slightly different. Let's stick to a robust deletion strategy that uses counts.
    # Strategy: Find all deletable indices. Sort them by the minimum count of the affected permutations.
    # If a deletion affects a permutation with count 2, it reduces it to 1 (riskier).
    # If a deletion affects permutations all with count > 2, it's safer.
    # We want to delete "safe" deletions first to clear up space, or "risky" ones to break symmetry?
    # Usually, deleting "safe" ones (high counts) is better for finding a stable minimal solution.
    
    while time.time() < deadline:
        # Collect candidates for deletion
        candidates = []
        
        # Optimization: Only check a random subset of indices if the word is very long?
        # For n=7, length ~5900. Checking all is O(N * n) ~ 40k ops. Very fast.
        for i in range(len(current_word)):
            if time.time() > deadline:
                break
            
            start_j = max(0, i - n + 1)
            end_j = min(i, len(current_word) - n)
            
            min_count = float('inf')
            can_delete = True
            affected = []
            
            for j in range(start_j, end_j + 1):
                p = tuple(current_word[j:j+n])
                c = counts.get(p, 0)
                if c <= 1:
                    can_delete = False
                    break
                affected.append(p)
                if c < min_count:
                    min_count = c
            
            if can_delete:
                # Score: Lower min_count means "tighter" constraints.
                # We might want to delete from "looser" areas first (higher min_count) to avoid breaking fragile parts?
                # Or delete from "tighter" areas to see if they break?
                # Let's try prioritizing deletions that affect permutations with HIGHER counts (safer).
                # This is a heuristic.
                candidates.append((min_count, i, affected))
        
        if not candidates:
            # No deletable indices. Stuck.
            # Try a random restart or different strategy?
            # For now, break to avoid infinite loop.
            break
        
        # Sort candidates by min_count descending (safer first)
        # Adding a small random tie-breaker to avoid getting stuck in local minima
        candidates.sort(key=lambda x: (-x[0], random.random()))
        
        # Process the best candidate
        # Note: We only process one deletion per outer loop iteration?
        # That would be slow (5900 iterations * 40k checks = 200M ops). Too slow.
        # We should process multiple deletions in one pass.
        
        # Let's re-evaluate.
        # The previous code processed ALL deletable indices in one pass (shuffled).
        # That was fast.
        # Let's try: Collect all deletable indices. Sort them. Process them in order.
        # But as we delete, indices shift.
        # If we process from end to start, indices of previous elements don't shift.
        
        # Revised Strategy:
        # 1. Identify all currently deletable indices.
        # 2. Assign a priority score to each.
        # 3. Process deletions in reverse index order (to maintain index validity for earlier indices) 
        #    BUT we want to process by priority, not index.
        #    This is tricky.
        
        # Alternative:
        # Use a priority queue.
        # Push all deletable indices with their scores.
        # Pop the best one. Delete it.
        # Update neighbors. Push neighbors if they become deletable.
        # This is like Dijkstra/A* on the deletion space.
        
        # Let's implement the Priority Queue approach.
        
        # Reset counts? No, counts are maintained incrementally.
        # But we need to initialize the PQ with current state.
        
        # Since we are in a loop, we can re-initialize the PQ each time?
        # That's O(N log N) + O(N * n) for initial check.
        # 5900 * 7 = 41k ops. Log is ~13. 500k ops.
        # If we do this 100 times, 50M ops. Might be tight in 120s.
        # But we only do it if we are stuck?
        
        # Let's try a simpler improvement on the existing "Shuffled Pass":
        # Instead of random shuffle, sort by "safety" (min_count descending).
        # But we have the index shift problem.
        
        # Solution:
        # Iterate from END to START.
        # Check if index i is deletable.
        # If yes, we can decide to delete or not.
        # But we want to prioritize "safe" deletions.
        # If we iterate end-to-start, we can't easily sort by safety across the whole array.
        
        # Let's stick to the Priority Queue approach but optimize it.
        # We will maintain a set of "active" candidates.
        
        # Actually, the current code's "Shuffled Pass" is quite good because it's simple and robust.
        # The main issue is it might get stuck in local minima.
        # Let's try to add a "restart" mechanism if no improvement is made in a full pass.
        
        pass_count = 0
        last_best_len = best_len
        
        while time.time() < deadline:
            pass_count += 1
            if pass_count > 50: # Limit passes to avoid infinite loop on stuck states
                break
            
            # Collect deletable indices
            deletable = []
            for i in range(len(current_word)):
                if time.time() > deadline:
                    break
                start_j = max(0, i - n + 1)
                end_j = min(i, len(current_word) - n)
                can_delete = True
                min_count = float('inf')
                for j in range(start_j, end_j + 1):
                    p = tuple(current_word[j:j+n])
                    c = counts.get(p, 0)
                    if c <= 1:
                        can_delete = False
                        break
                    if c < min_count:
                        min_count = c
                if can_delete:
                    deletable.append((min_count, i))
            
            if not deletable:
                break
            
            # Sort by min_count descending (safest first)
            deletable.sort(key=lambda x: -x[0])
            
            # Process deletions
            # To handle index shifting, we process in reverse index order?
            # No, we want to process by priority.
            # If we delete index i, all indices > i shift left by 1.
            # If we process indices in descending order, earlier indices are unaffected.
            # So: Sort by index DESCENDING.
            # But we want to prioritize safety.
            
            # Compromise:
            # Bucket sort by min_count?
            # Or just process in descending index order, but skip if not "safe enough"?
            
            # Let's try: Process in descending index order.
            # This allows us to delete safely without shifting issues for earlier indices.
            # We lose the ability to pick the "safest" deletion first, but we maintain consistency.
            
            deletable.sort(key=lambda x: x[1], reverse=True)
            
            deleted_any = False
            for min_count, i in deletable:
                if i >= len(current_word):
                    continue
                if time.time() > deadline:
                    break
                
                # Re-verify if still deletable (counts might have changed? No, we only delete, counts decrease)
                # Wait, if we delete an index j > i, does it affect i?
                # No, i is before j. Deleting j shifts indices > j. i is unaffected.
                # So the check is still valid.
                
                start_j = max(0, i - n + 1)
                end_j = min(i, len(current_word) - n)
                
                can_delete = True
                affected = []
                for j in range(start_j, end_j + 1):
                    p = tuple(current_word[j:j+n])
                    if counts.get(p, 0) <= 1:
                        can_delete = False
                        break
                    affected.append(p)
                
                if can_delete:
                    for p in affected:
                        counts[p] -= 1
                    current_word.pop(i)
                    deleted_any = True
                    
                    if len(current_word) < best_len:
                        best_len = len(current_word)
                        best_word = current_word[:]
            
            if not deleted_any:
                break
        
    return "".join(map(str, best_word))
