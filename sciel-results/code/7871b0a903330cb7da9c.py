import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    required = [1, 2, 5, 11]
    max_val = 284
    
    def get_sums(seq):
        sums = set()
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                sums.add(seq[i] + seq[j])
        return sums
    
    def can_add(seq, x, sums_set):
        for y in seq:
            s = x + y
            if s in sums_set:
                return False
        if 2 * x in sums_set:
            return False
        return True
    
    def greedy_build(required_set, pool_order):
        curr = list(required_set)
        curr_sums = get_sums(curr)
        curr_set = set(curr)
        for x in pool_order:
            if x in curr_set:
                continue
            if can_add(curr, x, curr_sums):
                curr.append(x)
                curr_set.add(x)
                for y in curr[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2 * x)
        return curr
    
    def find_conflicts(seq):
        """Find all sum conflicts: sums that appear more than once"""
        sum_map = defaultdict(list)
        n = len(seq)
        for i in range(n):
            for j in range(i, n):
                s = seq[i] + seq[j]
                sum_map[s].append((i, j))
        conflicts = []
        for s, pairs in sum_map.items():
            if len(pairs) > 1:
                conflicts.append((s, pairs))
        return conflicts
    
    def conflict_count(seq):
        """Count total number of conflicting pairs (excess over 1 per sum)"""
        sum_map = defaultdict(int)
        n = len(seq)
        for i in range(n):
            for j in range(i, n):
                sum_map[seq[i] + seq[j]] += 1
        total = 0
        for s, cnt in sum_map.items():
            if cnt > 1:
                total += cnt - 1
        return total
    
    # Phase 1: Establish a good baseline with multiple randomized greedy runs
    best_set = None
    best_size = 0
    
    required_set = set(required)
    all_candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    # Deterministic greedy first
    curr = greedy_build(required, list(range(12, max_val + 1)))
    if len(curr) > best_size:
        best_size = len(curr)
        best_set = curr[:]
    
    # Multiple randomized greedy runs
    for iteration in range(200):
        if time.time() > deadline:
            break
        random.shuffle(all_candidates)
        curr = greedy_build(required, all_candidates)
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
    
    # Phase 2: Conflict-based local search
    # Try to improve by removing conflicting elements and refilling
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        if not best_set:
            break
        
        # Start from current best
        curr = best_set[:]
        curr_set = set(curr)
        
        # Try removing 1-3 random non-required elements and refilling
        removable = [x for x in curr if x not in required_set]
        if not removable:
            break
        
        # Randomly remove 1-3 elements
        remove_count = random.randint(1, min(3, len(removable)))
        to_remove = random.sample(removable, remove_count)
        
        for x in to_remove:
            curr.remove(x)
            curr_set.remove(x)
        
        # Refill greedily with random order
        remaining_pool = [x for x in range(1, max_val + 1) if x not in curr_set]
        random.shuffle(remaining_pool)
        
        curr_sums = get_sums(curr)
        for x in remaining_pool:
            if can_add(curr, x, curr_sums):
                curr.append(x)
                curr_set.add(x)
                for y in curr[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2 * x)
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
    
    # Phase 3: More aggressive local search - try swapping elements
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        if not best_set:
            break
        
        curr = best_set[:]
        curr_set = set(curr)
        removable = [x for x in curr if x not in required_set]
        if not removable:
            break
        
        # Pick a random element to remove
        rem_val = random.choice(removable)
        curr.remove(rem_val)
        curr_set.remove(rem_val)
        
        # Try to add a different element
        candidates = [x for x in range(1, max_val + 1) if x not in curr_set]
        random.shuffle(candidates)
        
        curr_sums = get_sums(curr)
        added = False
        for x in candidates:
            if can_add(curr, x, curr_sums):
                curr.append(x)
                curr_set.add(x)
                for y in curr[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2 * x)
                added = True
                break
        
        if added and len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
    
    # Phase 4: Try to push toward size 19/20 with targeted search
    # If we're at size 18, try harder to find 19
    target_size = max(best_size + 1, 19)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        if best_size >= 20:
            break
        
        # Start from best, remove 2-4 elements, refill with bias toward filling gaps
        if not best_set:
            break
            
        curr = best_set[:]
        curr_set = set(curr)
        removable = [x for x in curr if x not in required_set]
        if not removable:
            break
        
        remove_count = random.randint(2, min(4, len(removable)))
        to_remove = random.sample(removable, remove_count)
        for x in to_remove:
            curr.remove(x)
            curr_set.remove(x)
        
        # Refill with random order
        remaining_pool = [x for x in range(1, max_val + 1) if x not in curr_set]
        random.shuffle(remaining_pool)
        
        curr_sums = get_sums(curr)
        for x in remaining_pool:
            if can_add(curr, x, curr_sums):
                curr.append(x)
                curr_set.add(x)
                for y in curr[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2 * x)
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
    
    # Phase 5: Final polish - try small perturbations
    for iteration in range(200):
        if time.time() > deadline:
            break
        if best_size >= 20:
            break
        
        if not best_set:
            break
            
        curr = best_set[:]
        curr_set = set(curr)
        removable = [x for x in curr if x not in required_set]
        if not removable:
            break
        
        # Remove 1 element, try to add 2 different ones
        rem_val = random.choice(removable)
        curr.remove(rem_val)
        curr_set.remove(rem_val)
        
        remaining_pool = [x for x in range(1, max_val + 1) if x not in curr_set]
        random.shuffle(remaining_pool)
        
        curr_sums = get_sums(curr)
        added_count = 0
        for x in remaining_pool:
            if added_count >= 2:
                break
            if can_add(curr, x, curr_sums):
                curr.append(x)
                curr_set.add(x)
                for y in curr[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2 * x)
                added_count += 1
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
    
    if best_set:
        return sorted(best_set)
    else:
        return sorted(required)
