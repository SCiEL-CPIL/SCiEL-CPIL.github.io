import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Simulated Annealing.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # 1. Pre-compute all possible k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subsets = {}
    for b in all_blocks:
        subs = tuple(itertools.combinations(b, t))
        block_to_subsets[b] = subs
        
    # Map from subset to list of blocks (for initialization)
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_blocks = {s: [] for s in all_subsets}
    for b, subs in block_to_subsets.items():
        for s in subs:
            subset_to_blocks[s].append(b)

    # 2. Initialization: Random Greedy
    # We want a valid covering.
    uncovered = set(all_subsets)
    blocks = []
    used_blocks = set()
    
    while uncovered:
        # Pick a random uncovered subset
        s = random.choice(list(uncovered))
        # Pick a random block covering s
        candidates = subset_to_blocks[s]
        b = random.choice(candidates)
        
        if b in used_blocks:
            continue
            
        blocks.append(b)
        used_blocks.add(b)
        
        # Update uncovered
        for sub in block_to_subsets[b]:
            uncovered.discard(sub)
            
    # Initial state is valid
    # coverage_counts
    coverage_counts = {}
    for b in blocks:
        for sub in block_to_subsets[b]:
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
    current_blocks = blocks
    current_E = len(current_blocks) # uncovered is 0
    
    best_blocks = list(current_blocks)
    best_E = current_E
    best_valid_E = current_E # Track best valid state
    
    # SA Parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of all blocks for random addition
    # We might want to add blocks that are "good" i.e. cover many uncovered?
    # No, random is better for SA.
    
    while time.time() - start_time < time_limit and T > min_T:
        
        # Generate a move
        move_type = random.choice(['remove', 'add', 'swap'])
        
        delta_E = 0
        accepted = False
        new_blocks = None
        new_coverage = None
        new_uncovered_count = 0
        
        if move_type == 'remove' and len(current_blocks) > 0:
            idx = random.randint(0, len(current_blocks) - 1)
            b = current_blocks[idx]
            
            # Check if removing b increases uncovered count
            # We need to know how many subsets in b are covered ONLY by b
            new_uncovered_count = 0
            for sub in block_to_subsets[b]:
                if coverage_counts[sub] == 1:
                    new_uncovered_count += 1
            
            # delta_E = -1 (block) + penalty * new_uncovered_count
            penalty = 10
            delta_E = -1 + penalty * new_uncovered_count
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                # Update state
                del current_blocks[idx]
                for sub in block_to_subsets[b]:
                    coverage_counts[sub] -= 1
                current_E += delta_E
                accepted = True
                
        elif move_type == 'add':
            # Pick a random block
            b = random.choice(all_blocks)
            
            # Check if b is already in current_blocks
            # This check is O(N). N is small (~100).
            if b in current_blocks:
                continue
                
            # Check change in uncovered
            new_uncovered_count = 0
            for sub in block_to_subsets[b]:
                if coverage_counts.get(sub, 0) == 0:
                    new_uncovered_count += 1
            
            penalty = 10
            delta_E = 1 - penalty * new_uncovered_count
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                current_blocks.append(b)
                for sub in block_to_subsets[b]:
                    coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                current_E += delta_E
                accepted = True
                
        elif move_type == 'swap' and len(current_blocks) > 0:
            idx = random.randint(0, len(current_blocks) - 1)
            b = current_blocks[idx]
            
            # Pick a point to swap out
            x_idx = random.randint(0, k - 1)
            x = b[x_idx]
            
            # Pick a point to swap in
            candidates = [y for y in range(v) if y not in b]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            new_b_list = [p for p in b if p != x]
            new_b_list.append(y)
            new_b = tuple(sorted(new_b_list))
            
            # If new_b is same as b, skip
            if new_b == b:
                continue
            
            # If new_b is already in blocks (and not the one we are swapping), we might have duplicates?
            # The problem requires distinct blocks.
            # If new_b is in current_blocks (at a different index), we can't just swap, we'd have a duplicate.
            # We should check if new_b is in current_blocks.
            # If it is, this move is invalid (or we should treat it as a remove+add? No, just skip).
            if new_b in current_blocks:
                continue
            
            # Calculate delta_E
            # Remove b, Add new_b
            # Subsets in b:
            subs_b = block_to_subsets[b]
            subs_new_b = block_to_subsets[new_b]
            
            # We need to count how many subsets become uncovered/covered
            # This is tricky to do in O(1) without updating.
            # Let's just simulate the change in uncovered count.
            
            # For each sub in subs_b:
            #   if coverage_counts[sub] == 1, it becomes uncovered (unless it's also in subs_new_b and count stays same? No, we remove b first)
            #   Actually, let's just calculate the new coverage counts for the affected subsets.
            
            # Affected subsets are union(subs_b, subs_new_b)
            
            # Simpler:
            # 1. Decrement counts for subs_b
            # 2. Increment counts for subs_new_b
            # 3. Count how many went from 0 to >0 (covered) and >0 to 0 (uncovered)
            
            # Let's do it efficiently.
            # We don't want to modify the actual coverage_counts yet.
            
            delta_uncovered = 0
            
            # Subsets only in b (or count drops to 0)
            for sub in subs_b:
                c = coverage_counts.get(sub, 0)
                if c == 1:
                    # If sub is NOT in subs_new_b, it becomes uncovered
                    # If sub IS in subs_new_b, it stays covered (count 1 -> 1? No, count 1 -> 0 -> 1? No.)
                    # If sub is in subs_new_b:
                    #   Old count: 1 (from b).
                    #   New count: 1 (from new_b).
                    #   So it remains covered.
                    # If sub is NOT in subs_new_b:
                    #   Old count: 1.
                    #   New count: 0.
                    #   Becomes uncovered.
                    if sub not in subs_new_b:
                        delta_uncovered += 1
                # If c > 1, it remains covered regardless (unless c=1 and not in new_b, handled above)
                # Wait, if c=2 and sub not in new_b, new count=1, still covered.
                
            # Subsets only in new_b (or count rises from 0)
            for sub in subs_new_b:
                c = coverage_counts.get(sub, 0)
                if c == 0:
                    # If sub is NOT in subs_b, it becomes covered
                    # If sub IS in subs_b, it was already covered (count >= 1)
                    #   If c=0, it wasn't in subs_b (since if it was in subs_b, count would be >=1)
                    #   So if c=0, it is NOT in subs_b.
                    #   So it becomes covered.
                    delta_uncovered -= 1
                # If c > 0, it remains covered.
                
            # Note: Subsets in both subs_b and subs_new_b:
            #   Old count: c >= 1.
            #   New count: c (remove 1, add 1).
            #   No change in covered/uncovered status.
            
            penalty = 10
            delta_E = 0 + penalty * delta_uncovered # Block count unchanged
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                # Update state
                for sub in subs_b:
                    coverage_counts[sub] -= 1
                for sub in subs_new_b:
                    coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                current_blocks[idx] = new_b
                current_E += delta_E
                accepted = True

        if accepted:
            # Check if current state is valid and better than best valid
            # We need to know current uncovered count.
            # We can track it, but we didn't in the loop above explicitly for all moves.
            # Let's calculate it if we accepted?
            # Or just check if current_E corresponds to 0 uncovered?
            # current_E = num_blocks + 10 * uncovered_count.
            # If we know num_blocks, we can infer uncovered_count = (current_E - num_blocks) / 10.
            # This is fragile if penalty is not integer or if we make errors.
            
            # Better: Maintain `current_uncovered_count` explicitly.
            # Let's refactor to track `current_uncovered_count`.
            
            pass  # Placeholder to satisfy indentation requirement

        T *= cooling

    # The above loop is a bit messy with tracking uncovered count.
    # Let's restart the implementation with explicit tracking of `current_uncovered_count`.
    
    # --- RESTART IMPLEMENTATION WITH CLEANER STATE TRACKING ---
    
    # Re-initialize state
    uncovered_set = set(all_subsets)
    blocks = []
    used_blocks = set()
    coverage_counts = {}
    
    while uncovered_set:
        s = random.choice(list(uncovered_set))
        candidates = subset_to_blocks[s]
        b = random.choice(candidates)
        if b in used_blocks:
            continue
        blocks.append(b)
        used_blocks.add(b)
        for sub in block_to_subsets[b]:
            if sub not in coverage_counts:
                coverage_counts[sub] = 0
            coverage_counts[sub] += 1
            if coverage_counts[sub] == 1:
                uncovered_set.discard(sub)
                
    current_blocks = blocks
    current_uncovered = 0
    current_num_blocks = len(current_blocks)
    current_E = current_num_blocks + 10 * current_uncovered
    
    best_blocks = list(current_blocks)
    best_E = current_E
    best_uncovered = 0
    
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    penalty = 10
    
    while time.time() - start_time < time_limit and T > min_T:
        
        move_type = random.choice(['remove', 'add', 'swap'])
        
        if move_type == 'remove' and len(current_blocks) > 0:
            idx = random.randint(0, len(current_blocks) - 1)
            b = current_blocks[idx]
            
            new_uncovered = current_uncovered
            for sub in block_to_subsets[b]:
                if coverage_counts[sub] == 1:
                    new_uncovered += 1
            
            delta_E = -1 + penalty * (new_uncovered - current_uncovered)
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                del current_blocks[idx]
                for sub in block_to_subsets[b]:
                    coverage_counts[sub] -= 1
                current_num_blocks -= 1
                current_uncovered = new_uncovered
                current_E += delta_E
                
                if current_uncovered == 0 and current_E < best_E:
                    best_E = current_E
                    best_blocks = list(current_blocks)
                    
        elif move_type == 'add':
            b = random.choice(all_blocks)
            if b in current_blocks:
                continue
                
            new_uncovered = current_uncovered
            for sub in block_to_subsets[b]:
                if coverage_counts.get(sub, 0) == 0:
                    new_uncovered -= 1
            
            delta_E = 1 + penalty * (new_uncovered - current_uncovered)
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                current_blocks.append(b)
                for sub in block_to_subsets[b]:
                    coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                current_num_blocks += 1
                current_uncovered = new_uncovered
                current_E += delta_E
                
                if current_uncovered == 0 and current_E < best_E:
                    best_E = current_E
                    best_blocks = list(current_blocks)
                    
        elif move_type == 'swap' and len(current_blocks) > 0:
            idx = random.randint(0, len(current_blocks) - 1)
            b = current_blocks[idx]
            
            x_idx = random.randint(0, k - 1)
            x = b[x_idx]
            
            candidates = [y for y in range(v) if y not in b]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            new_b_list = [p for p in b if p != x]
            new_b_list.append(y)
            new_b = tuple(sorted(new_b_list))
            
            if new_b == b or new_b in current_blocks:
                continue
            
            subs_b = block_to_subsets[b]
            subs_new_b = block_to_subsets[new_b]
            
            new_uncovered = current_uncovered
            
            # Check subsets in b
            for sub in subs_b:
                c = coverage_counts.get(sub, 0)
                if c == 1:
                    # If sub is not in new_b, it becomes uncovered
                    if sub not in subs_new_b:
                        new_uncovered += 1
                # If c > 1, no change in uncovered status (unless c=1, handled)
                # If c=0, impossible since b covers it
                
            # Check subsets in new_b
            for sub in subs_new_b:
                c = coverage_counts.get(sub, 0)
                if c == 0:
                    # If sub is not in b, it becomes covered
                    # (If it was in b, c would be >= 1)
                    if sub not in subs_b:
                        new_uncovered -= 1
                        
            delta_E = penalty * (new_uncovered - current_uncovered)
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                for sub in subs_b:
                    coverage_counts[sub] -= 1
                for sub in subs_new_b:
                    coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                current_blocks[idx] = new_b
                current_uncovered = new_uncovered
                current_E += delta_E
                
                if current_uncovered == 0 and current_E < best_E:
                    best_E = current_E
                    best_blocks = list(current_blocks)

        T *= cooling

    return [list(b) for b in best_blocks]
