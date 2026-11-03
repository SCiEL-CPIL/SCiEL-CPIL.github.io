import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 150.0
    best_blocks = None
    best_count = float('inf')
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    def greedy_construct(randomize=True):
        uncovered = set(range(n_subsets))
        blocks = []
        while uncovered:
            if randomize and len(uncovered) > 1:
                idx = random.choice(list(uncovered))
            else:
                idx = min(uncovered)
            
            block = list(all_t_subsets[idx])
            while len(block) < k:
                best_x = -1
                best_gain = -1
                best_candidates = []
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        sub = tuple(sorted(rest + (x,)))
                        i = subset_to_idx.get(sub)
                        if i is not None and i in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if randomize and len(best_candidates) > 1:
                    block.append(random.choice(best_candidates))
                else:
                    block.append(best_x)
            
            block.sort()
            blocks.append(block)
            for s in itertools.combinations(block, t):
                i = subset_to_idx[tuple(s)]
                uncovered.discard(i)
        
        return blocks

    def init_counts(blocks):
        counts = [0] * n_subsets
        for b in blocks:
            for s in itertools.combinations(b, t):
                counts[subset_to_idx[tuple(s)]] += 1
        return counts

    def is_redundant(b, counts):
        for s in itertools.combinations(b, t):
            if counts[subset_to_idx[tuple(s)]] == 1:
                return False
        return True

    def local_search(blocks, counts):
        while time.time() < deadline - 5.0:
            # 1. Remove redundant
            removed = False
            for i in range(len(blocks)):
                if is_redundant(blocks[i], counts):
                    b = blocks[i]
                    for s in itertools.combinations(b, t):
                        counts[subset_to_idx[tuple(s)]] -= 1
                    blocks.pop(i)
                    removed = True
                    break
            
            if removed:
                continue
            
            # 2. Swaps
            if not blocks:
                break
                
            i = random.randint(0, len(blocks)-1)
            b = blocks[i]
            
            # Pruned x selection
            valid_x_candidates = []
            for x in b:
                can_remove = True
                for s in itertools.combinations(b, t):
                    if x in s:
                        if counts[subset_to_idx[tuple(s)]] == 1:
                            can_remove = False
                            break
                if can_remove:
                    valid_x_candidates.append(x)
            
            if not valid_x_candidates:
                continue
                
            x = random.choice(valid_x_candidates)
            candidates_y = [p for p in range(v) if p not in b]
            if not candidates_y:
                continue
            y = random.choice(candidates_y)
            
            # Check y validity (mostly about not breaking coverage, which is handled by x check? 
            # No, adding y creates new subsets. We don't care if they become covered, 
            # we only care if removing x breaks coverage. 
            # Wait, if we remove x, we lose coverage for subsets containing x. 
            # We checked that those subsets have count > 1. 
            # Adding y creates new subsets containing y. 
            # These subsets get count + 1. 
            # This never breaks coverage. 
            # So the only validity check is on the removal of x.
            # So the swap is ALWAYS valid if x is "safe" to remove.
            
            # Apply swap
            for s in itertools.combinations(b, t):
                if x in s:
                    counts[subset_to_idx[tuple(s)]] -= 1
            
            new_b_set = set(b)
            new_b_set.remove(x)
            new_b_set.add(y)
            new_b = list(new_b_set)
            
            for s in itertools.combinations(new_b, t):
                if y in s:
                    counts[subset_to_idx[tuple(s)]] += 1
            
            blocks[i] = sorted(new_b)
            
            # Check if new block is redundant immediately
            if is_redundant(blocks[i], counts):
                b_rem = blocks[i]
                for s in itertools.combinations(b_rem, t):
                    counts[subset_to_idx[tuple(s)]] -= 1
                blocks.pop(i)
                # Continue to next iteration to scan for more removals

        return blocks, counts

    while time.time() < deadline - 10.0:
        blocks = greedy_construct(randomize=True)
        counts = init_counts(blocks)
        blocks, counts = local_search(blocks, counts)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
        # Optional: if we find 151, we can try to run longer on that instance?
        # But restarts are cheap.
        
    return best_blocks if best_blocks else []
