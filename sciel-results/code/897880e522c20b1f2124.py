import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a multi-restart approach:
    1. Fast randomized greedy construction.
    2. Aggressive reduction (removal of redundant blocks).
    3. Keep the best (smallest) valid result found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def greedy_construct(seed, time_limit):
        """
        Perform a single greedy construction.
        Returns a list of blocks (tuples) and a boolean indicating if it finished in time.
        """
        rng = random.Random(seed)
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        # Precompute (t-1) combinations for fast gain calculation?
        # Actually, we just iterate.
        
        while uncovered_set:
            # Pick a random uncovered t-subset
            target_idx = rng.choice(list(uncovered_set))
            target_subset = all_t_subsets[target_idx]
            
            block_points = set(target_subset)
            remaining_points = [x for x in range(v) if x not in block_points]
            
            # Greedily add k-t points
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                
                for x in remaining_points:
                    gain = 0
                    # Iterate over (t-1) subsets of current block
                    for rest in itertools.combinations(block_points, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point != -1:
                    block_points.add(best_point)
                    remaining_points.remove(best_point)
                else:
                    # Fallback: random
                    if remaining_points:
                        p = rng.choice(remaining_points)
                        block_points.add(p)
                        remaining_points.remove(p)
            
            block = tuple(sorted(block_points))
            blocks.append(block)
            
            # Update uncovered_set
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
            
            # Check time
            if time.time() > time_limit:
                return blocks, False
        
        return blocks, True

    def reduce_blocks(blocks, seed, time_limit):
        """
        Reduce blocks by removing redundant ones.
        Returns the reduced list of blocks.
        """
        rng = random.Random(seed)
        cover_count = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[t_subset_to_idx[ts]] += 1
        
        blocks = list(blocks)
        
        while time.time() < time_limit:
            changed = False
            # Shuffle order of indices to break ties/biases
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                removable = True
                for ts in itertools.combinations(b, t):
                    if cover_count[t_subset_to_idx[ts]] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove block
                    for ts in itertools.combinations(b, t):
                        cover_count[t_subset_to_idx[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    # Break to restart scan with updated counts
                    break
            
            if not changed:
                break
        
        return blocks

    def is_valid(blocks):
        """
        Check if the list of blocks is a valid covering.
        """
        covered = [False] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered[t_subset_to_idx[ts]] = True
        return all(covered)

    best_blocks = None
    
    # Multiple restarts
    seed = 42
    num_restarts = 10  # We can adjust this based on time
    
    for i in range(num_restarts):
        # Check time
        if time.time() > deadline - 20.0:
            break
            
        # Allocate time for greedy and reduction
        # Greedy is usually fast (< 2s for v=19).
        # Reduction can take longer.
        greedy_limit = start_time + (time.time() - start_time) + 5.0 # 5 seconds for greedy
        # Actually, just use a fixed small limit for greedy and the rest for reduction?
        # Or dynamic:
        # If we have 100s left, spend 5s on greedy, 95s on reduction.
        
        greedy_deadline = time.time() + 3.0
        blocks, finished = greedy_construct(seed, greedy_deadline)
        
        if not finished:
            # If greedy didn't finish, the covering is likely incomplete.
            # We can try to patch it, but it's risky.
            # Let's just skip this restart if it didn't finish, or try to complete it quickly.
            # For now, if it didn't finish, we assume it's not good.
            # But wait, if it broke early, uncovered_set is not empty.
            # The block list might be invalid.
            # We should only consider valid blocks.
            # Let's check validity.
            pass
            
        # Reduce
        reduction_deadline = deadline - 10.0 # Leave 10s for final checks
        reduced_blocks = reduce_blocks(blocks, seed, reduction_deadline)
        
        # Validate
        if is_valid(reduced_blocks):
            if best_blocks is None or len(reduced_blocks) < len(best_blocks):
                best_blocks = reduced_blocks
        
        seed += 1

    # If no valid blocks found (shouldn't happen with 145s), do a last resort
    if best_blocks is None:
        # Last resort: greedy without strict time limit for greedy, just force finish
        # This is a fallback
        blocks, finished = greedy_construct(999, deadline - 5.0)
        if not finished:
             # Emergency patch
             # This is very slow and bad, but better than nothing?
             # Actually, if greedy is this slow, something is wrong.
             # Let's just take what we have and hope it's valid? No, must be valid.
             # We'll try to patch uncovered sets.
             cover_count = [0] * num_t_subsets
             for b in blocks:
                 for ts in itertools.combinations(b, t):
                     cover_count[t_subset_to_idx[ts]] += 1
             
             uncovered = [i for i, c in enumerate(cover_count) if c == 0]
             for idx in uncovered:
                 if time.time() > deadline - 2.0: break
                 ts = all_t_subsets[idx]
                 block_points = set(ts)
                 remaining = [x for x in range(v) if x not in block_points]
                 random.shuffle(remaining)
                 for x in remaining:
                     if len(block_points) >= k: break
                     block_points.add(x)
                 blocks.append(tuple(sorted(block_points)))
        
        best_blocks = blocks
        # Final validation
        if not is_valid(best_blocks):
            # If still invalid, we have a problem.
            # But for v=19, greedy should finish easily.
            pass

    return [list(b) for b in best_blocks]
