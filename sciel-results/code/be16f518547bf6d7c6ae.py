import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin for 150s limit
    
    def get_elapsed():
        return time.time() - start_time

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # All possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
    
    # For each t-subset, list of block indices that cover it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
    
    # Convert to numpy arrays for fast set operations
    # Use sorted lists and np.intersect1d or set operations
    
    def greedy_cover():
        """Standard greedy set cover with random tie-breaking."""
        uncovered = set(range(num_tsubs))
        selected = []
        
        while uncovered and get_elapsed() < deadline - 5.0:
            # Pick a random uncovered t-subset
            if len(uncovered) < 50:
                s_idx = random.choice(list(uncovered))
            else:
                s_idx = random.randint(0, num_tsubs - 1)
                if s_idx not in uncovered:
                    continue
            
            # Find block with maximum marginal gain
            best_block = -1
            best_gain = -1
            
            candidates = tsub_blocks[s_idx]
            # Sample if too many candidates
            if len(candidates) > 30:
                sampled = random.sample(candidates, 30)
            else:
                sampled = candidates
            
            for b_idx in sampled:
                gain = len(uncovered & set(block_tsubs[b_idx]))
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1 or best_gain == 0:
                break
                
            selected.append(best_block)
            uncovered -= set(block_tsubs[best_block])
        
        return selected
    
    def prune_to_minimal(blocks):
        """Remove redundant blocks iteratively."""
        blocks = list(set(blocks))
        changed = True
        while changed and get_elapsed() < deadline - 3.0:
            changed = False
            # Count coverage
            cover_count = np.zeros(num_tsubs, dtype=np.int32)
            for b_idx in blocks:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            
            new_blocks = []
            for b_idx in blocks:
                # Check if any t-subset is uniquely covered by this block
                essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        essential = True
                        break
                if essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
            
            blocks = new_blocks
        return blocks
    
    def local_search_2swap(blocks):
        """
        Local search with 2-swap moves:
        - Remove 1 or 2 blocks, add 1 or 2 blocks
        - Accept if size decreases or stays same (with random acceptance for neutral)
        """
        blocks = set(blocks)
        best_size = len(blocks)
        best_blocks = set(blocks)
        
        # Compute coverage
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in blocks:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1
        
        temperature = 3.0
        cooling = 0.995
        min_temp = 0.01
        iterations = 0
        
        while temperature > min_temp and get_elapsed() < deadline - 10.0:
            iterations += 1
            if iterations % 1000 == 0:
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = set(blocks)
            
            # Randomly choose a move type
            move_type = random.random()
            
            if move_type < 0.4:
                # 1-remove move
                if not blocks:
                    break
                b_out = random.choice(list(blocks))
                
                # Check if safe to remove
                critical = [j for j in block_tsubs[b_out] if cover_count[j] == 1]
                if not critical:
                    # Safe removal
                    blocks.remove(b_out)
                    for j in block_tsubs[b_out]:
                        cover_count[j] -= 1
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = set(blocks)
                else:
                    # Try to find a replacement
                    # Find blocks that cover all critical t-subsets
                    candidates = tsub_blocks[critical[0]]
                    best_in = -1
                    best_score = -1
                    
                    if len(candidates) > 20:
                        sampled = random.sample(candidates, 20)
                    else:
                        sampled = candidates
                    
                    for b_in in sampled:
                        if b_in == b_out:
                            continue
                        # How many critical does it cover?
                        covered = sum(1 for j in critical if j in set(block_tsubs[b_in]))
                        # Score: covered - penalty for extra
                        extra = len(set(block_tsubs[b_in]) - set(critical))
                        score = covered * 2 - extra
                        if score > best_score:
                            best_score = score
                            best_in = b_in
                    
                    if best_in != -1:
                        # Calculate delta
                        delta = len(critical) - sum(1 for j in critical if j in set(block_tsubs[best_in]))
                        if delta == 0 or random.random() < math.exp(-delta / temperature):
                            blocks.remove(b_out)
                            for j in block_tsubs[b_out]:
                                cover_count[j] -= 1
                            blocks.add(best_in)
                            for j in block_tsubs[best_in]:
                                cover_count[j] += 1
                            if len(blocks) < best_size:
                                best_size = len(blocks)
                                best_blocks = set(blocks)
            
            elif move_type < 0.7:
                # 1-swap move (remove 1, add 1)
                if not blocks:
                    break
                b_out = random.choice(list(blocks))
                
                # Find blocks that share at least some t-subsets with b_out
                # Use the t-subsets covered by b_out
                tsubs_of_out = set(block_tsubs[b_out])
                
                # Find candidate blocks that cover some of the same t-subsets
                candidate_set = set()
                for j in list(tsubs_of_out)[:5]:  # Sample some t-subsets
                    for b in tsub_blocks[j]:
                        if b != b_out:
                            candidate_set.add(b)
                
                if candidate_set:
                    b_in = random.choice(list(candidate_set))
                    
                    # Calculate new size and coverage change
                    old_size = len(blocks)
                    
                    # Simulate the swap
                    new_cover_count = cover_count.copy()
                    for j in block_tsubs[b_out]:
                        new_cover_count[j] -= 1
                    for j in block_tsubs[b_in]:
                        new_cover_count[j] += 1
                    
                    # Check validity: no t-subset should have count < 1
                    if np.all(new_cover_count >= 1):
                        # Valid swap
                        blocks.remove(b_out)
                        blocks.add(b_in)
                        cover_count = new_cover_count
                        if len(blocks) < best_size:
                            best_size = len(blocks)
                            best_blocks = set(blocks)
                    else:
                        # Invalid, try random acceptance with penalty
                        uncovered_count = np.sum(new_cover_count < 1)
                        if random.random() < math.exp(-uncovered_count / temperature):
                            # Accept invalid state temporarily? No, revert for simplicity
                            pass
            
            else:
                # 2-remove, 1-add move (if possible)
                if len(blocks) < 2:
                    break
                b_out1, b_out2 = random.sample(list(blocks), 2)
                
                # Critical t-subsets: those covered only by out1 or out2
                tsubs_1 = set(block_tsubs[b_out1])
                tsubs_2 = set(block_tsubs[b_out2])
                critical = [j for j in (tsubs_1 | tsubs_2) if cover_count[j] <= 2 and 
                          ((j in tsubs_1 and cover_count[j] == 1) or 
                           (j in tsubs_2 and cover_count[j] == 1) or
                           (j in tsubs_1 and j in tsubs_2 and cover_count[j] == 2))]
                
                if not critical:
                    # Safe 2-removal
                    blocks.remove(b_out1)
                    blocks.remove(b_out2)
                    for j in block_tsubs[b_out1]:
                        cover_count[j] -= 1
                    for j in block_tsubs[b_out2]:
                        cover_count[j] -= 1
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = set(blocks)
                else:
                    # Try to find a single block that covers all critical
                    if critical:
                        candidates = tsub_blocks[critical[0]]
                        best_in = -1
                        best_covered = -1
                        
                        if len(candidates) > 15:
                            sampled = random.sample(candidates, 15)
                        else:
                            sampled = candidates
                        
                        for b_in in sampled:
                            if b_in in (b_out1, b_out2):
                                continue
                            covered = sum(1 for j in critical if j in set(block_tsubs[b_in]))
                            if covered > best_covered:
                                best_covered = covered
                                best_in = b_in
                        
                        if best_in != -1 and best_covered == len(critical):
                            # Perfect 2-remove, 1-add
                            blocks.remove(b_out1)
                            blocks.remove(b_out2)
                            for j in block_tsubs[b_out1]:
                                cover_count[j] -= 1
                            for j in block_tsubs[b_out2]:
                                cover_count[j] -= 1
                            blocks.add(best_in)
                            for j in block_tsubs[best_in]:
                                cover_count[j] += 1
                            if len(blocks) < best_size:
                                best_size = len(blocks)
                                best_blocks = set(blocks)
            
            temperature *= cooling
        
        return best_blocks
    
    # Main execution: multiple restarts of greedy + prune + local search
    best_blocks = []
    best_len = float('inf')
    
    # Run multiple iterations
    num_restarts = 0
    while get_elapsed() < deadline - 8.0:
        num_restarts += 1
        
        # 1. Greedy cover
        cover = greedy_cover()
        
        # 2. Prune to minimal
        cover = prune_to_minimal(cover)
        
        # 3. Local search
        cover = local_search_2swap(cover)
        
        # 4. Final prune
        cover = prune_to_minimal(cover)
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_blocks = set(cover)
        
        # If we've done enough restarts, break
        if num_restarts >= 50 and get_elapsed() > deadline - 20.0:
            break
    
    if not best_blocks:
        # Fallback: return a valid (possibly large) cover
        cover = greedy_cover()
        best_blocks = set(cover)
    
    return [list(all_blocks[i]) for i in best_blocks]
