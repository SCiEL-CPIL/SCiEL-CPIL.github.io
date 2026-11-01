import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer before 120s limit
    
    # Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_k_subs = list(itertools.combinations(range(v), k))
    k_sub_t_indices = []
    for k_sub in all_k_subs:
        t_indices = [t_sub_index[sub] for sub in itertools.combinations(k_sub, t)]
        k_sub_t_indices.append(t_indices)
        
    num_k_subs = len(all_k_subs)
    
    # === PHASE 1: Greedy Cover ===
    cover_count = [0] * num_t_subs
    blocks = []
    block_set = set()
    
    def add_block(idx):
        for t_idx in k_sub_t_indices[idx]:
            cover_count[t_idx] += 1
        blocks.append(idx)
        block_set.add(idx)
        
    def remove_block(idx):
        blocks.remove(idx)
        block_set.discard(idx)
        for t_idx in k_sub_t_indices[idx]:
            cover_count[t_idx] -= 1
    
    # Greedy: pick block covering most uncovered
    while time.time() < deadline - 80:  # Reserve 80s for local search
        best_gain = -1
        best_idx = -1
        for i in range(num_k_subs):
            if i in block_set:
                continue
            gain = 0
            for t_idx in k_sub_t_indices[i]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if best_gain == len(k_sub_t_indices[i]):
                break
        if best_gain == 0:
            break
        add_block(best_idx)
    
    def is_valid(cover):
        for c in cover:
            if c == 0:
                return False
        return True
    
    if not is_valid(cover_count):
        # Emergency repair
        while not is_valid(cover_count):
            if time.time() > deadline - 5:
                break
            # Find uncovered t-subsets
            uncovered = [i for i, c in enumerate(cover_count) if c == 0]
            if not uncovered:
                break
            # Find block covering most uncovered
            best_gain = 0
            best_idx = -1
            for i in range(num_k_subs):
                if i in block_set:
                    continue
                gain = sum(1 for t_idx in k_sub_t_indices[i] if cover_count[t_idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            add_block(best_idx)
    
    best_blocks = blocks[:]
    best_len = len(best_blocks)
    current_cover = cover_count[:]
    current_block_set = block_set.copy()
    
    # === PHASE 2: Targeted Local Search ===
    # Focus on removing blocks that are "redundant" (all their t-subsets are covered by others)
    # and swapping to find better configurations
    
    def get_redundancy(idx, cover, bset):
        """Count how many t-subsets of this block are covered by other blocks"""
        redundant = 0
        for t_idx in k_sub_t_indices[idx]:
            if cover[t_idx] > 1:  # Covered by at least one other block
                redundant += 1
        return redundant
    
    def try_remove_block(idx, cover, bset, blist):
        """Try to remove a block, return True if successful (remains valid)"""
        # Check if removing would leave any t-subset uncovered
        for t_idx in k_sub_t_indices[idx]:
            if cover[t_idx] == 1:
                return False  # This block is the only one covering this t-subset
        # Safe to remove
        for t_idx in k_sub_t_indices[idx]:
            cover[t_idx] -= 1
        blist.remove(idx)
        bset.discard(idx)
        return True
    
    def find_redundant_blocks(cover, bset):
        """Find blocks that can be safely removed"""
        redundant = []
        for idx in list(bset):
            can_remove = True
            for t_idx in k_sub_t_indices[idx]:
                if cover[t_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                redundant.append(idx)
        return redundant
    
    # Iterative removal of redundant blocks
    while time.time() < deadline - 20:
        redundant = find_redundant_blocks(cover_count, block_set)
        if not redundant:
            break
        # Remove one redundant block at a time, prioritizing those with most redundancy
        # Actually, just remove any redundant block
        idx_to_remove = redundant[0]
        for t_idx in k_sub_t_indices[idx_to_remove]:
            cover_count[t_idx] -= 1
        blocks.remove(idx_to_remove)
        block_set.discard(idx_to_remove)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Now do local search with swaps
    T = 2.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline - 5:
        move_type = random.random()
        
        if move_type < 0.5:  # Try remove
            if not blocks:
                continue
            # Prefer removing blocks with high redundancy
            candidates = []
            for idx in blocks:
                red = 0
                for t_idx in k_sub_t_indices[idx]:
                    if cover_count[t_idx] > 1:
                        red += 1
                # Only consider if safe
                safe = all(cover_count[t_idx] > 1 for t_idx in k_sub_t_indices[idx])
                if safe:
                    candidates.append((red, idx))
            
            if not candidates:
                continue
            
            # Pick from top candidates
            candidates.sort(reverse=True)
            top_candidates = candidates[:min(5, len(candidates))]
            _, idx_to_remove = random.choice(top_candidates)
            
            # Apply
            for t_idx in k_sub_t_indices[idx_to_remove]:
                cover_count[t_idx] -= 1
            blocks.remove(idx_to_remove)
            block_set.discard(idx_to_remove)
            
            new_len = len(blocks)
            delta = new_len - best_len  # We want to minimize
            
            if new_len < best_len or (new_len == best_len and random.random() < 0.5):
                best_len = new_len
                best_blocks = blocks[:]
            else:
                # Revert
                blocks.append(idx_to_remove)
                block_set.add(idx_to_remove)
                for t_idx in k_sub_t_indices[idx_to_remove]:
                    cover_count[t_idx] += 1
                    
        elif move_type < 0.8:  # Try swap
            if not blocks:
                continue
            idx_to_swap = random.choice(blocks)
            
            # Find candidate that covers more unique t-subsets
            while True:
                cand_idx = random.randint(0, num_k_subs - 1)
                if cand_idx not in block_set:
                    break
            
            # Calculate gain of swap
            # Remove old: lose coverage of its t-subsets
            # Add new: gain coverage of its t-subsets
            old_unique = 0
            for t_idx in k_sub_t_indices[idx_to_swap]:
                if cover_count[t_idx] == 1:
                    old_unique += 1
            
            new_unique = 0
            for t_idx in k_sub_t_indices[cand_idx]:
                if cover_count[t_idx] == 0:
                    new_unique += 1
            
            # Only swap if it doesn't create uncovered sets
            would_uncover = old_unique > 0  # Removing would uncover some
            if would_uncover:
                # Check if new block covers those uncovered ones
                uncovered_by_old = [t_idx for t_idx in k_sub_t_indices[idx_to_swap] if cover_count[t_idx] == 1]
                covered_by_new = set(k_sub_t_indices[cand_idx])
                if not all(t_idx in covered_by_new for t_idx in uncovered_by_old):
                    continue  # This swap would create uncovered sets
            
            # Apply swap
            for t_idx in k_sub_t_indices[idx_to_swap]:
                cover_count[t_idx] -= 1
            for t_idx in k_sub_t_indices[cand_idx]:
                cover_count[t_idx] += 1
            
            blocks.remove(idx_to_swap)
            block_set.discard(idx_to_swap)
            blocks.append(cand_idx)
            block_set.add(cand_idx)
            
            # Check validity
            valid = all(c > 0 for c in cover_count)
            if valid and len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
            elif not valid:
                # Revert
                for t_idx in k_sub_t_indices[cand_idx]:
                    cover_count[t_idx] -= 1
                for t_idx in k_sub_t_indices[idx_to_swap]:
                    cover_count[t_idx] += 1
                blocks.remove(cand_idx)
                block_set.discard(cand_idx)
                blocks.append(idx_to_swap)
                block_set.add(idx_to_swap)
            else:
                # Valid but not better, might still keep for exploration
                if random.random() < 0.3:  # 30% chance to keep
                    pass
                else:
                    # Revert
                    for t_idx in k_sub_t_indices[cand_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_indices[idx_to_swap]:
                        cover_count[t_idx] += 1
                    blocks.remove(cand_idx)
                    block_set.discard(cand_idx)
                    blocks.append(idx_to_swap)
                    block_set.add(idx_to_swap)
                    
        else:  # Try add (less common, for exploration)
            while True:
                cand_idx = random.randint(0, num_k_subs - 1)
                if cand_idx not in block_set:
                    break
            
            # Only add if it covers many uncovered
            new_coverage = sum(1 for t_idx in k_sub_t_indices[cand_idx] if cover_count[t_idx] == 0)
            if new_coverage >= 3:  # Must cover at least 3 uncovered
                for t_idx in k_sub_t_indices[cand_idx]:
                    cover_count[t_idx] += 1
                blocks.append(cand_idx)
                block_set.add(cand_idx)
            
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final cleanup: try to remove redundant blocks one more time
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        redundant = find_redundant_blocks(cover_count, block_set)
        if not redundant:
            break
        idx_to_remove = redundant[0]
        for t_idx in k_sub_t_indices[idx_to_remove]:
            cover_count[t_idx] -= 1
        blocks.remove(idx_to_remove)
        block_set.discard(idx_to_remove)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Ensure best_blocks is valid
    if not best_blocks:
        best_blocks = blocks[:]
    
    # Convert indices back to lists
    result = [list(all_k_subs[i]) for i in best_blocks]
    
    # Sort blocks and points for consistency
    for b in result:
        b.sort()
    result.sort()
    
    return result
