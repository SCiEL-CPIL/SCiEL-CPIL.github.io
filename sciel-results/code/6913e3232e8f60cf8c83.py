import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build sparse structure: block_to_tsubs (which t-subsets each block covers)
    block_to_tsubs = []
    for B in all_kblocks:
        covered = []
        for ts in itertools.combinations(B, t):
            covered.append(tsub_to_idx[ts])
        block_to_tsubs.append(covered)
    
    # Build inverse: tsub_to_blocks (which blocks cover each t-subset)
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays for fast indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Greedy Construction with sparse gain tracking
    uncovered = np.ones(n_tsubs, dtype=bool)
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(n_blocks, k_choose_t, dtype=np.int32)
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline - 30.0:
            break
        
        b = int(np.argmax(gains))
        if gains[b] == 0:
            break
        
        selected_block_indices.append(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                gains[tsub_to_blocks_np[s_idx]] -= 1
    
    # 3. Local Optimization: ILS with constrained swaps
    current_blocks = list(selected_block_indices)
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_blocks:
        for s in block_to_tsubs[idx]:
            cover_counts[s] += 1
    
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    def reduce_redundant(blocks, counts):
        """Remove blocks that do not cover any unique t-subset."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in blocks:
                # Check if any t-subset covered by idx has count == 1
                is_unique = False
                for s in block_to_tsubs[idx]:
                    if counts[s] == 1:
                        is_unique = True
                        break
                if is_unique:
                    new_blocks.append(idx)
                else:
                    for s in block_to_tsubs[idx]:
                        counts[s] -= 1
                    changed = True
            blocks = new_blocks
        return blocks, counts
    
    def repair(blocks, counts):
        """Greedily add blocks to cover all t-subsets using sparse gains."""
        # Compute uncovered t-subsets
        uncovered_mask = (counts == 0)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gains for all blocks based on uncovered t-subsets
            gains = np.zeros(n_blocks, dtype=np.int32)
            for s_idx in np.where(uncovered_mask)[0]:
                for b_idx in tsub_to_blocks_np[s_idx]:
                    gains[b_idx] += 1
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            blocks.append(best_idx)
            for s in block_to_tsubs[best_idx]:
                counts[s] += 1
                if counts[s] == 1:
                    uncovered_mask[s] = False
        
        return blocks
    
    def try_replace_one(blocks, counts, max_candidates=100):
        """Try to replace one block with another that maintains coverage."""
        n = len(blocks)
        if n == 0:
            return blocks, counts, False
        
        pos = random.randrange(n)
        old_idx = blocks[pos]
        
        # Get t-subsets covered by old block
        old_tsubs = block_to_tsubs[old_idx]
        
        # Compute overlap: for each candidate, how many of old_idx's t-subsets does it cover?
        # We want candidates that cover most of the old block's unique t-subsets
        # First, find which t-subsets are uniquely covered by old_idx
        unique_tsubs = [s for s in old_tsubs if counts[s] == 1]
        
        if not unique_tsubs:
            # Old block is redundant, just remove it
            for s in old_tsubs:
                counts[s] -= 1
            blocks = blocks[:pos] + blocks[pos+1:]
            return blocks, counts, True
        
        # Get candidates that cover the first unique t-subset
        first_s = unique_tsubs[0]
        candidates = tsub_to_blocks_np[first_s]
        
        # Filter candidates: must cover all unique t-subsets
        valid_candidates = []
        for c in candidates:
            c = int(c)
            if c == old_idx or c in blocks:
                continue
            # Check if c covers all unique t-subsets
            covers_all = True
            c_tsubs_set = set(block_to_tsubs[c])
            for s in unique_tsubs:
                if s not in c_tsubs_set:
                    covers_all = False
                    break
            if covers_all:
                valid_candidates.append(c)
        
        if not valid_candidates:
            return blocks, counts, False
        
        # Try each valid candidate
        random.shuffle(valid_candidates)
        for cand_idx in valid_candidates[:max_candidates]:
            # Simulate replacement
            new_counts = counts.copy()
            for s in old_tsubs:
                new_counts[s] -= 1
            for s in block_to_tsubs[cand_idx]:
                new_counts[s] += 1
            
            # Check validity
            if np.any(new_counts == 0):
                continue
            
            # Try reduction
            new_blocks = list(blocks)
            new_blocks[pos] = cand_idx
            reduced, reduced_counts = reduce_redundant(new_blocks, new_counts)
            
            if len(reduced) < len(blocks):
                return reduced, reduced_counts, True
            elif len(reduced) == len(blocks) and random.random() < 0.15:
                return reduced, reduced_counts, True
        
        return blocks, counts, False
    
    # Main ILS Loop
    random.seed(42)
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
            stall_count = 0
        else:
            stall_count += 1
        
        if stall_count > max_stall:
            # Restart from best
            current_blocks = list(best_blocks)
            cover_counts = np.zeros(n_tsubs, dtype=np.int32)
            for idx in current_blocks:
                for s in block_to_tsubs[idx]:
                    cover_counts[s] += 1
            stall_count = 0
            max_stall = int(max_stall * 1.2)
        
        # 2. Perturb - choose strategy
        r = random.random()
        
        if r < 0.40 and len(current_blocks) > 1:
            # Strategy A: Remove 1 random block
            idx = random.choice(current_blocks)
            for s in block_to_tsubs[idx]:
                cover_counts[s] -= 1
            current_blocks = [b for b in current_blocks if b != idx]
            
        elif r < 0.75 and len(current_blocks) > 1:
            # Strategy B: Try block replacement (constrained swap)
            current_blocks, cover_counts, success = try_replace_one(
                current_blocks, cover_counts, max_candidates=80
            )
            if not success:
                # Fallback: remove one random block
                idx = random.choice(current_blocks)
                for s in block_to_tsubs[idx]:
                    cover_counts[s] -= 1
                current_blocks = [b for b in current_blocks if b != idx]
            
        else:
            # Strategy C: Remove 2 random blocks (aggressive)
            if len(current_blocks) > 2:
                removals = random.sample(current_blocks, 2)
                for idx in removals:
                    for s in block_to_tsubs[idx]:
                        cover_counts[s] -= 1
                current_blocks = [b for b in current_blocks if b not in removals]
            elif len(current_blocks) > 1:
                idx = random.choice(current_blocks)
                for s in block_to_tsubs[idx]:
                    cover_counts[s] -= 1
                current_blocks = [b for b in current_blocks if b != idx]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance: ILS accepts if size <= best, or with probability based on delta
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            best_blocks = list(current_blocks)
            best_size = new_size
            stall_count = 0
        elif delta == 0:
            # Same size, accept with some probability for diversity
            if random.random() < 0.3:
                pass  # Keep current
            else:
                # Revert to best
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    for s in block_to_tsubs[idx]:
                        cover_counts[s] += 1
        else:
            # Worse, accept with small probability (annealing-like)
            prob = np.exp(-delta / 5.0)
            if random.random() < prob:
                pass  # Keep current
            else:
                # Revert to best
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    for s in block_to_tsubs[idx]:
                        cover_counts[s] += 1
    
    # Final reduction and validation
    final_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in best_blocks:
        for s in block_to_tsubs[idx]:
            final_counts[s] += 1
    
    final_blocks, final_counts = reduce_redundant(best_blocks, final_counts)
    
    # Verify validity
    if np.any(final_counts == 0):
        final_blocks = best_blocks
    
    # Final polish: try a few more replacement moves
    if time.time() < deadline - 2.0:
        temp_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in final_blocks:
            for s in block_to_tsubs[idx]:
                temp_counts[s] += 1
        
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            final_blocks, temp_counts, success = try_replace_one(
                final_blocks, temp_counts, max_candidates=60
            )
            if not success:
                break
        
        if np.any(temp_counts == 0):
            final_blocks = best_blocks
    
    return [all_kblocks[i] for i in final_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
