import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # Precompute triples for each potential block? No, too many blocks.
    # Instead, precompute the list of triple indices for a block on the fly or cache.
    # For v=24, k=7, there are C(24,7) = 346104 blocks.
    # We can precompute the triple indices for each block to speed up add/remove.
    # But storing 346k * 35 ints is ~12M ints, which is fine.
    # Let's create a mapping from block_tuple -> list_of_triple_indices
    
    # Actually, generating all blocks is fast.
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_triples = {}
    for b in all_blocks:
        trs = list(itertools.combinations(b, t))
        block_to_triples[b] = np.array([triple_to_idx[tr] for tr in trs], dtype=np.int32)
        
    # Map block tuple to its index for easier reference?
    # We will represent blocks as tuples.
    
    # Initial Solution: Greedy Construction
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []
    block_set = set()
    
    def add_block(block):
        if block in block_set:
            return False
        block_set.add(block)
        blocks.append(block)
        tr_indices = block_to_triples[block]
        count[tr_indices] += 1
        return True

    def remove_block(block):
        tr_indices = block_to_triples[block]
        count[tr_indices] -= 1
        block_set.discard(block)
        # Find index in list and remove
        # This is O(N), but N is small (<100)
        for i, b in enumerate(blocks):
            if b == block:
                blocks.pop(i)
                return
        # Should not happen if block_set is consistent

    # Greedy Phase
    # We want to cover all triples.
    # Pick a random uncovered triple, find a block containing it that covers most other uncovered triples.
    for _ in range(500):
        if time.time() > deadline - 20:
            break
        uncovered_indices = np.where(count == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered triple
        target_tr_idx = uncovered_indices[rng.randint(len(uncovered_indices))]
        target_tr = all_triples[target_tr_idx]
        
        # Find all blocks containing this triple
        # This is expensive if done by scanning all blocks.
        # Better: Precompute for each triple, the list of blocks containing it.
        # triple_to_blocks[tr_idx] = [block1, block2, ...]
        
        # Let's build this index once.
        # 2024 triples * ~120 blocks/triple = 240k entries. Fast.
        
    # Re-structuring for speed: Build triple_to_blocks index
    triple_to_blocks = defaultdict(list)
    for b in all_blocks:
        tr_indices = block_to_triples[b]
        for idx in tr_indices:
            triple_to_blocks[idx].append(b)
            
    # Redo Greedy Phase with index
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []
    block_set = set()
    
    for _ in range(1000):
        if time.time() > deadline - 30:
            break
        uncovered_indices = np.where(count == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        target_tr_idx = uncovered_indices[rng.randint(len(uncovered_indices))]
        candidates = triple_to_blocks[target_tr_idx]
        
        if not candidates:
            # Should not happen for v=24, k=7, t=3
            break
            
        best_block = None
        best_gain = -1
        
        # Sample candidates if too many? There are ~120 blocks per triple. 120 is small enough.
        for b in candidates:
            if b in block_set:
                continue
            tr_indices = block_to_triples[b]
            gain = np.sum(count[tr_indices] == 0)
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is not None:
            add_block(best_block)

    # Ensure full coverage (just in case)
    while time.time() < deadline - 10:
        uncovered_indices = np.where(count == 0)[0]
        if len(uncovered_indices) == 0:
            break
        target_tr_idx = uncovered_indices[0]
        candidates = triple_to_blocks[target_tr_idx]
        # Pick first available
        for b in candidates:
            if b not in block_set:
                add_block(b)
                break
        else:
            # If all candidates are in set, we have a problem.
            # But greedy should have covered this.
            # Fallback: add a random block
            if all_blocks:
                b = all_blocks[rng.randint(len(all_blocks))]
                if b not in block_set:
                    add_block(b)
            else:
                break

    # Simulated Annealing Phase
    # Objective: Minimize number of blocks.
    # Move: 
    # 1. Remove a random block. If valid, accept.
    # 2. Swap a random block with a random other block. If valid and (improves or accept with prob), accept.
    # 3. Add a random block (rarely useful for minimization, but helps escape).
    
    # We will focus on Removal and Swap.
    
    current_blocks = list(blocks)
    current_count = count.copy()
    
    best_blocks = list(current_blocks)
    best_count = current_count.copy()
    best_len = len(current_blocks)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # Precompute block indices for faster access?
    # We use tuples as keys.
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Random Move
        move_type = rng.rand()
        
        if move_type < 0.6:
            # Try to remove a block
            if len(current_blocks) == 0:
                continue
            idx = rng.randint(len(current_blocks))
            block_to_remove = current_blocks[idx]
            
            # Check if removal is valid
            tr_indices = block_to_triples[block_to_remove]
            # We need to check if any count becomes 0
            # current_count[tr_indices] >= 1 for all?
            if np.all(current_count[tr_indices] >= 1):
                # Valid removal
                current_count[tr_indices] -= 1
                current_blocks.pop(idx)
                
                # Accept (always accept improvement in minimization)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
                    best_count = current_count.copy()
            else:
                # Invalid removal. 
                # In SA, we might "accept" an invalid state with penalty? 
                # No, we must maintain validity for the final output.
                # So we only accept valid moves.
                # However, to escape local minima, we can allow "temporary" invalidity?
                # That's complex. Let's stick to valid moves only for now, 
                # but use Swap to change the structure.
                pass
                
        elif move_type < 0.9:
            # Swap
            if len(current_blocks) == 0:
                continue
            idx = rng.randint(len(current_blocks))
            block_old = current_blocks[idx]
            
            # Pick a new block
            # To make swap effective, pick a block that covers some uncovered triples 
            # or overlaps poorly.
            # Simple: Random block from all_blocks.
            block_new = all_blocks[rng.randint(len(all_blocks))]
            
            if block_new in block_set:
                # If block_new is already in the set, swapping is just moving it? 
                # No, if block_new is in set, and block_old is in set, 
                # removing old and adding new (which is already there) is just removing old.
                # But we want to change the set.
                # Let's ensure block_new is NOT in current set.
                # If it is, skip.
                continue
                
            # Check validity of swap
            tr_old = block_to_triples[block_old]
            tr_new = block_to_triples[block_new]
            
            # New counts would be:
            # count_new = count - count_old_triples + count_new_triples
            
            # We can check validity without updating:
            # For all tr in tr_old: count[tr] - 1 + (1 if tr in tr_new else 0) >= 1
            # For all tr in tr_new: count[tr] + 1 - (1 if tr in tr_old else 0) >= 0 (always true)
            
            # So we only need to check tr_old.
            # count[tr_old] >= 1 - (1 if tr_old in tr_new else 0)
            
            # Intersection of tr_old and tr_new
            # We can use numpy to check.
            # Let's just simulate the count change on a copy? No, too slow.
            
            # Logic:
            # A triple is uncovered after swap if:
            # 1. It was in block_old (count decreased by 1)
            # 2. It is NOT in block_new (count not increased)
            # 3. Its current count was 1.
            
            # So: count[tr] == 1 AND tr in block_old AND tr NOT in block_new
            
            # Let's compute this.
            # tr_old is an array of indices.
            # tr_new is an array of indices.
            
            # Create a set of tr_new for O(1) lookup?
            # tr_new has 35 elements. Set creation is fast.
            set_new = set(tr_new.tolist())
            
            # Check tr_old
            # We need to check count[tr_old] == 1 for those not in set_new.
            # mask = ~np.isin(tr_old, tr_new) # This might be slow for small arrays?
            # Let's just loop, 35 elements is small.
            
            valid_swap = True
            for tr_idx in tr_old:
                if current_count[tr_idx] == 1 and tr_idx not in set_new:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform swap
                current_count[tr_old] -= 1
                current_count[tr_new] += 1
                current_blocks[idx] = block_new
                block_set.discard(block_old)
                block_set.add(block_new)
                
                # Swap doesn't change length, so best_len doesn't update.
                # But it might enable future removals.
                
        else:
            # Add a block (to escape)
            if len(current_blocks) > 100: # Don't grow too much
                continue
            block_new = all_blocks[rng.randint(len(all_blocks))]
            if block_new in block_set:
                continue
            
            tr_new = block_to_triples[block_new]
            current_count[tr_new] += 1
            current_blocks.append(block_new)
            block_set.add(block_new)
            
            # This increases length, so we don't update best.
            # But it might help later.

        T *= cooling_rate
        
        # Restart if stuck?
        # Simple restart: if no improvement in last X steps, reset T?
        # Let's just let it cool.

    # Final Polish: Greedy Removal
    # Try to remove blocks in the best solution found
    final_blocks = list(best_blocks)
    final_count = best_count.copy()
    
    # Re-initialize count for final_blocks to be safe
    final_count = np.zeros(n_all, dtype=np.int32)
    final_set = set()
    for b in final_blocks:
        final_count[block_to_triples[b]] += 1
        final_set.add(b)
        
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(final_blocks)):
            if time.time() > deadline - 1:
                break
            b = final_blocks[i]
            tr_indices = block_to_triples[b]
            if np.all(final_count[tr_indices] >= 1):
                # Can remove
                final_count[tr_indices] -= 1
                final_blocks.pop(i)
                final_set.discard(b)
                changed = True
                break # Restart loop as indices changed

    return [list(b) for b in final_blocks]
