import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Pre-computation
    # Generate all blocks
    all_points = range(v)
    all_blocks = list(itertools.combinations(all_points, k))
    num_blocks = len(all_blocks)
    
    # Generate all triples
    all_triples = list(itertools.combinations(all_points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Map block index to list of triple indices it covers
    block_triples = [[] for _ in range(num_blocks)]
    # Map triple index to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    
    for bi, block in enumerate(all_blocks):
        for tri in itertools.combinations(block, t):
            ti = triple_to_idx[tri]
            block_triples[bi].append(ti)
            triple_blocks[ti].append(bi)

    # 2. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(num_triples))
        selected = []
        
        # Optimization: Pick the block that covers the most uncovered triples
        # To speed this up, we can use a simple heuristic:
        # Iterate through uncovered triples, find candidates, pick one with high degree?
        # Or just standard greedy:
        
        while uncovered:
            # Pick a random uncovered triple to bias search slightly
            # Or pick the one with fewest covering blocks (most constrained)
            # Let's pick the triple with the fewest covering blocks to reduce options
            target_tri = min(uncovered, key=lambda x: len(triple_blocks[x]))
            
            best_block = -1
            max_gain = -1
            
            # Candidates are blocks covering target_tri
            candidates = triple_blocks[target_tri]
            
            for bi in candidates:
                gain = 0
                # Calculate gain: how many new triples does bi cover?
                # We can do this by checking intersection with uncovered
                # Since uncovered is a set, this is O(1) per check? No, O(|block_triples[bi]|)
                # block size 7, t=3, so 35 triples per block.
                for ti in block_triples[bi]:
                    if ti in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_block = bi
            
            if best_block == -1:
                break # Should not happen
                
            selected.append(best_block)
            # Update uncovered
            for ti in block_triples[best_block]:
                if ti in uncovered:
                    uncovered.remove(ti)
                    
        return selected

    # 3. Local Optimization
    def local_optimize(state, counts):
        # state: list of block indices
        # counts: list of int, counts[i] = number of blocks covering triple i
        
        changed = True
        while changed:
            changed = False
            
            # 1. Try Deletion
            for i in range(len(state) - 1, -1, -1):
                bi = state[i]
                # Check if any triple in bi has count == 1
                is_redundant = True
                for ti in block_triples[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block bi
                    for ti in block_triples[bi]:
                        counts[ti] -= 1
                    state.pop(i)
                    changed = True
                    break # Restart scan
            
            if changed:
                continue

            # 2. Try 1-Swap
            # For each block in state, see if we can replace it
            for i in range(len(state)):
                bi = state[i]
                # Find critical triples: those covered ONLY by bi
                critical = [ti for ti in block_triples[bi] if counts[ti] == 1]
                
                if not critical:
                    continue # This would have been handled by deletion logic?
                             # Wait, if critical is empty, it's redundant.
                             # The deletion loop above should have caught it.
                             # Why didn't it? Maybe it did, and we are in a new iteration?
                             # No, if we found a redundant block, we break and restart.
                             # So if we are here, NO block is redundant.
                             # Thus, every block MUST have at least one critical triple.
                             # So critical should never be empty here.
                             # Let's assume it's possible due to race conditions or logic gap.
                             # If critical is empty, we can just delete it.
                             # But let's stick to swaps for now.

                # Find candidates for replacement
                # Must cover all critical triples
                candidates = set(triple_blocks[critical[0]])
                for ti in critical[1:]:
                    candidates.intersection_update(triple_blocks[ti])
                    
                # Filter candidates not in current state
                current_set = set(state)
                valid_candidates = [c for c in candidates if c not in current_set]
                
                # Try random candidates to avoid bias
                random.shuffle(valid_candidates)
                
                for bi_new in valid_candidates:
                    # Check if swap is valid
                    # We established: if bi_new covers critical(bi), swap is valid.
                    # So we just need to check if bi_new is not in state (already done)
                    # And we assume the coverage condition holds.
                    
                    # Perform Swap
                    state[i] = bi_new
                    
                    # Update counts
                    # Old block bi
                    for ti in block_triples[bi]:
                        counts[ti] -= 1
                    # New block bi_new
                    for ti in block_triples[bi_new]:
                        counts[ti] += 1
                    
                    changed = True
                    break # Restart optimization
                    
                if changed:
                    break
        
        return state, counts

    def compute_counts(state):
        counts = [0] * num_triples
        for bi in state:
            for ti in block_triples[bi]:
                counts[ti] += 1
        return counts

    # 4. Iterated Local Search
    best_state = None
    best_len = float('inf')
    
    # Initial restarts
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        state = greedy_construct(42 + restart)
        counts = compute_counts(state)
        state, counts = local_optimize(state, counts)
        
        if len(state) < best_len:
            best_len = len(state)
            best_state = state[:]
            # print(f"Restart {restart}: {len(state)} blocks")

    # ILS Loop
    current_state = best_state[:]
    current_counts = compute_counts(current_state)
    current_state, current_counts = local_optimize(current_state, current_counts)
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        
        # Kick: Perturb
        # Remove 2-4 random blocks
        kick_size = random.randint(2, 4)
        if len(current_state) <= kick_size:
            kick_size = len(current_state) // 2
            
        remove_indices = random.sample(range(len(current_state)), kick_size)
        remove_set = set(remove_indices)
        
        # Create new state without these blocks
        new_state = [state[i] for i in range(len(current_state)) if i not in remove_set]
        
        # Repair: Greedily add blocks to cover uncovered triples
        counts = compute_counts(new_state)
        uncovered = [i for i in range(num_triples) if counts[i] == 0]
        
        while uncovered and time.time() < deadline - 5:
            # Pick uncovered triple with fewest candidates
            target_tri = min(uncovered, key=lambda x: len(triple_blocks[x]))
            
            best_block = -1
            max_gain = -1
            
            candidates = triple_blocks[target_tri]
            for bi in candidates:
                gain = 0
                for ti in block_triples[bi]:
                    if counts[ti] == 0:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_block = bi
            
            if best_block == -1:
                break
                
            new_state.append(best_block)
            for ti in block_triples[best_block]:
                counts[ti] += 1
            uncovered.remove(target_tri)
            
        # Optimize
        new_state, counts = local_optimize(new_state, counts)
        
        if len(new_state) < best_len:
            best_len = len(new_state)
            best_state = new_state[:]
            current_state = new_state[:]
            current_counts = counts
            no_improve = 0
            # print(f"Improvement: {best_len}")
        else:
            no_improve += 1
            # Optionally, keep current_state or move to new_state?
            # ILS usually moves to the perturbed state to explore.
            current_state = new_state[:]
            current_counts = compute_counts(new_state)
            
    # Final Check
    if best_state is None:
        # Fallback to single greedy
        state = greedy_construct(0)
        counts = compute_counts(state)
        best_state, _ = local_optimize(state, counts)

    return [list(all_blocks[bi]) for bi in best_state]

if __name__ == "__main__":
    # Quick self-test
    blocks = construct(24, 7, 3)
    print(f"Blocks found: {len(blocks)}")
