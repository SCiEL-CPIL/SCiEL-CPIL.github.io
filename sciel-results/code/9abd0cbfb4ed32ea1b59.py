import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Map triplet id to the triplet itself
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute for each point, which triplets contain it
    point_to_triplets = defaultdict(list)
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_to_triplets[p].append(tid)
    
    # Precompute all possible blocks and their triplet masks
    # C(24,7) = 346104, which is manageable
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the set of triplet ids it covers
    # Using a list of sets for memory efficiency
    block_triplet_sets = []
    for block in all_blocks:
        tids = set()
        for triplet in itertools.combinations(block, t):
            tids.add(triplet_to_id[triplet])
        block_triplet_sets.append(tids)
    
    # Precompute for each triplet, which blocks cover it
    triplet_to_blocks = defaultdict(list)
    for bid, block in enumerate(all_blocks):
        for tid in block_triplet_sets[bid]:
            triplet_to_blocks[tid].append(bid)
    
    # Helper to get a random uncovered triplet
    def get_random_uncovered(uncovered_set):
        if not uncovered_set:
            return None
        # Use random.choice on the set
        return random.choice(list(uncovered_set))
    
    # Greedy construction with randomization
    def greedy_construct(seed, randomization_factor=0.3):
        random.seed(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet
            target_tid = random.choice(list(uncovered))
            
            # Find blocks that cover this triplet
            candidate_blocks = triplet_to_blocks[target_tid]
            
            if not candidate_blocks:
                # Should not happen in a valid covering problem
                break
            
            # Score each candidate block by how many new triplets it covers
            # with some randomization
            best_block = None
            best_score = -1
            
            # Limit candidates to top ones for speed if too many
            if len(candidate_blocks) > 200:
                # Random sample of candidates
                candidates = random.sample(candidate_blocks, 200)
            else:
                candidates = candidate_blocks
            
            for bid in candidates:
                block = all_blocks[bid]
                new_covered = block_triplet_sets[bid] & uncovered
                gain = len(new_covered)
                
                # Add randomization
                score = gain * (1.0 + randomization_factor * random.random())
                
                if score > best_score:
                    best_score = score
                    best_block = bid
            
            if best_block is None:
                # Fallback to first candidate
                best_block = candidate_blocks[0]
            
            blocks.append(best_block)
            uncovered -= block_triplet_sets[best_block]
            
        return blocks
    
    # Simulated Annealing optimization
    def simulated_annealing(blocks, time_limit):
        # Convert to list of block indices for efficiency
        block_indices = blocks[:]
        n = len(block_indices)
        
        # Maintain coverage counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        for bid in block_indices:
            for tid in block_triplet_sets[bid]:
                counts[tid] += 1
        
        current_cost = n
        best_blocks = block_indices[:]
        best_cost = n
        
        # Temperature schedule
        T0 = 1.0
        T_end = 0.01
        cooling = 0.995
        
        T = T0
        no_improve = 0
        
        while time.time() < time_limit and T > T_end:
            # Choose a move: add, remove, or swap a block
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a block
                if n <= 1:
                    continue
                i = random.randint(0, n - 1)
                old_bid = block_indices[i]
                
                # Check if removal is valid
                valid = True
                for tid in block_triplet_sets[old_bid]:
                    if counts[tid] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Apply removal
                    for tid in block_triplet_sets[old_bid]:
                        counts[tid] -= 1
                    block_indices.pop(i)
                    n -= 1
                    new_cost = n
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_blocks = block_indices[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        block_indices.insert(i, old_bid)
                        for tid in block_triplet_sets[old_bid]:
                            counts[tid] += 1
                        n += 1
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type < 0.6:
                # Add a block
                # Pick a random uncovered triplet if any, else random
                uncovered_tids = [tid for tid in range(n_triplets) if counts[tid] == 0]
                if uncovered_tids:
                    target_tid = random.choice(uncovered_tids)
                else:
                    target_tid = random.randint(0, n_triplets - 1)
                
                candidates = triplet_to_blocks[target_tid]
                if not candidates:
                    continue
                new_bid = random.choice(candidates)
                
                # Check if already present
                if new_bid in block_indices:
                    no_improve += 1
                    continue
                
                # Apply addition
                for tid in block_triplet_sets[new_bid]:
                    counts[tid] += 1
                block_indices.append(new_bid)
                n += 1
                new_cost = n
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = block_indices[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    block_indices.pop()
                    for tid in block_triplet_sets[new_bid]:
                        counts[tid] -= 1
                    n -= 1
                    no_improve += 1
                    
            else:
                # Swap a block
                if n == 0:
                    continue
                i = random.randint(0, n - 1)
                old_bid = block_indices[i]
                
                # Find a replacement block
                # Pick a random triplet from the old block
                old_tids = list(block_triplet_sets[old_bid])
                if not old_tids:
                    continue
                target_tid = random.choice(old_tids)
                
                candidates = [bid for bid in triplet_to_blocks[target_tid] if bid != old_bid]
                if not candidates:
                    no_improve += 1
                    continue
                
                new_bid = random.choice(candidates)
                
                # Check coverage: after swap, all triplets must still be covered
                # Triplets only in old_bid and not in new_bid must have count > 1
                valid = True
                for tid in block_triplet_sets[old_bid]:
                    if tid not in block_triplet_sets[new_bid]:
                        if counts[tid] <= 1:
                            valid = False
                            break
                
                if valid:
                    # Apply swap
                    for tid in block_triplet_sets[old_bid]:
                        counts[tid] -= 1
                    for tid in block_triplet_sets[new_bid]:
                        counts[tid] += 1
                    block_indices[i] = new_bid
                    new_cost = n  # Same number of blocks
                    delta_cost = 0
                    
                    # Accept with some probability to allow exploration
                    if random.random() < 0.5:  # Higher acceptance for neutral moves
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_blocks = block_indices[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        for tid in block_triplet_sets[new_bid]:
                            counts[tid] -= 1
                        for tid in block_triplet_sets[old_bid]:
                            counts[tid] += 1
                        block_indices[i] = old_bid
                        no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling
            
            if no_improve > 1000:
                break
        
        return best_blocks
    
    # Local search: try to remove redundant blocks
    def remove_redundant(blocks):
        block_indices = blocks[:]
        n = len(block_indices)
        
        # Count coverage
        counts = np.zeros(n_triplets, dtype=np.int32)
        for bid in block_indices:
            for tid in block_triplet_sets[bid]:
                counts[tid] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(block_indices)):
                bid = block_indices[i]
                redundant = True
                for tid in block_triplet_sets[bid]:
                    if counts[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    for tid in block_triplet_sets[bid]:
                        counts[tid] -= 1
                    block_indices.pop(i)
                    changed = True
                    break
        
        return block_indices
    
    # Main search
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy constructions with different seeds
    n_greedy = 8
    time_per_greedy = 5.0
    
    for seed in range(n_greedy):
        if time.time() > deadline - 30:
            break
        
        randomization = 0.2 + 0.1 * seed
        blocks = greedy_construct(seed, randomization)
        
        # Remove redundant
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Phase 2: Simulated Annealing on best solution
    if best_blocks is not None and time.time() < deadline - 10:
        sa_blocks = simulated_annealing(best_blocks, deadline - 5)
        
        # Verify
        counts = np.zeros(n_triplets, dtype=np.int32)
        for bid in sa_blocks:
            for tid in block_triplet_sets[bid]:
                counts[tid] += 1
        
        if np.min(counts) > 0:
            if len(sa_blocks) < best_len:
                best_len = len(sa_blocks)
                best_blocks = sa_blocks[:]
    
    # Phase 3: More SA restarts if time permits
    if time.time() < deadline - 15:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            perturbed = best_blocks[:]
            for _ in range(3):
                if len(perturbed) > 1:
                    i = random.randint(0, len(perturbed) - 1)
                    old_bid = perturbed[i]
                    old_tids = list(block_triplet_sets[old_bid])
                    if old_tids:
                        target_tid = random.choice(old_tids)
                        candidates = [bid for bid in triplet_to_blocks[target_tid] if bid != old_bid]
                        if candidates:
                            perturbed[i] = random.choice(candidates)
            
            sa_blocks = simulated_annealing(perturbed, deadline - 5)
            
            # Verify
            counts = np.zeros(n_triplets, dtype=np.int32)
            for bid in sa_blocks:
                for tid in block_triplet_sets[bid]:
                    counts[tid] += 1
            
            if np.min(counts) > 0:
                if len(sa_blocks) < best_len:
                    best_len = len(sa_blocks)
                    best_blocks = sa_blocks[:]
    
    # Final verification
    if best_blocks is None:
        best_blocks = greedy_construct(0)
    
    # Convert block indices back to blocks
    result = [list(all_blocks[bid]) for bid in best_blocks]
    
    return result

if __name__ == "__main__":
    pass
