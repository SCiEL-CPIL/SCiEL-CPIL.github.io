import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Helper to get triple indices for a block
    def get_block_triples(block):
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    # 2. Greedy Construction
    coverage_count = np.zeros(num_triples, dtype=np.int32)
    uncovered = set(range(num_triples))
    blocks = []
    
    while uncovered:
        if time.time() > deadline - 20: # Leave time for SA
            break
            
        # Pick an uncovered triple
        target_triple_idx = next(iter(uncovered))
        target_triple = all_triples[target_triple_idx]
        
        remaining_points = [p for p in points if p not in target_triple]
        best_block = None
        best_gain = -1
        
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target_triple + combo))
            gain = 0
            for b_triple in itertools.combinations(block, t):
                if triple_to_idx[b_triple] in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        for idx in get_block_triples(best_block):
            coverage_count[idx] += 1
            if coverage_count[idx] == 1:
                uncovered.discard(idx)
                
    # Convert to tuples for hashing
    block_tuples = [tuple(b) for b in blocks]
    
    # 3. Initial Local Search (Deletion and Swap)
    # This is similar to the current program's local search, 
    # but we will run it briefly to get a good start for SA.
    
    def get_unique_triples(block):
        return [t_idx for t_idx in get_block_triples(block) if coverage_count[t_idx] == 1]

    def can_delete(block_idx):
        b = block_tuples[block_idx]
        for t_idx in get_block_triples(b):
            if coverage_count[t_idx] == 1:
                return False
        return True

    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        
        # Pass 1: Deletion
        for i in range(len(block_tuples)):
            if time.time() > deadline - 30:
                break
            if can_delete(i):
                b = block_tuples[i]
                for t_idx in get_block_triples(b):
                    coverage_count[t_idx] -= 1
                del block_tuples[i]
                improved = True
                break
        
        if time.time() > deadline - 30:
            break
            
        if improved:
            continue
            
        # Pass 2: Exchange (1-swap)
        for i in range(len(block_tuples)):
            if time.time() > deadline - 30:
                break
                
            b = block_tuples[i]
            unique_triples = get_unique_triples(b)
            
            if not unique_triples:
                continue
                
            mandatory_points = set()
            for t_idx in unique_triples:
                mandatory_points.update(all_triples[t_idx])
                
            if len(mandatory_points) > k:
                continue
                
            available_points = [p for p in points if p not in mandatory_points]
            
            num_attempts = 10
            found_replacement = False
            for _ in range(num_attempts):
                if time.time() > deadline - 30:
                    break
                    
                if len(available_points) < k - len(mandatory_points):
                    break
                extra_points = random.sample(available_points, k - len(mandatory_points))
                new_block_tuple = tuple(sorted(list(mandatory_points) + extra_points))
                
                if new_block_tuple == b or new_block_tuple in block_tuples:
                    continue
                    
                new_block_triples = get_block_triples(new_block_tuple)
                
                # Check if swap improves or maintains validity
                # We need to check if removing b and adding new_b keeps all triples covered
                # Actually, the current logic in the seed program just does the swap and checks if it's valid?
                # The seed program's swap logic:
                # 1. Subtract b
                # 2. Add new_b
                # 3. If valid, keep. If not, revert?
                # The seed program code provided:
                # "Perform swap ... improved = True ... break"
                # It doesn't check validity after swap! It assumes the mandatory points logic is sufficient?
                # No, mandatory points ensures the *unique* triples of b are still covered by new_b.
                # But what about other triples?
                # If a triple was covered by b and some other block, it's still covered.
                # If a triple was covered ONLY by b, it's in unique_triples, so its points are in mandatory_points,
                # so it is in new_b. So it is still covered.
                # So the swap maintains validity.
                
                for t_idx in get_block_triples(b):
                    coverage_count[t_idx] -= 1
                block_tuples[i] = new_block_tuple
                for t_idx in new_block_triples:
                    coverage_count[t_idx] += 1
                    
                improved = True
                found_replacement = True
                break
                
            if found_replacement:
                break

    # 4. Simulated Annealing
    # State
    current_blocks = list(block_tuples)
    current_coverage = coverage_count.copy()
    current_num_blocks = len(current_blocks)
    
    # Calculate initial uncovered
    current_num_uncovered = np.sum(current_coverage == 0)
    
    beta = 10.0
    def get_energy(n_blocks, n_uncovered):
        return n_blocks + beta * n_uncovered
    
    current_energy = get_energy(current_num_blocks, current_num_uncovered)
    
    best_blocks = list(current_blocks)
    best_coverage = current_coverage.copy()
    best_num_blocks = current_num_blocks
    best_num_uncovered = current_num_uncovered
    best_energy = current_energy
    
    T = 1.0
    cooling = 0.9995
    
    # Precompute all possible blocks? No, generate random.
    # To speed up, we can precompute a list of random blocks to try?
    # Or just generate on the fly.
    
    while time.time() < deadline - 5:
        if T < 0.01:
            break
            
        # Choose move
        move_type = random.random()
        
        new_blocks = current_blocks
        new_coverage = current_coverage
        new_num_blocks = current_num_blocks
        new_num_uncovered = current_num_uncovered
        
        if move_type < 0.4 and current_num_blocks > 0:
            # Remove
            idx = random.randint(0, current_num_blocks - 1)
            b = current_blocks[idx]
            b_triples = get_block_triples(b)
            
            # Simulate removal
            temp_cov = current_coverage.copy()
            temp_uncovered = current_num_uncovered
            for t_idx in b_triples:
                old = temp_cov[t_idx]
                new = old - 1
                temp_cov[t_idx] = new
                if old == 1 and new == 0:
                    temp_uncovered += 1
            
            new_blocks = current_blocks[:idx] + current_blocks[idx+1:]
            new_coverage = temp_cov
            new_num_blocks = current_num_blocks - 1
            new_num_uncovered = temp_uncovered
            
        elif move_type < 0.7:
            # Add
            # Generate random block
            b = tuple(sorted(random.sample(points, k)))
            b_triples = get_block_triples(b)
            
            if b in current_blocks:
                continue # Skip duplicate
            
            temp_cov = current_coverage.copy()
            temp_uncovered = current_num_uncovered
            for t_idx in b_triples:
                old = temp_cov[t_idx]
                new = old + 1
                temp_cov[t_idx] = new
                if old == 0 and new > 0:
                    temp_uncovered -= 1
            
            new_blocks = current_blocks + [b]
            new_coverage = temp_cov
            new_num_blocks = current_num_blocks + 1
            new_num_uncovered = temp_uncovered
            
        else:
            # Swap
            if current_num_blocks == 0:
                continue
            idx = random.randint(0, current_num_blocks - 1)
            b_old = current_blocks[idx]
            b_new = tuple(sorted(random.sample(points, k)))
            
            if b_new == b_old or b_new in current_blocks:
                continue
            
            old_triples = get_block_triples(b_old)
            new_triples = get_block_triples(b_new)
            
            temp_cov = current_coverage.copy()
            temp_uncovered = current_num_uncovered
            
            # Remove old
            for t_idx in old_triples:
                old = temp_cov[t_idx]
                new = old - 1
                temp_cov[t_idx] = new
                if old == 1 and new == 0:
                    temp_uncovered += 1
            
            # Add new
            for t_idx in new_triples:
                old = temp_cov[t_idx]
                new = old + 1
                temp_cov[t_idx] = new
                if old == 0 and new > 0:
                    temp_uncovered -= 1
            
            new_blocks = current_blocks[:idx] + [b_new] + current_blocks[idx+1:]
            new_coverage = temp_cov
            new_num_blocks = current_num_blocks
            new_num_uncovered = temp_uncovered
            
        new_energy = get_energy(new_num_blocks, new_num_uncovered)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current_blocks = new_blocks
            current_coverage = new_coverage
            current_num_blocks = new_num_blocks
            current_num_uncovered = new_num_uncovered
            current_energy = new_energy
            
            # Update best if valid
            if current_num_uncovered == 0 and current_num_blocks < best_num_blocks:
                best_blocks = list(current_blocks)
                best_coverage = current_coverage.copy()
                best_num_blocks = current_num_blocks
                best_num_uncovered = 0
                best_energy = current_energy
            elif current_num_uncovered == 0 and current_num_blocks == best_num_blocks and current_energy < best_energy:
                # Tie break? Not needed if blocks count is same.
                pass

        T *= cooling

    # 5. Final Polish on best valid solution
    if best_num_uncovered == 0:
        block_tuples = best_blocks
        coverage_count = best_coverage
    else:
        # If SA didn't find a valid solution, use the initial one
        pass

    # Run local search again on the best solution
    # (Same as step 3)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for i in range(len(block_tuples)):
            if time.time() > deadline:
                break
            if can_delete(i):
                b = block_tuples[i]
                for t_idx in get_block_triples(b):
                    coverage_count[t_idx] -= 1
                del block_tuples[i]
                improved = True
                break
        
        if time.time() > deadline:
            break
            
        if improved:
            continue
            
        for i in range(len(block_tuples)):
            if time.time() > deadline:
                break
                
            b = block_tuples[i]
            unique_triples = get_unique_triples(b)
            
            if not unique_triples:
                continue
                
            mandatory_points = set()
            for t_idx in unique_triples:
                mandatory_points.update(all_triples[t_idx])
                
            if len(mandatory_points) > k:
                continue
                
            available_points = [p for p in points if p not in mandatory_points]
            
            num_attempts = 20
            found_replacement = False
            for _ in range(num_attempts):
                if time.time() > deadline:
                    break
                    
                if len(available_points) < k - len(mandatory_points):
                    break
                extra_points = random.sample(available_points, k - len(mandatory_points))
                new_block_tuple = tuple(sorted(list(mandatory_points) + extra_points))
                
                if new_block_tuple == b or new_block_tuple in block_tuples:
                    continue
                    
                new_block_triples = get_block_triples(new_block_tuple)
                
                for t_idx in get_block_triples(b):
                    coverage_count[t_idx] -= 1
                block_tuples[i] = new_block_tuple
                for t_idx in new_block_triples:
                    coverage_count[t_idx] += 1
                    
                improved = True
                found_replacement = True
                break
                
            if found_replacement:
                break

    return [list(b) for b in block_tuples]
