import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # We need to find a set of blocks covering all triples.
    # Strategy: 
    # 1. Use a known good seed (La Jolla 76 blocks for 24,7,3) if available or generate a fast greedy.
    # 2. Attempt to reduce the size by 1, 2, ... using Algorithm X (DLX) on the "uncovered triples" 
    #    or more effectively, use a SAT-like backtracking search for a covering of size N < current_best.
    #    
    # Since C(24,7,3) is open between 69 and 76, and 76 is the best known, 
    # we will start with a fast greedy to get a valid solution (likely ~78-80), 
    # then use a specialized local search / exact cover hybrid to try to beat it.
    
    # Fast Greedy to get a baseline
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute bitsets for triples covered by each block
    # v=24, k=7, t=3 -> C(7,3)=35 triples per block
    # Total triples C(24,3)=2024
    # We can represent covered triples as a set of indices or a bitset.
    # 2024 bits fits in a Python int.
    
    block_triples = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        block_triples[i] = mask

    def greedy_construct():
        uncovered = (1 << num_triples) - 1
        selected = []
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                if i in selected:
                    continue
                gain = (block_triples[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain <= 0:
                break
            selected.append(best_idx)
            uncovered &= ~block_triples[best_idx]
        return selected

    baseline_sol = greedy_construct()
    best_sol = baseline_sol
    best_size = len(best_sol)
    
    # Local Search: Swap blocks to reduce redundancy or allow removal
    # We want to minimize the number of blocks.
    # A block is redundant if all its triples are covered by other blocks.
    # We can also swap a block for another if the new block covers the unique triples
    # of the old block and allows for future removals.
    
    def local_search_minimize(sol, time_limit):
        sol = list(sol)
        sol_set = set(sol)
        
        # Count coverage
        counts = [0] * num_triples
        for idx in sol:
            mask = block_triples[idx]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                idx_tri = lsb.bit_length() - 1
                counts[idx_tri] += 1
                m ^= lsb
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Remove redundant blocks
            i = 0
            while i < len(sol):
                idx = sol[i]
                mask = block_triples[idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_tri = lsb.bit_length() - 1
                    if counts[idx_tri] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    sol.pop(i)
                    sol_set.discard(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_tri = lsb.bit_length() - 1
                        counts[idx_tri] -= 1
                        m ^= lsb
                    improved = True
                else:
                    i += 1
            
            if time.time() > time_limit:
                break
                
            # 2. Try swaps: Replace block A with block B if B covers all unique triples of A
            # and B is not in sol.
            for i, idx_a in enumerate(sol):
                if time.time() > time_limit:
                    break
                
                # Find unique triples of A
                unique_tris = []
                m = block_triples[idx_a]
                while m:
                    lsb = m & -m
                    idx_tri = lsb.bit_length() - 1
                    if counts[idx_tri] == 1:
                        unique_tris.append(idx_tri)
                    m ^= lsb
                
                if not unique_tris:
                    continue # Should have been removed, but maybe just added?
                
                # Find a candidate B that covers all unique_tris
                # To speed up, pick the most constrained unique triple (fewest blocks cover it)
                # Precompute triple_to_blocks? It's expensive for 2024 triples x 3460 blocks.
                # 2024 * 3460 ~ 7M operations. Doable once.
                
                # Let's do a simple scan for now, or precompute if time permits.
                # For v=24, k=7, num_blocks=3460.
                # We'll just scan.
                
                found_swap = False
                for idx_b in range(num_blocks):
                    if idx_b in sol_set:
                        continue
                    mask_b = block_triples[idx_b]
                    covers_all = True
                    for tri_idx in unique_tris:
                        if not (mask_b & (1 << tri_idx)):
                            covers_all = False
                            break
                    if covers_all:
                        # Perform swap
                        # Remove A
                        sol_set.discard(idx_a)
                        m = block_triples[idx_a]
                        while m:
                            lsb = m & -m
                            idx_tri = lsb.bit_length() - 1
                            counts[idx_tri] -= 1
                            m ^= lsb
                        
                        # Add B
                        sol[i] = idx_b
                        sol_set.add(idx_b)
                        m = block_triples[idx_b]
                        while m:
                            lsb = m & -m
                            idx_tri = lsb.bit_length() - 1
                            counts[idx_tri] += 1
                            m ^= lsb
                        
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    # Restart the removal/swap loop as counts changed significantly
                    break
            
        return sol

    # Run local search
    current_time_limit = deadline - 10
    optimized_sol = local_search_minimize(best_sol, current_time_limit)
    
    # Verify validity
    covered = 0
    for idx in optimized_sol:
        covered |= block_triples[idx]
    if covered == (1 << num_triples) - 1:
        if len(optimized_sol) < best_size:
            best_size = len(optimized_sol)
            best_sol = optimized_sol
    
    # If we have time, try a more aggressive multi-start or SA
    if time.time() < deadline - 20:
        # Simple SA on the solution
        def get_energy(blocks_set):
            covered = 0
            for idx in blocks_set:
                covered |= block_triples[idx]
            num_covered = covered.bit_count()
            if num_covered == num_triples:
                return len(blocks_set)
            else:
                return len(blocks_set) + (num_triples - num_covered) * 1000

        current_blocks = set(best_sol)
        current_energy = get_energy(current_blocks)
        best_sa_blocks = set(best_sol)
        best_sa_energy = current_energy
        
        T = 5.0
        cooling = 0.99
        rng = random.Random(123)
        
        while time.time() < deadline - 5:
            move_type = rng.choice(['remove', 'add', 'swap'])
            
            if move_type == 'remove' and current_blocks:
                to_remove = rng.choice(list(current_blocks))
                new_blocks = set(current_blocks)
                new_blocks.discard(to_remove)
            elif move_type == 'add':
                to_add = rng.choice(range(num_blocks))
                new_blocks = set(current_blocks)
                new_blocks.add(to_add)
            else: # swap
                if not current_blocks:
                    continue
                to_remove = rng.choice(list(current_blocks))
                to_add = rng.choice(range(num_blocks))
                new_blocks = set(current_blocks)
                new_blocks.discard(to_remove)
                new_blocks.add(to_add)
            
            new_energy = get_energy(new_blocks)
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < pow(2.718281828, -delta_e / T):
                current_blocks = new_blocks
                current_energy = new_energy
                if current_energy < best_sa_energy:
                    best_sa_energy = current_energy
                    best_sa_blocks = set(current_blocks)
            
            T *= cooling
            if T < 0.1:
                break
        
        if best_sa_energy < best_size:
            best_size = best_sa_energy
            best_sol = list(best_sa_blocks)

    # Return the best valid solution found
    # Re-verify one last time
    covered = 0
    for idx in best_sol:
        covered |= block_triples[idx]
    if covered == (1 << num_triples) - 1:
        return [all_blocks[i] for i in best_sol]
    else:
        # Fallback to baseline if something went wrong
        return [all_blocks[i] for i in baseline_sol]
