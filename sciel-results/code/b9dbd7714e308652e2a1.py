import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy or tetracode extension
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage masks and neighbor lists
    # masks[i] is a bitset of words covered by word i (dist <= 1)
    # neighbors[i] is a list of indices covered by word i
    masks = [0] * N
    neighbors = [[] for _ in range(N)]
    
    for i in range(N):
        w_i = words[i]
        mask_val = 0
        neigh = []
        # Self
        mask_val |= (1 << i)
        neigh.append(i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                mask_val |= (1 << j)
                neigh.append(j)
        masks[i] = mask_val
        neighbors[i] = neigh

    full_mask = (1 << N) - 1

    def greedy_construct(seed_idx=None):
        """
        Greedy set cover algorithm.
        Returns a set of indices.
        """
        current_cover = 0
        code = set()
        # We want to pick words that maximize new coverage.
        # To introduce randomness/different paths, we can shuffle the order of evaluation or use a heap.
        # For speed, we iterate and pick the best.
        
        # Optimization: Keep track of uncovered count for each candidate?
        # N=729, so O(N^2) per greedy step is too slow if we do it naively.
        # But greedy steps are ~80. 80 * 729 * (bit ops) is fine.
        
        # We can use a priority queue or just scan.
        # Scanning is O(N) per step. 80 steps * 729 = ~60k ops. Very fast.
        
        while current_cover != full_mask:
            best_idx = -1
            max_new = -1
            # Randomize tie-breaking to get different greedy solutions
            # We can just pick the first max, or random among max.
            # Let's pick the one with max new coverage.
            
            # To speed up, we can precompute the "new coverage" of each word given current_cover?
            # new_cov(i) = masks[i] & ~current_cover
            # bit_count of that.
            
            # Since N is small, we just loop.
            for i in range(N):
                if i in code:
                    continue
                new_bits = masks[i] & ~current_cover
                cnt = new_bits.bit_count()
                if cnt > max_new:
                    max_new = cnt
                    best_idx = i
                    # Early exit if we cover everything?
                    if max_new == 0:
                        break # Should not happen if current_cover != full_mask and we haven't picked all
            
            if best_idx == -1:
                # Should not happen
                break
                
            code.add(best_idx)
            current_cover |= masks[best_idx]
            
        return code

    def local_refinement(code_set):
        """
        1. Remove redundant nodes.
        2. Try 1-swaps (remove u, add v) to maintain coverage.
        3. Repeat until no improvement.
        """
        code = set(code_set)
        
        # Compute coverage counts for fast redundancy check
        # counts[j] = number of codewords covering j
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            for j in neighbors[idx]:
                counts[j] += 1
        
        changed = True
        while changed:
            changed = False
            
            # Phase 1: Remove redundant
            to_remove = []
            for idx in code:
                # Check if any neighbor has count 1
                is_redundant = True
                for j in neighbors[idx]:
                    if counts[j] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(idx)
            
            for idx in to_remove:
                for j in neighbors[idx]:
                    counts[j] -= 1
                code.remove(idx)
                changed = True
            
            # Phase 2: 1-Swaps
            # Try to replace u with v such that coverage is maintained.
            # If we remove u, some words might become uncovered (count 0).
            # We need v to cover all those words.
            # Also, we want to eventually reduce size, so swaps alone don't reduce size.
            # But swaps can enable future removals if v covers "more" effectively or if we swap to a position that allows removing another node later.
            # Actually, standard 1-swap for MinDominatingSet:
            # If (C \ {u}) U {v} is a valid cover, we can swap. This doesn't change size.
            # To reduce size, we rely on Phase 1.
            # So we should accept swaps that maintain validity, hoping to reach a state where Phase 1 can remove more.
            
            # To avoid infinite loops, we can use a "tabu" or just limit iterations.
            # Or, we can prioritize swaps that increase the number of "removable" nodes.
            
            # Let's just do a simple sweep of swaps.
            # For each u in C, for each v not in C:
            #   if (C \ {u}) U {v} is valid cover, accept.
            
            # Optimization: 
            # If we remove u, the words that become uncovered are those j in neighbors[u] where counts[j]==1.
            # Let U_u = { j in neighbors[u] | counts[j] == 1 }
            # If U_u is empty, u is redundant (handled in Phase 1).
            # If U_u is not empty, we need v to cover all j in U_u.
            # i.e., U_u subset of neighbors[v].
            
            # We can precompute for each u the set U_u.
            # Then for each v, check if it covers U_u.
            
            # Since N=729, |C|~80.
            # For each u (80), compute U_u (small list).
            # For each v (729), check if it covers U_u.
            # Total ops ~ 80 * 729 * |U_u|. |U_u| is small (usually < 10).
            # This is ~400k ops per sweep. Very fast.
            
            # Let's implement this specific swap logic.
            
            # First, identify candidates for removal (u where U_u is non-empty)
            # Actually, we just try all u.
            
            swaps_made = False
            # We need to be careful not to undo previous swaps in the same sweep immediately?
            # Simple approach: iterate u, find best v, apply swap, restart sweep?
            # Or apply all non-conflicting swaps?
            # Let's just do one swap per pass if found, to be safe.
            
            found_swap = False
            for u in list(code):
                # Calculate U_u
                U_u = []
                for j in neighbors[u]:
                    if counts[j] == 1:
                        U_u.append(j)
                
                if not U_u:
                    # Redundant, should have been removed in Phase 1
                    # But maybe counts changed?
                    # Just skip.
                    continue
                
                # We need a v not in code that covers all j in U_u.
                # Also, v should not be u.
                
                # Optimization: Precompute which v covers a specific set?
                # Just loop v.
                for v in range(N):
                    if v in code:
                        continue
                    # Check if v covers all j in U_u
                    # We can check if (masks[v] & mask_of_U_u) == mask_of_U_u
                    # But U_u is a list.
                    # Let's just check intersection.
                    # Since U_u is small, loop is fine.
                    covers_all = True
                    for j in U_u:
                        if not (masks[v] >> j) & 1:
                            covers_all = False
                            break
                    if covers_all:
                        # Valid swap
                        # Apply it
                        # Remove u
                        for j in neighbors[u]:
                            counts[j] -= 1
                        code.remove(u)
                        # Add v
                        for j in neighbors[v]:
                            counts[j] += 1
                        code.add(v)
                        
                        found_swap = True
                        changed = True
                        break # Restart u loop to re-evaluate counts
                
                if found_swap:
                    break # Restart entire while loop
            
            # If no swap found and no removal, we are stuck.
            
        return code

    # Main Loop
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_code = None
    best_size = float('inf')
    
    # Initial Tetracode for comparison
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            w = tuple(h + list(tail))
            init_indices.add(word_to_idx[w])
    
    # Try initial tetracode refinement
    refined_init = local_refinement(init_indices)
    best_code = refined_init
    best_size = len(best_code)
    
    # Iterated Local Search
    # We run greedy construction multiple times with different random seeds/orderings
    # Greedy is deterministic unless we add randomness.
    # We can randomize the tie-breaking in greedy.
    
    for i in range(1000): # Max iterations, will break on time
        if time.time() > deadline - 5.0:
            break
        
        # Randomize greedy by shuffling the order of candidates?
        # Or just add a small random noise to the score?
        # Let's implement a randomized greedy:
        # Instead of picking the single best, pick from top K?
        # Or just shuffle the list of indices before scanning?
        
        # Let's use a simple randomized greedy:
        # At each step, pick the word that maximizes (new_coverage + random_noise)
        
        current_cover = 0
        code = set()
        
        # Pre-shuffled indices for this run
        indices = list(range(N))
        random.shuffle(indices)
        
        while current_cover != full_mask:
            best_idx = -1
            max_score = -1
            # We scan all, but use shuffled order to break ties randomly?
            # Actually, scanning all and taking max is deterministic.
            # To make it randomized, we can add a small random value to the count.
            
            # But adding random noise might degrade the greedy quality.
            # Alternative: Run standard greedy, but start from a random subset?
            # Or just run standard greedy multiple times? It's the same every time.
            
            # Let's modify greedy to be slightly randomized:
            # Pick from the top 10 candidates uniformly at random?
            
            # For simplicity and speed, let's just run the standard greedy 
            # but we can vary the initial "seed" by starting with a random word?
            # No, that might be worse.
            
            # Let's stick to deterministic greedy but vary the LOCAL SEARCH.
            # Or, better: Use the greedy result as a starting point for a 
            # "Randomized Local Search" where we sometimes accept worse moves?
            
            # Actually, the most effective ILS for Dominating Set is:
            # 1. Greedy
            # 2. Local Search (Refine)
            # 3. Perturb (Remove k random nodes, Re-greedy to cover gaps)
            # 4. Local Search
            
            # Let's implement Perturb.
            
            if i == 0:
                # First run: Standard Greedy
                code = greedy_construct()
            else:
                # Perturb best_code
                code = set(best_code)
                # Remove k random nodes
                k = random.randint(2, 5)
                to_remove = random.sample(list(code), min(k, len(code)))
                for idx in to_remove:
                    code.remove(idx)
                
                # Re-greedy to cover the gaps
                # Calculate current cover
                current_cover = 0
                for idx in code:
                    current_cover |= masks[idx]
                
                # Greedy fill
                while current_cover != full_mask:
                    best_idx = -1
                    max_new = -1
                    for idx in range(N):
                        if idx in code:
                            continue
                        new_bits = masks[idx] & ~current_cover
                        cnt = new_bits.bit_count()
                        if cnt > max_new:
                            max_new = cnt
                            best_idx = idx
                    if best_idx == -1:
                        break
                    code.add(best_idx)
                    current_cover |= masks[best_idx]
                
                # Refine
                code = local_refinement(code)
        
        # Refine the greedy result (if i==0, we didn't refine inside the loop above? 
        # Wait, in the i==0 case, I called greedy_construct but not local_refinement.
        # Let's ensure we always refine.
        
        # Actually, the loop above for i>0 includes refinement.
        # For i==0, let's refine now.
        if i == 0:
            code = local_refinement(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = set(code)
            
            # If we hit 73, we can try to hold on, but maybe 72 is possible?
            # Lower bound is 71. 72 is unknown.
            # If we hit 73, we continue to see if we can beat it.
            if best_size <= 72:
                # Unlikely, but if we hit it, we can break early or continue.
                # Let's continue to be safe.
                pass

    # Final check
    # Ensure best_code is valid
    final_cov = 0
    for idx in best_code:
        final_cov |= masks[idx]
    
    if final_cov != full_mask:
        # Fallback to tetracode
        best_code = init_indices
    
    return [list(words[i]) for i in best_code]
