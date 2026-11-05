import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin for safety
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute triple indices for each block
    block_triple_indices = [[] for _ in range(num_blocks)]
    # triple_to_blocks: for each triple, list of blocks covering it
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    # Also precompute block_bitmasks or similar for fast intersection?
    # For v=24, k=7, t=3, num_blocks is C(24,7) = 346104.
    # num_triples is C(24,3) = 2024.
    # block_triple_indices[i] has C(7,3)=35 entries.
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris

    def grasp_construct(seed, alpha=0.5):
        """
        GRASP: Greedy Randomized Adaptive Search Procedure
        1. Construction phase: Select blocks using RCL based on gain/overlap ratio.
        2. Local search phase: Minimize blocks via 1-exchange (remove one, add one to cover lost triples).
        """
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=np.int32)
        selected = []
        selected_set = set()
        
        # For efficiency, maintain count of how many selected blocks cover each triple
        # But for GRASP construction, we just need to know if a triple is covered or not.
        # Actually, to calculate "overlap" (redundancy), we need to know how many
        # *other* selected blocks cover the triples in the candidate block.
        
        # Let's maintain a coverage count array
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        # Pre-compute block overlap metrics? No, dynamic.
        # For each candidate block, gain = sum(uncovered[tris])
        # Overlap = sum(cover_count[tris] > 0) ? No, that's not right.
        # "Cost" in the ratio: We want to penalize blocks that cover triples already covered.
        # Ratio = Gain / (1 + Overlap_Count)
        # Where Gain = new triples covered.
        # Overlap_Count = number of triples in this block that are ALREADY covered.
        
        while np.any(uncovered) and time.time() < deadline - 10:
            # Calculate gain and overlap for all candidates
            # This is O(num_blocks * 35) per step. 346k * 35 ~ 12M ops.
            # 115s / 12M ~ 9ms per step. We can do ~10k steps? No, we need ~76 blocks.
            # 76 steps * 12M = 912M ops. Too slow in Python loops.
            
            # Optimization: Vectorize gain calculation?
            # block_triple_indices is a list of lists.
            # Let's try a faster loop.
            
            best_gain = -1
            best_ratio = -1
            candidates = []
            
            # We can skip blocks that are already selected
            # And we can limit candidates to those with gain > 0
            
            # To speed up, we can pre-filter?
            # For now, iterate.
            
            # Optimization: Use NumPy for gain calculation if possible?
            # block_triple_indices is irregular.
            
            # Let's try to optimize the inner loop.
            # We can break early if we find a perfect gain? No.
            
            # Alternative: Maintain a list of "active" blocks that cover at least one uncovered triple.
            # When we select a block, we update the list.
            
            # For simplicity and correctness first, let's do the full scan but optimized.
            
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                
                tris = block_triple_indices[i]
                # Calculate gain
                # gain = sum(uncovered[j] for j in tris)
                # overlap = sum(cover_count[j] > 0 for j in tris)
                
                gain = 0
                overlap = 0
                for j in tris:
                    if uncovered[j]:
                        gain += 1
                    else:
                        # It's covered, so it's overlap
                        overlap += 1
                
                if gain == 0:
                    continue
                
                # Ratio: We want high gain, low overlap.
                # Score = gain / (1 + overlap)
                # Or Score = gain - overlap * lambda
                # Let's use Ratio as suggested in the prompt.
                score = gain / (1.0 + overlap)
                
                if score > best_ratio:
                    best_ratio = score
                    candidates = [i]
                elif score == best_ratio:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # RCL: Select from top candidates?
            # Standard GRASP: RCL contains elements with score >= (1-alpha)*best + alpha*worst
            # Here we just have the best score.
            # Let's pick randomly from the top 10% or top 5 candidates to introduce randomness.
            # Or just pick randomly from all candidates with max score?
            # The prompt says: "selects blocks maximizing the ratio... Implement this with a random..."
            # So we select from the set of blocks that maximize the ratio.
            
            chosen = rng.choice(candidates)
            selected.append(chosen)
            selected_set.add(chosen)
            
            for j in block_triple_indices[chosen]:
                uncovered[j] = 0
                cover_count[j] += 1
        
        # Local Search: Minimize blocks
        sol = local_search_minimize(selected, deadline - 10)
        return sol

    def local_search_minimize(selected_blocks, time_limit):
        """
        CSP-style local search.
        1. Remove redundant blocks.
        2. 1-Exchange: Try removing block A and adding block B such that coverage is maintained.
           This effectively swaps A for B. If B covers the unique triples of A, and A is removed,
           size stays same. But if we can find a sequence of swaps that reduces size?
           No, 1-exchange keeps size same.
           To reduce size, we need to find a block that is "more redundant" or remove one and
           cover its unique triples with *existing* blocks? That's impossible if it has unique triples.
           
           Wait, the current local_search only removes redundant blocks.
           To get better than greedy, we need to find a different set of blocks of smaller size.
           
           Strategy for Local Search:
           1. Identify blocks with "high cost" (many unique triples? No, we want to remove blocks with FEW unique triples).
           2. If we remove a block, some triples become uncovered.
           3. We must add a new block that covers ALL those uncovered triples.
           4. If we can do this, the size remains the same.
           5. How to reduce size?
              - Maybe we can remove a block and cover its unique triples with *multiple* other blocks?
                No, that increases size.
              - The only way to reduce size is if the current solution has redundancy that can be exploited.
              - Actually, greedy solutions often have blocks that are "almost" redundant.
              - A better local search:
                - Try to remove a block.
                - If it creates uncovered triples, try to find a single block that covers them.
                - If found, swap. This explores the same-size manifold.
                - Occasionally, if we find a swap that allows a subsequent removal?
                
           Let's stick to:
           1. Remove redundant blocks (size decreases).
           2. Swap: Remove A, Add B (covers A's unique triples). Size same.
              - Why do this? To change the composition of the solution, potentially making
                another block redundant in the next step.
        """
        sol = list(selected_blocks)
        sol_set = set(sol)
        
        # counts: number of blocks covering each triple
        counts = np.zeros(num_triples, dtype=np.int32)
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Pass 1: Remove redundant blocks
            i = 0
            while i < len(sol):
                idx = sol[i]
                is_redundant = True
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    sol.pop(i)
                    sol_set.discard(idx)
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    improved = True
                else:
                    i += 1
            
            # Pass 2: Swaps (1-exchange)
            # Try to swap a block A with a block B.
            # A is removed. Uncovered triples U = {t in A | count[t] == 1}.
            # B must cover all t in U.
            # If such B exists, swap A -> B.
            # This allows us to "move" the coverage responsibility.
            
            if time.time() > time_limit - 2:
                break
                
            # Shuffle order to avoid bias
            rng = random.Random(int(time.time() * 1000))
            indices = list(range(len(sol)))
            rng.shuffle(indices)
            
            for pos in indices:
                if time.time() > time_limit - 1:
                    break
                    
                idx_a = sol[pos]
                # Find unique triples in A
                unique_tris = []
                for tri_idx in block_triple_indices[idx_a]:
                    if counts[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    # Redundant, should have been caught in Pass 1, but maybe counts changed?
                    # Actually if unique_tris is empty, it's redundant.
                    # Remove it.
                    sol.pop(pos)
                    sol_set.discard(idx_a)
                    for tri_idx in block_triple_indices[idx_a]:
                        counts[tri_idx] -= 1
                    improved = True
                    break # Restart scan as list changed
                else:
                    # Find a block B that covers all unique_tris
                    # B must not be in sol_set
                    # B must cover all t in unique_tris
                    
                    # Heuristic: Look for B that covers unique_tris and has high "efficiency"
                    # For now, just find ANY B.
                    found_b = -1
                    # To speed up, intersect the lists of blocks covering each unique triple
                    # candidates = intersection of triple_to_blocks[t] for t in unique_tris
                    # Then filter out those in sol_set
                    
                    # Since unique_tris is small (<= 35), we can do this.
                    # Start with the smallest list
                    min_list_len = float('inf')
                    min_list_idx = -1
                    for t in unique_tris:
                        if len(triple_to_blocks[t]) < min_list_len:
                            min_list_len = len(triple_to_blocks[t])
                            min_list_idx = t
                            
                    candidate_blocks = set(triple_to_blocks[min_list_idx])
                    
                    for t in unique_tris:
                        if t == min_list_idx:
                            continue
                        candidate_blocks.intersection_update(triple_to_blocks[t])
                    
                    # Filter out selected
                    for b in candidate_blocks:
                        if b not in sol_set:
                            found_b = b
                            break
                    
                    if found_b != -1:
                        # Perform swap
                        sol[pos] = found_b
                        sol_set.discard(idx_a)
                        sol_set.add(found_b)
                        
                        # Update counts
                        for tri_idx in block_triple_indices[idx_a]:
                            counts[tri_idx] -= 1
                        for tri_idx in block_triple_indices[found_b]:
                            counts[tri_idx] += 1
                        
                        improved = True
                        break # Restart scan as solution changed
            
        return sol

    best_sol = None
    best_size = num_blocks
    
    # Run GRASP with multiple restarts
    # The construction is fast-ish, local search is the bottleneck.
    # We need to balance restarts vs local search depth.
    
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Allocate time for local search
        # If we have 100s left, and 10 restarts, ~10s each?
        # Let's give 5s to construction, 4s to local search per restart.
        ls_time_limit = deadline - 10 - (i * 0) # Dynamic?
        # Better: Fixed budget per iteration.
        iter_budget = 8.0
        ls_deadline = time.time() + iter_budget
        
        g_sol = grasp_construct(1000 + i)
        
        # Ensure it's valid? GRASP should produce valid if it finishes.
        # But if it times out during construction, it might be invalid.
        # Check validity
        covered = set()
        for idx in g_sol:
            for tri_idx in block_triple_indices[idx]:
                covered.add(tri_idx)
        if len(covered) < num_triples:
            # Fallback: Greedy to finish
            # Or just skip this restart
            continue
            
        ls_sol = local_search_minimize(g_sol, ls_deadline)
        
        # Verify LS solution
        covered = set()
        for idx in ls_sol:
            for tri_idx in block_triple_indices[idx]:
                covered.add(tri_idx)
        if len(covered) == num_triples:
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_sol = ls_sol
                print(f"Restart {i}: New best {best_size}", file=sys.stderr)

    if best_sol is None:
        # Fallback: Simple Greedy
        print("Fallback to simple greedy", file=sys.stderr)
        uncovered = np.ones(num_triples, dtype=np.int32)
        selected = []
        while np.any(uncovered) and time.time() < deadline - 5:
            best_gain = -1
            best_block = -1
            for i in range(num_blocks):
                if i in selected:
                    continue
                gain = 0
                for j in block_triple_indices[i]:
                    if uncovered[j]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            if best_gain <= 0:
                break
            selected.append(best_block)
            for j in block_triple_indices[best_block]:
                uncovered[j] = 0
        
        best_sol = local_search_minimize(selected, deadline - 2)
        
    return [all_blocks[i] for i in best_sol]

if __name__ == "__main__":
    # Quick self-test
    blocks = construct(24, 7, 3)
    print(f"Found covering with {len(blocks)} blocks")
    # Verify
    covered = set()
    for b in blocks:
        for tri in itertools.combinations(b, 3):
            covered.add(tri)
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered {len(covered)}/{total} triples")
