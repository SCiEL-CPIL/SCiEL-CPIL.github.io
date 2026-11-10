"""
Strategy: Greedy Set Cover with Randomized Restarts for C(19,8,5).

The current seed uses a "greedy point addition" heuristic: it picks the first
uncovered t-subset, then greedily adds points to that specific block to maximize
coverage. This is a local heuristic.

This new strategy treats the problem as a standard Set Cover instance:
1. Universe: All C(19,5) = 11628 possible 5-subsets.
2. Sets: All C(19,8) = 75582 possible 8-blocks. Each block "covers" C(8,5) = 56 subsets.
3. Algorithm: Iteratively select the block that covers the most *uncovered* subsets.
   - To handle the large search space and avoid getting stuck in a single local optimum,
     we perform multiple randomized restarts.
   - Randomization is introduced by adding small random noise to the coverage counts
     or by shuffling the order of blocks with equal coverage, or by randomly initializing
     a few blocks before starting the greedy process.
   - We also implement a "lazy" deletion or efficient tracking of uncovered counts.
     Given v=19, k=8, t=5, the number of subsets is small enough (11628) that we can
     maintain a counter for how many times each subset is covered by the current solution.
     However, calculating the gain of a new block requires checking how many of its
     56 subsets are currently uncovered (count == 0).
     
     A more efficient approach for the greedy step:
     - Maintain `uncovered_count` for each 5-subset? No, maintain `covered_by` count.
     - Actually, for greedy set cover, we need to know for a candidate block B,
       how many subsets S in B have `covered_count[S] == 0`.
     - Since there are ~75k blocks and ~11k subsets, and we need to do this ~300 times,
       naive calculation is 300 * 75k * 56 ≈ 1.2 billion operations. Too slow in Python.
       
     Optimization:
     We can precompute the list of 5-subsets for each 8-block.
     We maintain a dictionary or array `cover_count` of size C(19,5).
     At each step, we need to find the block maximizing `sum(1 for s in block_subsets if cover_count[s] == 0)`.
     
     To speed this up, we can use a "reverse" index:
     For each 5-subset s, list the blocks that contain s.
     When we add a block B, we update cover_count for its 56 subsets.
     For each subset s in B, if cover_count[s] becomes 1 (from 0), it means s is now covered.
     This doesn't directly help find the *next* best block quickly without scanning all blocks.
     
     Alternative: 
     Since v=19 is small, we can represent blocks as bitmasks or tuples.
     Let's try to optimize the inner loop.
     75,582 blocks * 56 subsets/block = ~4.2M checks per greedy step.
     If we do 300 steps, that's 1.2B checks. In Python, this is likely > 100s.
     
     We need a faster heuristic or approximation.
     
     Let's look at the "Greedy Point Addition" again. It's O(k * v * C(k,t-1)) per block.
     The Set Cover greedy is O(N_blocks * C(k,t)) per block.
     
     Maybe we can combine: Use Set Cover Greedy but with a reduced candidate set?
     Or use a different heuristic: 
     "Greedy with Lookahead" or "Best-Fit Decreasing"?
     
     Actually, the La Jolla repository uses sophisticated algorithms. 
     A simple improvement over the seed is to use **Randomized Greedy Point Addition**.
     The seed picks `min(uncovered)` which is deterministic and can be biased.
     Randomizing the choice of the initial uncovered subset and the tie-breaking for points
     allows us to explore different covering structures.
     
     Let's stick to a **Randomized Greedy Point Addition** with multiple restarts.
     It is much faster than full Set Cover Greedy because it only evaluates v=19 candidates per point addition, not 75k blocks.
     
     Algorithm:
     1. Time budget: 110s.
     2. Loop until time up:
        a. Run one instance of Randomized Greedy Point Addition.
        b. Keep the best result (fewest blocks).
        c. Randomization:
           - Shuffle the order of uncovered subsets? No, we must cover them all.
           - Instead, when picking the first uncovered subset, pick a random one from the uncovered set.
           - When picking the next point to add to the block, if multiple points have the same max gain, pick randomly.
           - Also, we can randomize the initial "seed" block slightly.
           
     This is materially different from the seed because the seed is deterministic and runs only once.
     This approach explores multiple local optima via randomization and restarts.
"""

import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets for fast membership checks? 
    # The seed uses a set of tuples. This is fine.
    
    best_blocks = None
    best_len = float('inf')
    
    # To allow restarts, we need the greedy function to be fast.
    # The seed's greedy is O(N_blocks * k * v * C(k,t-1)).
    # For C(19,8,5), N_blocks ~ 335.
    # 335 * 8 * 19 * C(8,4)=70 -> 335 * 8 * 19 * 70 ≈ 3.5M operations per run.
    # In Python, this might take ~1-2 seconds per run.
    # So we can do ~50-100 restarts in 110s.
    
    def greedy_run(seed_val):
        random.seed(seed_val)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        # Optimization: Instead of `min(uncovered)`, pick a random uncovered subset
        # to break symmetry and explore different structures.
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Find the point x that maximizes the number of new uncovered t-subsets covered
                best_gain = -1
                best_candidates = []
                
                # Precompute combinations of current block of size t-1
                # This is constant for the inner loop
                block_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check if adding x covers any uncovered subsets
                    # A subset s is covered by block+{x} if s is a subset of block+{x}
                    # and s contains x (otherwise it was already covered by block? No, block doesn't have x yet)
                    # Actually, a t-subset S is covered by B' = B ∪ {x} if S ⊆ B'.
                    # Since S has size t and B has size t-1 (max), S must contain x and S\{x} ⊆ B.
                    # So we check if any combination of t-1 from B, combined with x, is in uncovered.
                    
                    for rest in block_combos:
                        # rest is a tuple of t-1 elements from block
                        # The new t-subset is sorted(rest + (x,))
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                # If no gain, pick a random point (shouldn't happen if uncovered is not empty and k>t)
                if best_gain <= 0:
                    # Fallback: pick any point not in block
                    available = [x for x in range(v) if x not in block]
                    if available:
                        x = random.choice(available)
                    else:
                        break
                else:
                    x = random.choice(best_candidates)
                
                block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # Remove all t-subsets of this block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return blocks

    # Main loop: Multiple restarts
    run_id = 0
    while time.time() < deadline:
        # Use a different seed for each run
        seed_val = run_id * 1000 + 42
        blocks = greedy_run(seed_val)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Optional: If we hit the theoretical lower bound or a known good bound, we could stop,
            # but we don't know the exact optimal. We just minimize.
        
        run_id += 1
        # Small sleep to ensure we don't spin too fast if greedy is very fast?
        # No, greedy is the bottleneck.
        
        # Check time again
        if time.time() > deadline:
            break

    if best_blocks is None:
        # Fallback to deterministic if no run completed (unlikely)
        return greedy_run(42)
        
    return best_blocks

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# We must ensure the function signature matches.
# The prompt asks to keep function names and signatures.
# The seed had `construct(v, k, t)`.
# I have defined `construct` above.
# I need to wrap this in the EVOLVE-BLOCK markers as per the seed structure?
# The seed structure was:
# """Docstring"""
# # EVOLVE-BLOCK-START
# import ...
# def construct...
# # EVOLVE-BLOCK-END

# I will output the complete program with the markers.
# EVOLVE-BLOCK-END
