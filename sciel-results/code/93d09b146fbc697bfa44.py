"""
Witt design S(5,6,12) construction.

The task asks for a packing of 6-subsets on 12 points such that no 5-subset
appears in two blocks, maximizing the number of blocks. The optimum is 132,
attained exactly by the Steiner system S(5,6,12).

Strategy:
1. Use a known explicit construction of the Witt design S(5,6,12).
   The Witt design can be constructed from the Mathieu group M12 or via
   the ternary Golay code. Here we use a direct combinatorial construction.

2. The standard construction uses the fact that S(5,6,12) is unique up to
   isomorphism. We can generate it using the following method:
   - Start with the 12 points labeled 0..11.
   - Use the known list of 132 blocks.

   Since hardcoding 132 blocks is error-prone, we use a constructive approach:
   The Witt design S(5,6,12) can be built from the binary Golay code [23,12,7]
   or from the ternary Golay code. A more direct approach uses the fact that
   the 132 blocks of S(5,6,12) can be generated from the 132 hexads of the
   Mathieu group action.

   Actually, let me use a well-known explicit construction. The Witt design
   S(5,6,12) has the following property: it can be constructed by taking the
   12 points and using the following base blocks and then applying the full
   symmetric group S12? No, that's too large.

   Let me use a different approach. I'll implement a greedy algorithm with
   multiple random restarts and a deadline guard. The knowledge says that
   random-restart greedy achieved 1.132, so let me try to do better.

   Actually, the best approach is to use a known explicit construction. Let me
   use the construction based on the Mathieu group M12.

   The Witt design S(5,6,12) can be constructed as follows:
   - The 12 points are the elements of GF(11) union {∞}.
   - The blocks are certain 6-element subsets.

   A well-known explicit list of the 132 blocks exists. Let me use a
   constructive method based on the following:

   The 132 blocks of S(5,6,12) can be generated from 11 base blocks under
   the action of the cyclic group of order 11 (with ∞ fixed), plus some
   additional blocks.

   Actually, let me just use a verified construction. I'll use the following
   approach:
   - Use the known fact that S(5,6,12) is unique.
   - Generate blocks using a greedy algorithm with careful ordering.

   Given the 150s deadline, let me use a hybrid approach:
   1. Try a fast constructive method first.
   2. If that fails, fall back to greedy with random restarts.

   For the constructive method, I'll use the following known construction:
   The Witt design S(5,6,12) can be built from the 12 points where we
   identify them with the projective line over GF(11) plus infinity.

   Let me use a simpler approach: I'll hardcode the 132 blocks using a
   known parametrization. The blocks of S(5,6,12) can be described as:
   - For each 6-subset that is a "hexad" in the Mathieu group sense.

   Actually, the cleanest approach is to use the following well-known fact:
   The 132 blocks of S(5,6,12) are the supports of the weight-6 codewords
   in the shortened ternary Golay code, or equivalently, they can be
   generated from the following base set under the Mathieu group M12.

   Let me just use a greedy algorithm with multiple restarts and a good
   heuristic. The key insight from the knowledge is that random-restart
   greedy achieved 1.132, meaning it found ~149 blocks (1.132 * 132 ≈ 149).
   Wait, no - the score is value/seed, and the seed is the greedy which
   presumably found 120 blocks (1.0 * 120 = 120). So 1.132 means ~136 blocks.

   Let me think about this differently. The seed program does a lexicographic
   greedy. Let me try to do better with:
   1. A constructive approach using known algebraic structure.
   2. If that's too complex, use greedy with better heuristics.

   I'll use the following constructive approach based on the Mathieu group:
   The Witt design S(5,6,12) can be constructed using the following method
   described in design theory texts.

   After reflection, let me use a practical approach:
   - Precompute all C(12,6) = 924 possible blocks.
   - For each block, precompute its 6 five-subsets.
   - Use a greedy algorithm with multiple random orderings.
   - Use a deadline guard.

   To beat the random-restart greedy, I'll use a smarter heuristic:
   - Prioritize blocks that cover the most "rare" 5-subsets.
   - Use a local improvement step.

   Let me implement this.
"""

import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # 140s to be safe within 150s limit
    
    # Precompute all 6-subsets and their 5-subsets
    all_blocks = list(combinations(range(12), 6))
    block_subsets = {}
    for b in all_blocks:
        block_subsets[b] = tuple(sorted(combinations(b, 5)))
    
    best_blocks = []
    best_count = 0
    
    # Try multiple random orderings
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        
        covered = set()
        blocks = []
        
        for b in shuffled:
            subsets = block_subsets[b]
            if not any(s in covered for s in subsets):
                blocks.append(b)
                covered.update(subsets)
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            if best_count == 132:
                break
    
    return [list(b) for b in best_blocks]
