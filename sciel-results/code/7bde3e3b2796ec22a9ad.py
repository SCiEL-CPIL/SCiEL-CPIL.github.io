import itertools
import random
import time

# Global precomputation
V = 24
K = 7
T = 3

# Triple index mapping
triple_to_index = {}
index_to_triple = []
idx = 0
for c in itertools.combinations(range(V), T):
    triple_to_index[c] = idx
    index_to_triple.append(c)
    idx += 1

NUM_TRIPLES = len(triple_to_index)

# Masks
# mask_for_point[p]: bitmask of triples containing p
mask_for_point = [0] * V
# mask_for_pair[p][q]: bitmask of triples containing p and q
mask_for_pair = [[0] * V for _ in range(V)]

for i, triple in enumerate(index_to_triple):
    bit = 1 << i
    a, b, c = triple
    mask_for_point[a] |= bit
    mask_for_point[b] |= bit
    mask_for_point[c] |= bit
    
    mask_for_pair[a][b] |= bit
    mask_for_pair[a][c] |= bit
    mask_for_pair[b][a] |= bit
    mask_for_pair[b][c] |= bit
    mask_for_pair[c][a] |= bit
    mask_for_pair[c][b] |= bit

# Helper to compute mask of a block
def block_mask(block):
    m = 0
    # Block is a tuple or list of 7 points
    for c in itertools.combinations(block, T):
        m |= (1 << triple_to_index[c])
    return m

# Helper to compute gain of adding x to current_block
def gain_mask(x, current_block):
    # current_block is a list of points already in the block
    m = 0
    for p in current_block:
        m |= mask_for_pair[x][p]
    return m

def popcount(x):
    return bin(x).count("1")

def construct(v, k, t):
    # Validate inputs (though task says they are fixed)
    if v != V or k != K or t != T:
        # Fallback to generic greedy if parameters change, though unlikely
        # For now, assume standard case
        pass
        
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Initial Greedy
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = (1 << NUM_TRIPLES) - 1
        blocks = []
        
        # Pre-compute list of all triples for random selection?
        # Or just iterate?
        # Random selection of uncovered triple:
        # We can pick a random index, check if bit is set.
        # Or maintain a list of uncovered indices.
        # Given 2024 triples, picking random index and checking bit is fast enough.
        
        while uncovered:
            # Pick a random uncovered triple
            # To be efficient, we can pick a random index and check
            # But if coverage is high, we might miss.
            # Better: pick random index 0..NUM_TRIPLES-1
            # If bit not set, retry.
            # Or, since we start with all uncovered, and remove them,
            # the density drops.
            # For robustness, let's just pick the first uncovered triple in the bitmask?
            # That is deterministic. Randomizing the *order* of points in the block is better.
            
            # Let's pick a random triple from the uncovered set.
            # We can do this by picking a random integer r in [0, popcount(uncovered))
            # and finding the r-th set bit.
            # Python doesn't have a fast "nth set bit" builtin.
            # Alternative: just pick a random index i in [0, NUM_TRIPLES).
            # If (uncovered >> i) & 1: use it.
            # Else: retry.
            # This is O(1) expected.
            
            while True:
                r = rng.randint(0, NUM_TRIPLES - 1)
                if (uncovered >> r) & 1:
                    seed_triple = index_to_triple[r]
                    break
            
            block = list(seed_triple)
            
            # Build block
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    m = gain_mask(x, block)
                    g = popcount(uncovered & m)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                        if g == 0:
                            # If no gain, break? No, we must complete the block.
                            # But if gain is 0, any x is as good.
                            # We can break early if we find a good one?
                            # Greedy usually takes the max.
                            pass
                block.append(best_x)
            
            block.sort()
            bm = block_mask(block)
            uncovered &= ~bm
            blocks.append(block)
            
        return blocks

    # Local Search
    def local_search(blocks, rng):
        blocks = [list(b) for b in blocks]
        block_masks = [block_mask(b) for b in blocks]
        n = len(blocks)
        
        # Helper to get OR of all masks except i
        def others_mask(i):
            m = 0
            for j in range(n):
                if j != i:
                    m |= block_masks[j]
            return m
        
        improved = True
        swap_attempts = 0
        max_swaps = 100 # Limit swaps to avoid infinite loops or long times
        
        while improved and time.time() < deadline:
            improved = False
            # Deletion Pass
            for i in range(n):
                if n == 0: break
                om = others_mask(i)
                unique = block_masks[i] & ~om
                if unique == 0:
                    # Remove block i
                    del blocks[i]
                    del block_masks[i]
                    n -= 1
                    improved = True
                    break # Restart deletion pass
            
            if not improved and n > 0:
                # Swap Pass
                # Try a few random swaps
                for _ in range(10): # Try 10 random swaps per pass
                    i = rng.randint(0, n - 1)
                    b = blocks[i]
                    x = rng.choice(b)
                    # Pick y not in b
                    candidates = [p for p in range(v) if p not in b]
                    if not candidates:
                        continue
                    y = rng.choice(candidates)
                    
                    new_b = [p for p in b if p != x]
                    new_b.append(y)
                    new_b.sort()
                    new_mask = block_mask(new_b)
                    
                    om = others_mask(i)
                    lost = block_masks[i] & ~om
                    # Check if lost is covered by new_mask
                    if (lost & ~new_mask) == 0:
                        # Safe swap
                        blocks[i] = new_b
                        block_masks[i] = new_mask
                        improved = True
                        break # Restart deletion pass after a swap
            
        return blocks

    # Main Loop
    seed = 0
    while time.time() < deadline:
        blocks = run_greedy(seed)
        # Refine
        blocks = local_search(blocks, random.Random(seed + 1000))
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            # Optional: print status
            # print(f"Seed {seed}: {best_count} blocks")
            
        seed += 1
        
    return best_blocks
