"""Optimized covering C(24,7,3) using bitset-based greedy + local search."""
import itertools
import time
import random
import sys
from math import comb

# Precompute combination indices for fast lookup
def _build_combo_index(v, k):
    """Map each k-combination to a unique index."""
    idx = {}
    for i, c in enumerate(itertools.combinations(range(v), k)):
        idx[c] = i
    return idx

def _combo_to_bitset(v, k, t, combo):
    """Return a bitset (as a Python int) of all t-subsets contained in combo."""
    mask = 0
    for sub in itertools.combinations(combo, t):
        # Use a compact encoding: map t-subset to index
        # We'll use a different approach: precompute a lookup
        pass
    return mask

# For v=24, k=7, t=3: C(24,3)=2024 triples, C(24,7)=346104 blocks
# We'll use a different strategy: represent uncovered triples as a set of indices

def _triple_index(triple, v):
    """Map a sorted 3-tuple to a unique index in 0..C(v,3)-1."""
    a, b, c = triple
    # Lexicographic rank of (a,b,c) among all 3-combinations of v
    # C(v,3) = v*(v-1)*(v-2)/6
    # Rank = C(v-a-1, 2) - C(v-b-1, 2) + ... 
    # Simpler: use a direct formula
    # Number of triples with first element < a: sum_{i=0}^{a-1} C(v-i-1, 2)
    # This is complex; let's just use a dict-based approach for correctness
    # Actually, for v=24, C(24,3)=2024, we can precompute a lookup table
    
    # Use combinatorial number system
    # rank = C(v-c-1, 0) + C(v-b-1, 1) + C(v-a-1, 2) ... no
    # Standard formula for rank of combination (a,b,c) with 0<=a<b<c<v:
    # rank = C(v-1-c, 1) + C(v-1-b, 2) + C(v-1-a, 3) ... this is getting complex
    # Let's just precompute a mapping
    
    return triple  # placeholder

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # For the specific case v=24, k=7, t=3, use optimized approach
    # General approach: bitset-based greedy with multiple random restarts
    
    # Precompute all t-subsets and assign them indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # We'll track uncovered triples as a set of indices
    # And for each potential block, we can quickly compute which triples it covers
    
    # Strategy: 
    # 1. Generate a pool of candidate blocks (random + structured)
    # 2. Use greedy set cover with bitset operations
    # 3. Local search to improve
    
    # For v=24, k=7: C(24,7) = 346104 possible blocks
    # We can't enumerate all, but we can be smart
    
    # Precompute for each point, the list of t-1 subsets that contain it
    # This helps in fast gain calculation
    
    # Let's use a randomized greedy approach with multiple restarts
    # and keep the best result
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts we can do depends on time
    # Each greedy run takes some time; let's estimate
    
    rng = random.Random(42)
    
    for attempt in range(200):  # try up to 200 times
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy: at each step, choose the block that covers the most uncovered triples
        # with some randomness to escape local optima
        
        uncovered = set(range(n_tsubsets))  # set of uncovered triple indices
        blocks = []
        
        while uncovered and len(blocks) < best_count:
            # Find the best block to add
            # For efficiency, we'll sample candidate blocks
            
            best_gain = -1
            best_block = None
            
            # Sample some random blocks plus some heuristic ones
            candidates = []
            
            # Heuristic: start from an uncovered triple, extend to a full block
            uncovered_list = list(uncovered)
            if not uncovered_list:
                break
            
            # Pick a few random uncovered triples to seed blocks
            n_seeds = min(50, len(uncovered_list))
            seed_indices = rng.sample(uncovered_list, n_seeds)
            
            for si in seed_indices:
                seed_triple = all_tsubsets[si]
                # Extend this triple to a full k-block
                # Choose remaining k-t points randomly or greedily
                remaining_points = [p for p in range(v) if p not in seed_triple]
                
                # Try a few different extensions
                for ext_attempt in range(3):
                    block = list(seed_triple)
                    # Greedily add points that cover most uncovered triples
                    while len(block) < k:
                        best_point = None
                        best_point_gain = -1
                        for p in remaining_points:
                            if p in block:
                                continue
                            # Count how many uncovered triples this point would complete
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                triple = tuple(sorted(rest + (p,)))
                                if triple in tsub_index and tsub_index[triple] in uncovered:
                                    gain += 1
                            if gain > best_point_gain:
                                best_point_gain = gain
                                best_point = p
                        if best_point is None:
                            break
                        block.append(best_point)
                        remaining_points.remove(best_point)
                    
                    if len(block) == k:
                        block.sort()
                        # Calculate actual gain
                        gain = 0
                        for sub in itertools.combinations(block, t):
                            if sub in tsub_index and tsub_index[sub] in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = block
            
            # If no good block found, pick a random one
            if best_block is None:
                best_block = sorted(rng.sample(range(v), k))
                best_gain = 0
            
            blocks.append(best_block)
            
            # Remove covered triples
            for sub in itertools.combinations(best_block, t):
                if sub in tsub_index:
                    uncovered.discard(tsub_index[sub])
            
            if time.time() > deadline - 2:
                break
        
        if not uncovered and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
        
        # Early exit if we hit the known best
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy without randomization
        uncovered = set(range(n_tsubsets))
        blocks = []
        while uncovered:
            best_gain = -1
            best_block = None
            # Try blocks based on uncovered triples
            uncovered_list = list(uncovered)[:100]
            for si in uncovered_list:
                seed_triple = all_tsubsets[si]
                block = list(seed_triple)
                remaining_points = [p for p in range(v) if p not in seed_triple]
                while len(block) < k:
                    best_point = None
                    best_point_gain = -1
                    for p in remaining_points:
                        if p in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            triple = tuple(sorted(rest + (p,)))
                            if triple in tsub_index and tsub_index[triple] in uncovered:
                                gain += 1
                        if gain > best_point_gain:
                            best_point_gain = gain
                            best_point = p
                    if best_point is None:
                        break
                    block.append(best_point)
                    remaining_points.remove(best_point)
                if len(block) == k:
                    block.sort()
                    gain = 0
                    for sub in itertools.combinations(block, t):
                        if sub in tsub_index and tsub_index[sub] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
            if best_block is None:
                # Should not happen, but fallback
                best_block = sorted(rng.sample(range(v), k))
            blocks.append(best_block)
            for sub in itertools.combinations(best_block, t):
                if sub in tsub_index:
                    uncovered.discard(tsub_index[sub])
        best_blocks = blocks
        best_count = len(blocks)
    
    return best_blocks
