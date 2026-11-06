"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
NUM_VERTS = 1 << N  # 512

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_VERTS)]

# Precompute bit_count lookup for all possible values (0 to 2**N - 1)
# Actually we need bit_count of XOR results, which are in [0, 2**N - 1]
BC = np.zeros(NUM_VERTS, dtype=np.uint8)
for i in range(1, NUM_VERTS):
    BC[i] = BC[i >> 1] + (i & 1)

def construct():
    """Construct a long induced path (snake) in Q_9."""
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Start with the greedy snake
    best_path = _greedy_snake()
    best_len = len(best_path)
    
    # Try multiple random seeds and beam searches
    rng = np.random.default_rng(42)
    
    # Use a smaller number of seeds but larger beam widths or more steps?
    # The previous loop ran 50 seeds.
    # With the faster beam search (integer bitmask), we can afford more steps or larger beams.
    
    # Let's try 20 seeds with larger beam widths.
    for seed in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random starting vertex
        start = int(rng.integers(0, NUM_VERTS))
        
        # Beam search with larger widths
        # Start with 100, go up to 1000?
        # Beam search complexity is O(Steps * BeamWidth * N * PathLen)
        # With int bitmask, PathLen check is O(PathLen).
        # Let's try beam widths: 200, 400, 800...
        beam_width = 100 + seed * 50
        if beam_width > 1000:
            beam_width = 1000
            
        path = _beam_search(start, beam_width, deadline - 2.0)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    return best_path


def _greedy_snake():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any non-consecutive vertex in path
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _beam_search(start, beam_width, deadline):
    """Beam search to find a long induced path starting from 'start'."""
    # used is an integer bitmask of 512 bits
    # path is a list of integers
    
    # Initialize beam with single-element paths
    # (path, used_mask)
    beam = [([start], 1 << start)]
    
    while True:
        if time.time() > deadline:
            break
        
        # Expand each beam element
        candidates = []
        for path, used in beam:
            cur = path[-1]
            for b in range(N):
                v = cur ^ (1 << b)
                v_bit = 1 << v
                if used & v_bit:
                    continue
                # Check if v is adjacent to any non-consecutive vertex in path
                # path[-1] is cur, which is adjacent to v by construction.
                # We check path[:-1]
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    new_path = path + [v]
                    new_used = used | v_bit
                    candidates.append((new_path, new_used))
        
        if not candidates:
            break
        
        # Score candidates: prefer longer paths
        # To save time, we can just sort by length. 
        # Randomness in tie-breaking helps exploration but sorting is O(M log M).
        # If M is large, this is expensive.
        # We can sample if M is huge, but beam_width limits M to beam_width * N.
        # With beam_width=500, M=4500. Sorting 4500 items is fast.
        
        # Optimization: If all candidates have the same length (which they will, 
        # since we expand all current beam paths by 1 step), we don't need to sort by length!
        # All candidates in this generation have length len(beam[0][0]) + 1.
        # Wait, beam elements can have different lengths if we prune differently?
        # No, in a standard beam search step, we expand all current beams.
        # If we keep the top K, they all have the same length in the next step.
        # However, if we start with different lengths, they will drift.
        # But here we start with length 1.
        # Step 1: all length 2.
        # Step 2: all length 3.
        # So we can just pick random beam_width candidates if they all have the same length?
        # Or sort by some heuristic?
        # Usually, we want to keep the most "promising" ones.
        # A simple heuristic: random selection among valid extensions might be good for diversity.
        # Or we can score by "potential" (number of available neighbors for the new end).
        
        # Let's score by number of available neighbors for the new end vertex.
        # This encourages paths that don't get stuck.
        
        def score_candidate(c):
            path, used = c
            cur = path[-1]
            count = 0
            for b in range(N):
                v = cur ^ (1 << b)
                v_bit = 1 << v
                if not (used & v_bit):
                    # Check validity quickly? No, just count unvisited neighbors.
                    # This is a heuristic.
                    count += 1
            return count
        
        # If candidates are many, computing score for all is expensive.
        # Let's just use random selection if we want speed, or simple sort by length if lengths differ.
        # Given the tight time limit, let's try to be efficient.
        # Since all candidates in a single expansion step have the same length (len(path)+1),
        # sorting by length is useless.
        # We can just take a random sample of size beam_width if len(candidates) > beam_width.
        # This preserves diversity.
        
        if len(candidates) > beam_width:
            # Random sampling is O(M) to generate indices? No, O(K log M) for selection.
            # random.sample is efficient.
            beam = random.sample(candidates, beam_width)
        else:
            beam = candidates
        
        # Check if all beam elements are stuck
        # If we used random sampling, we might have kept stuck ones?
        # No, we only added valid candidates.
        # But a candidate might be valid now but have no valid extensions next step.
        # The loop will handle this: if no candidates are generated in the next step, we break.
        
        # Optimization: Early termination if no candidate in the new beam has any valid extensions?
        # We can check this in the next iteration's candidate generation.
        # If candidates list is empty next time, we break.
    
    # Return the longest path in the beam
    if beam:
        longest = max(beam, key=lambda x: len(x[0]))
        return longest[0]
    return [start]


# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to keep the interface.
# The EVOLVE-BLOCK is just a marker; the actual code is above.
# EVOLVE-BLOCK-END
