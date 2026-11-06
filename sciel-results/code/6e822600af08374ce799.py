import time
import heapq

N = 9
V = 1 << N
MAX_TIME = 110

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(V)]
for v in range(V):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    
    # Beam width
    BEAM_WIDTH = 1000
    
    # Initial state: (path, used_mask, bad_mask)
    # We use a list of states.
    # State: (path, used_mask, bad_mask)
    # To make comparison easy, we'll store them in a list and sort/heapify.
    
    # We need to track the best path found so far globally.
    best_path = [0]
    best_len = 1
    
    # Initial beam
    beam = [([0], 1, 0)]
    
    while True:
        if time.time() - start_time > MAX_TIME:
            break
            
        next_beam = []
        
        for path, used_mask, bad_mask in beam:
            cur = path[-1]
            
            # Find valid next vertices
            # Valid v must be:
            # 1. Neighbor of cur
            # 2. Not in used_mask
            # 3. Not in bad_mask
            # Note: bad_mask contains vertices that are adjacent to path[:-1].
            # If we add v, the new bad_mask for the *next* step will include neighbors of cur.
            # But for *this* step, v just needs to not be in the *current* bad_mask.
            # Wait, the definition of bad_mask in the seed was:
            # "bad_mask: bitmask of vertices that are adjacent to path[:-1]"
            # So if path is [v0, v1, ..., vk], bad_mask contains neighbors of v0...v(k-1).
            # vk is the current last.
            # Next vertex v(k+1) must not be adjacent to v0...v(k-1).
            # So v(k+1) must not be in bad_mask.
            # v(k+1) must be neighbor of vk.
            # v(k+1) must not be in used_mask.
            
            candidates = []
            for v in NEIGHBORS[cur]:
                if (used_mask & (1 << v)) == 0 and (bad_mask & (1 << v)) == 0:
                    candidates.append(v)
            
            if not candidates:
                continue
                
            for v in candidates:
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                # New bad_mask for the next step:
                # The "path[:-1]" of the new path is the old path.
                # So new_bad_mask = old_bad_mask | NEIGHBORS[cur]
                # Note: NEIGHBORS[cur] includes cur's neighbors.
                # We need to mask out used vertices? No, bad_mask is just a set of forbidden vertices.
                # It's okay if it contains used vertices, we check used_mask separately.
                new_bad = bad_mask
                for nb in NEIGHBORS[cur]:
                    new_bad |= (1 << nb)
                    
                # Heuristic: 
                # We want to prioritize paths that are long and have many options.
                # Let's calculate the number of valid moves from the new last vertex v.
                # This is a bit expensive if done for every candidate in every beam state.
                # But V=512, so it's small.
                # Let's precompute a quick count?
                # Or just use length as primary, and maybe some randomness or static order.
                # Actually, a good heuristic for snake-in-the-box is "degree of freedom".
                
                # Let's count valid moves from v in the new context.
                # valid moves from v: neighbors of v that are not in new_used and not in new_bad.
                # Note: new_bad includes neighbors of cur.
                # v is a neighbor of cur.
                # So v is in new_bad? No, new_bad is for the NEXT step.
                # The next vertex v' must not be in new_bad.
                # So we count neighbors of v that are not in new_used and not in new_bad.
                
                moves = 0
                for nb in NEIGHBORS[v]:
                    if (new_used & (1 << nb)) == 0 and (new_bad & (1 << nb)) == 0:
                        moves += 1
                        
                # Score: length * 100 + moves
                # We want to keep the best BEAM_WIDTH states.
                # Using a heap is efficient.
                # We push (score, path, used_mask, bad_mask)
                # But we need to keep the longest path overall.
                
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path[:]
                    
                # We use a negative score for min-heap to get max-heap behavior
                # Score = len(new_path) * 1000 + moves
                # We want to maximize this.
                score = len(new_path) * 1000 + moves
                next_beam.append((-score, new_path, new_used, new_bad))
        
        if not next_beam:
            break
            
        # Keep only the top BEAM_WIDTH states
        # Sort by score (which is negative, so smallest negative is best)
        next_beam.sort(key=lambda x: x[0])
        beam = [tuple(x[1:]) for x in next_beam[:BEAM_WIDTH]]
        
        # Optional: If we have a very long path, we might want to reduce beam width or just continue.
        # The time limit will stop us.
        
    return best_path
