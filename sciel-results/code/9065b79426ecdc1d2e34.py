import time
import random

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks for all vertices
# NEIGHBOR_MASKS[v] is a bitmask where bit i is set if vertex i is a neighbor of v
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = m

def construct():
    deadline = time.time() + 115.0
    best_path = [0]
    best_len = 1
    
    # Precompute list of bits for random shuffling
    bits = list(range(N))
    
    def beam_search(start, width, rng):
        """
        Beam search using bitmask state.
        State: (path_list, used_mask, forbidden_mask)
        used_mask: bitmask of vertices in the path
        forbidden_mask: bitmask of vertices that are adjacent to any vertex in the path 
                        except the last one. These vertices cannot be added next.
        """
        # Initial state
        # Path starts with [start]
        # used: start
        # forbidden: 0 (no non-last vertices yet)
        beam = [([start], 1 << start, 0)]
        best_local_path = [start]
        best_local_len = 1
        
        while True:
            if time.time() > deadline - 2:
                break
            
            new_beam = []
            seen_suffixes = set()
            
            for path, used_mask, forbidden_mask in beam:
                cur = path[-1]
                
                # Candidates are neighbors of cur that are not used and not forbidden
                # Note: 'forbidden' contains neighbors of all previous vertices.
                # We need to add neighbors of 'cur' to forbidden for the NEXT step, 
                # but for THIS step, cur's neighbors are allowed (unless used).
                # Wait, the definition of forbidden_mask in the state is: 
                # "vertices at dist 1 from any non-last path vertex".
                # So, candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
                
                candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
                
                if candidates == 0:
                    # Dead end
                    if len(path) > best_local_len:
                        best_local_len = len(path)
                        best_local_path = list(path)
                    continue
                
                # Generate extensions
                # To ensure diversity, we might want to try all candidates, 
                # but that explodes. We rely on the beam width to keep diversity.
                # We iterate through set bits in candidates.
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    
                    new_path = path + [v]
                    new_used = used_mask | (1 << v)
                    # The current vertex 'cur' becomes a non-last vertex, 
                    # so its neighbors are now forbidden for the next step
                    new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                    
                    # Check for diversity in the beam
                    # Using the last few vertices as a key to avoid identical suffixes
                    # This helps in keeping different prefixes
                    key = tuple(new_path[-5:]) 
                    if key in seen_suffixes:
                        continue
                    seen_suffixes.add(key)
                    
                    new_beam.append((new_path, new_used, new_forbidden))
                    
                    if len(new_path) > best_local_len:
                        best_local_len = len(new_path)
                        best_local_path = list(new_path)
                    
                    # Early termination if beam is too large? 
                    # We will sort and prune below.
                
                # Optimization: if we already have many candidates, we can stop expanding this beam
                # to save time, but beam search usually expands all.
                # Given the small N=9, expanding all is fine.
            
            if not new_beam:
                break
            
            # Prune: keep top 'width' beams
            # We want to keep the longest paths. If lengths are equal, we might want diversity.
            # Sorting by length descending is standard.
            new_beam.sort(key=lambda x: -len(x[0]))
            
            # To add more diversity, we can randomize the selection among equal-length paths
            # or just take the first 'width' ones.
            # Let's take the first 'width' ones.
            beam = new_beam[:width]
            
            # If the best path in the beam is not getting longer, we might be stuck.
            # But beam search continues until dead ends or time limit.
            
        return best_local_path, best_local_len

    # Run multiple beam searches with different parameters
    # We vary the beam width and the random seed/priority
    # Larger beam width = more memory and time per step, but better exploration
    
    widths = [50, 100, 200, 500, 1000]
    
    # We will cycle through widths and restarts
    # Each restart uses a random start vertex and random bit priority (though beam search doesn't use priority directly, 
    # the diversity comes from the beam itself. However, randomizing the start is good).
    
    # Actually, beam search doesn't use a "priority" list for ordering bits in the same way greedy does.
    # It expands all valid candidates. The "randomness" comes from the initial state and the pruning.
    # To introduce randomness in exploration, we can randomize the start vertex.
    
    rng = random.Random(42)
    
    # Allocate time to different beam widths
    # Smaller beams are faster, so we can run more of them.
    # Larger beams are slower but might find longer paths.
    
    while time.time() < deadline - 10:
        # Choose a beam width
        # Bias towards larger widths as we go? Or random?
        # Let's try a mix.
        width = rng.choice(widths)
        
        # Random start vertex
        start = rng.randint(0, SIZE - 1)
        
        p, L = beam_search(start, width, rng)
        
        if L > best_len:
            best_len = L
            best_path = p
            
        # If we found a very long path, we might want to try to extend it with a greedy step?
        # Or just continue with beam searches.
        
    return best_path

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
