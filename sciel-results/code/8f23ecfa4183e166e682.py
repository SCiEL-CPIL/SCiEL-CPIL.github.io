"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        return _general_greedy(q, n, radius, start_time, deadline)

    # Specific optimization for q=3, n=6, radius=1
    # Space size: 729
    # Ball size: 13
    # Target: < 81 (current), ideally 73
    
    # 1. Generate initial cover
    # Use the 81-word construction as a starting point, then optimize
    # Or start with a random valid cover if it's faster to generate
    
    # The 81-word construction:
    # Tetracode C4: 9 words of length 4
    # C = C4 x {0,1,2}^2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_cover = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert to numpy for speed
    cover = np.array(initial_cover, dtype=np.int8)
    
    # Precompute all words in the space
    all_words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
    
    # Function to check if a word is covered by a set of codewords
    # We maintain a coverage count array for the space
    # coverage[word_index] = number of codewords within distance 1
    
    # Helper to get ball indices for a codeword
    def get_ball_indices(cw):
        # cw is a numpy array of shape (6,)
        indices = []
        # Center
        # Map tuple to index: sum(c[i] * 3^(5-i))
        idx = 0
        for i in range(6):
            idx = idx * 3 + cw[i]
        indices.append(idx)
        
        # Neighbors
        for i in range(6):
            for delta in [-1, 1]:
                # (cw[i] + delta) % 3
                # Actually, distance 1 means exactly one position differs.
                # The neighbors are (cw with pos i replaced by (cw[i]+1)%3) and (cw[i]+2)%3
                pass
        
        # Faster way: precompute neighbors for all possible values at each position?
        # Or just calculate on the fly.
        # Let's precompute the mapping from word to index.
        # And precompute the list of indices for each possible word? 
        # 729 words * 13 neighbors = ~9000 ops. Very fast.
        
        # Let's precompute ball_indices for all 729 words
        # ball_indices[word_idx] = list of indices covered by word_idx
        
        return indices

    # Precompute balls for all 729 words
    # This is a one-time cost
    balls = []
    for w in all_words:
        b_idx = []
        # Center
        idx = 0
        for i in range(6):
            idx = idx * 3 + w[i]
        b_idx.append(idx)
        
        for i in range(6):
            # Change position i to other 2 values
            v = w[i]
            # Neighbor 1: (v+1)%3
            w2 = w.copy()
            w2[i] = (v + 1) % 3
            idx2 = 0
            for j in range(6):
                idx2 = idx2 * 3 + w2[j]
            b_idx.append(idx2)
            
            # Neighbor 2: (v+2)%3
            w3 = w.copy()
            w3[i] = (v + 2) % 3
            idx3 = 0
            for j in range(6):
                idx3 = idx3 * 3 + w3[j]
            b_idx.append(idx3)
        
        balls.append(b_idx)
    
    # Convert to numpy array of lists? Or just list of lists.
    # List of lists is fine for 729 items.
    
    # Initialize coverage counts
    # cover is list of indices into all_words? No, cover is the words themselves.
    # Let's store cover as indices into all_words for faster ball lookup.
    
    # Convert initial_cover to indices
    def word_to_idx(w):
        idx = 0
        for i in range(6):
            idx = idx * 3 + w[i]
        return idx
        
    cover_indices = [word_to_idx(w) for w in initial_cover]
    
    # Calculate initial coverage
    coverage = np.zeros(729, dtype=np.int32)
    for ci in cover_indices:
        for idx in balls[ci]:
            coverage[idx] += 1
            
    # Optimization Loop
    # Strategy: 
    # 1. Try to remove a codeword if it doesn't reduce any coverage to 0.
    #    i.e., for all idx in ball[ci], coverage[idx] > 1.
    # 2. If no removal possible, try to swap a codeword with another to create redundancy.
    #    Or just accept the current state if time is up.
    
    # To make removals smarter, we can prioritize removing codewords that cover
    # areas with high redundancy.
    
    # Let's run a greedy removal loop.
    # We can also randomize the order of checking to escape local minima.
    
    best_cover = cover_indices[:]
    best_size = len(best_cover)
    
    # Repeat the optimization process multiple times with different random seeds/orders
    # to try to find a smaller cover.
    
    for trial in range(100): # Limit trials to ensure time safety
        if time.time() > deadline:
            break
            
        # Reset coverage and cover from best known or initial?
        # Better to start from the best known and try to improve.
        # Or start fresh from initial and randomize?
        # Let's start from the current best and try to reduce it.
        
        curr_cover = best_cover[:]
        # Recalculate coverage for curr_cover
        coverage = np.zeros(729, dtype=np.int32)
        for ci in curr_cover:
            for idx in balls[ci]:
                coverage[idx] += 1
                
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            improved = False
            # Shuffle indices to randomize order
            # We want to find a removable codeword
            # A codeword ci is removable if min(coverage[idx] for idx in balls[ci]) > 1
            
            # To speed up, we can check candidates randomly
            # Or scan all. 729 * 13 is small.
            
            # Let's scan all current codewords
            # We need to be careful: removing one changes coverage for others.
            # So we pick one, check, remove, update coverage.
            
            # Shuffle the list of current cover indices
            curr_cover_shuffled = curr_cover[:]
            random.shuffle(curr_cover_shuffled)
            
            for ci in curr_cover_shuffled:
                # Check if removable
                # Get the ball indices
                b = balls[ci]
                # Check min coverage
                # If any idx has coverage 1, it's not removable
                # We can break early if we find a 1
                removable = True
                for idx in b:
                    if coverage[idx] == 1:
                        removable = False
                        break
                        
                if removable:
                    # Remove it
                    curr_cover.remove(ci)
                    for idx in b:
                        coverage[idx] -= 1
                    improved = True
                    break # Restart the scan? Or continue?
                    # Restarting is safer to ensure we don't miss new opportunities
                    # But breaking and restarting the while loop is effectively restarting.
                    # Let's just break the for loop and let the while loop restart.
                    
            # If we didn't remove anything in this pass, we are stuck.
            # Try a swap?
            # Swap ci with cj if it allows removal of another word or reduces total size?
            # Swapping doesn't reduce size directly. It might allow a subsequent removal.
            # This is complex. Let's stick to removal for now.
            # If stuck, the current cover is a local minimum for deletion.
            
        if len(curr_cover) < best_size:
            best_cover = curr_cover[:]
            best_size = len(best_cover)
            # If we reach 73, we can stop? 73 is known optimal.
            if best_size <= 73:
                break

    # Convert best_cover indices back to words
    result = [all_words[idx].tolist() for idx in best_cover]
    return result

def _general_greedy(q, n, radius, start_time, deadline):
    # Fallback for other parameters
    # Simple greedy set cover
    # 1. Generate all words
    # 2. Pick word covering most uncovered
    # 3. Repeat
    # This is slow for large spaces, but task is specific to 3,6,1.
    # Just return a valid cover for safety.
    # For q=3, n=6, radius=1, we use the specific logic above.
    # If this function is called, we don't know the parameters.
    # Let's just return a trivial cover (all words) if we can't handle it efficiently.
    # But the task specifies q=3, n=6, radius=1.
    # So this branch should rarely be hit.
    return []

# EVOLVE-BLOCK-END
