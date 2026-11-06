import numpy as np
import time

def construct():
    N = 51975
    # Get divisors
    divisors = [d for d in range(3, N+1, 2) if N % d == 0]
    
    # Greedy Init
    cover_count = np.zeros(N, dtype=np.int8) # int8 is enough? Max classes 48. Yes.
    classes = []
    
    for m in divisors:
        # Find best r
        best_r = 0
        max_gain = -1
        
        # We can vectorize the gain calculation for all r at once?
        # indices = np.arange(0, N, m)
        # residues = indices % m
        # vals = cover_count[indices]
        # gain_counts = np.bincount(residues, weights=(vals==0).astype(int), minlength=m)
        # best_r = np.argmax(gain_counts)
        
        # This is O(N) per m. 48 * N is fine.
        
        indices = np.arange(0, N, m)
        residues = indices % m
        vals = cover_count[indices]
        gain_counts = np.bincount(residues, weights=(vals==0).astype(int), minlength=m)
        best_r = int(np.argmax(gain_counts))
        
        classes.append([m, best_r])
        cover_count[best_r::m] += 1
        
    # Local Search
    start_time = time.time()
    deadline = 140.0
    
    # Shuffle order for local search to avoid bias
    # We will iterate through moduli in a random order each "epoch"
    
    while time.time() < deadline:
        # Shuffle moduli
        np.random.shuffle(divisors) # Modifying list in place is okay if we don't care about original order, or use a copy
        
        for m in divisors:
            if time.time() > deadline:
                break
            
            # Find current r for this m
            # We need to map m to its current r.
            # It's easier to keep a dict or just find it in classes.
            # Since classes is small, linear search is fine.
            current_r = None
            idx = None
            for i, c in enumerate(classes):
                if c[0] == m:
                    current_r = c[1]
                    idx = i
                    break
            
            if current_r is None:
                continue # Should not happen
                
            # Calculate exclusive and uncovered for all r
            indices = np.arange(0, N, m)
            residues = indices % m
            vals = cover_count[indices]
            
            exclusive_counts = np.bincount(residues, weights=(vals==1).astype(int), minlength=m)
            uncovered_counts = np.bincount(residues, weights=(vals==0).astype(int), minlength=m)
            
            # Score for each r: uncovered_counts[r] - exclusive_counts[r]
            # We want to maximize this score.
            # Current score is uncovered_counts[current_r] - exclusive_counts[current_r]
            # Note: exclusive_counts[current_r] is the number of residues ONLY covered by this class.
            # If we switch to r_new, we lose exclusive_counts[current_r] (they become uncovered)
            # and gain uncovered_counts[r_new] (they become covered).
            # Net change in uncovered count = exclusive_counts[current_r] - uncovered_counts[r_new]
            # We want Net change < 0 => uncovered_counts[r_new] > exclusive_counts[current_r]
            
            # Let's just find r that maximizes (uncovered_counts[r] - exclusive_counts[r])?
            # No, the comparison is against the CURRENT state's exclusive count for the OLD residue.
            # If we switch from r_old to r_new:
            # New Uncovered = Old Uncovered - (uncovered in r_new) + (exclusive in r_old)
            # We want to minimize New Uncovered.
            # So we want to maximize (uncovered in r_new) - (exclusive in r_old).
            # Since exclusive in r_old is constant for a given move, we just want to maximize uncovered_counts[r_new].
            # BUT, we must ensure that r_new != r_old?
            # If r_new == r_old, change is 0.
            
            # Wait, is it just max uncovered_counts?
            # If I pick r_new with max uncovered_counts, I gain that many.
            # I lose exclusive_counts[r_old].
            # So if max(uncovered_counts) > exclusive_counts[r_old], I should switch.
            
            best_r_new = int(np.argmax(uncovered_counts))
            
            if uncovered_counts[best_r_new] > exclusive_counts[current_r]:
                # Accept move
                # Update cover_count
                cover_count[current_r::m] -= 1
                cover_count[best_r_new::m] += 1
                classes[idx] = [m, best_r_new]
                
    # Final Score
    uncovered = int(np.sum(cover_count == 0))
    return classes
