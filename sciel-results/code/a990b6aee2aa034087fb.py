import random
import time
import numpy as np

N = 64

def construct():
    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    # State
    S = []          # List of points
    dist_sets = []  # List of sets of squared distances from S[i] to others
    point_set = set() # For O(1) membership check
    available = [(x, y) for x in range(N) for y in range(N)]
    
    # Shuffle available for initialization
    random.seed(42)
    random.shuffle(available)

    def try_add(p):
        """
        Tries to add point p to the set.
        Returns True if added, False otherwise.
        """
        if p in point_set:
            return False
            
        # Check validity
        new_dists = set()
        for i, q in enumerate(S):
            d = dist_sq(p, q)
            if d in new_dists:
                return False
            if d in dist_sets[i]:
                return False
            new_dists.add(d)
            
        # Valid, perform addition
        S.append(p)
        dist_sets.append(new_dists)
        point_set.add(p)
        
        # Update existing points' distance sets
        idx_new = len(S) - 1
        for i in range(idx_new):
            d = dist_sq(S[i], p)
            dist_sets[i].add(d)
            
        # Remove from available list (efficiency: we can use a set for available too, or just leave it and filter)
        # Since available is a list, removing is O(N). 
        # Better to use a set for available? 
        # If we use a set for available, we can't easily pick random without converting to list.
        # Given N=4096, O(N) removal is fine. But we do it millions of times? No, only when adding.
        # Swaps involve remove+add.
        # Let's use a set for available_points to make removal O(1).
        # But random.choice needs a sequence. 
        # We can maintain a list and a set.
        # Or just pick from the full grid and check membership. 
        # Grid size 4096. Set size ~500. 
        # Probability of picking an available point is high.
        # Let's stick to picking from `available` list.
        try:
            available.remove(p)
        except ValueError:
            pass # Should not happen if logic is correct

        return True

    def try_remove(idx):
        """
        Removes point at index idx from the set.
        """
        p = S[idx]
        
        # Update existing points' distance sets
        for i in range(len(S)):
            if i == idx:
                continue
            d = dist_sq(S[i], p)
            dist_sets[i].remove(d)
            
        # Remove from S and dist_sets
        S.pop(idx)
        dist_sets.pop(idx)
        point_set.remove(p)
        
        # Add back to available
        available.append(p)

    # 1. Greedy Initialization
    # We iterate through shuffled available points
    # We need to be careful: `available` is being modified.
    # Let's make a copy for initialization loop? Or just iterate a list of all points.
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    for p in all_points:
        if try_add(p):
            continue
        # If it fails, it's still in available. 
        # Wait, try_add doesn't remove from available if it fails.
        # So available still contains all points that were not added.
        # This is correct.
        
    best_set_size = len(S)
    best_set = S[:]
    best_dist_sets = [ds.copy() for ds in dist_sets] # Deep copy? Sets are mutable.
    best_point_set = point_set.copy()
    best_available = available[:]

    # 2. Local Search
    start_time = time.time()
    timeout = 280
    
    # To speed up random selection from available, we can just pick random indices
    # But available changes size.
    
    while time.time() - start_time < timeout:
        # Strategy:
        # 1. Try to add a random available point.
        # 2. If fail, try a swap: remove a random current point, try to add the candidate.
        
        # Pick a candidate from available
        if not available:
            break
            
        cand_idx = random.randint(0, len(available) - 1)
        p = available[cand_idx]
        
        # Try add
        if try_add(p):
            if len(S) > best_set_size:
                best_set_size = len(S)
                best_set = S[:]
                best_dist_sets = [ds.copy() for ds in dist_sets]
                best_point_set = point_set.copy()
                best_available = available[:]
        else:
            # Try swap
            if S:
                # Pick a point to remove
                rem_idx = random.randint(0, len(S) - 1)
                # Store state to revert if swap fails
                # We need to restore S, dist_sets, point_set, available
                # This is expensive if we copy everything.
                
                # Instead, we can do:
                # 1. Remove rem_idx
                # 2. Try add p
                # 3. If add fails, revert removal (add back rem_idx)
                
                # But "revert removal" is not the same as "undo remove". 
                # Remove updates dist_sets of others.
                # If we remove A, then try add B, and B fails, we must put A back.
                # Putting A back requires updating dist_sets of others again.
                # Is it safe?
                # Yes, if we just call an "add" routine that doesn't check validity?
                # No, we must ensure the state is consistent.
                
                # Let's implement a simpler swap:
                # 1. Snapshot S, dist_sets, point_set, available? 
                #    Copying dist_sets (list of sets) is O(N^2) worst case? 
                #    N~500, N^2=250k. Too slow for millions of iterations.
                
                # Alternative:
                # 1. Remove A (idx).
                # 2. Check if B is valid against the *modified* set.
                #    We can do this without actually adding B yet.
                #    We need to check B against S (without A).
                #    And check if B's distances collide with each other.
                # 3. If valid, Add B.
                # 4. If invalid, Re-add A.
                
                # Let's write a `check_valid(p, S, dist_sets)` function that doesn't modify.
                # But `try_add` does modify.
                # Let's refactor `try_add` to use a helper `check_valid`.
                
                pass # Logic handled below

    # Refactor: Separate check and add
    
    # Let's rewrite the core logic inside construct to support this better.
    # I will replace the whole function body.

    return best_set
