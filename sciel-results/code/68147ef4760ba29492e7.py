import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320.0

def construct():
    start_time = time.time()
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def check_validity(pts):
        """
        Check if the set of points contains no isosceles triangles.
        Returns True if valid, False otherwise.
        """
        n = len(pts)
        if n < 3:
            return True
        
        # Vectorize points
        P = np.array(pts, dtype=np.int32)
        
        # Compute squared distances for all pairs
        # This creates a (n, n) matrix
        # d[i, j] = (P[i] - P[j])^2
        # We can compute this efficiently using broadcasting
        # P[:, np.newaxis, :] - P[np.newaxis, :, :]
        # Shape: (n, n, 2)
        # Square and sum: (n, n)
        
        # To save memory/time for larger n, we might do it in chunks, but n is small (~20-30).
        # Let's just do it.
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # For each point i, we look at the distances to all other points j (j != i).
        # If any two distances are equal, it's isosceles with apex at i.
        # We need to check if the row dist_sq[i] has any duplicate values (excluding the 0 at dist_sq[i,i]).
        
        for i in range(n):
            # Get distances from i to all other points
            # dist_sq[i, :] has 0 at index i
            # We want to check if any value in dist_sq[i, :] appears more than once.
            # Since dist_sq[i, i] = 0, and all other distances are > 0,
            # we can just check if the set of non-zero distances has duplicates.
            # Or simpler: sort the row, check for adjacent duplicates.
            # But sorting is O(n log n).
            
            # Even simpler: 
            # An isosceles triangle with apex at i exists if there are j, k != i, j != k
            # such that dist(i, j) == dist(i, k).
            # This is equivalent to: the number of unique distances from i to other points is < n - 1.
            
            # Let's just use a set for each row. n is small.
            # But we are trying to be vectorized.
            
            # Let's stick to the set check for simplicity and correctness, 
            # but only for the rows.
            # Actually, the previous Python loop was O(n^3). 
            # This is O(n^2) to build dist_sq, then O(n^2) to check rows.
            
            # Let's check for duplicates in dist_sq[i, :]
            # We can use np.unique
            unique_dists = np.unique(dist_sq[i, :])
            # dist_sq[i, :] has length n. One is 0.
            # If len(unique_dists) < n, then there is a duplicate.
            # Wait, if there is a duplicate among non-zero distances, unique_dists will have fewer elements.
            # Example: [0, 5, 5, 10]. unique -> [0, 5, 10]. len 3 < 4.
            # Example: [0, 5, 10, 15]. unique -> [0, 5, 10, 15]. len 4 == 4.
            
            if len(unique_dists) < n:
                return False
                
        return True

    def greedy_initial(seed):
        """Greedy construction to get a starting set."""
        rng = random.Random(seed)
        pts = []
        
        # Precompute all points
        all_points = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(all_points)
        
        for q in all_points:
            # Check if q can be added
            candidate_pts = pts + [q]
            if check_validity(candidate_pts):
                pts.append(q)
                
        return pts

    def local_search(pts, time_budget):
        """
        Local search: try to swap a point in the set with a point outside.
        If the new set is valid, accept it.
        """
        pts = list(pts)
        start_ls = time.time()
        best_pts = list(pts)
        best_size = len(pts)
        
        rng = random.Random(123)
        
        while time.time() - start_ls < time_budget:
            if not pts:
                break
            
            # Pick a random point to remove
            i = rng.randint(0, len(pts) - 1)
            p_remove = pts[i]
            
            # Pick a random candidate to add
            # Bias towards points that are "far" or just random
            q_add = (rng.randint(0, N-1), rng.randint(0, N-1))
            
            # If q_add is already in pts, skip (or treat as deletion, but let's stick to swaps)
            if q_add in pts:
                continue
            
            # Create new set
            # Remove p_remove
            new_pts = [p for j, p in enumerate(pts) if j != i]
            # Add q_add
            new_pts.append(q_add)
            
            # Check validity
            if check_validity(new_pts):
                pts = new_pts
                if len(pts) > best_size:
                    best_size = len(pts)
                    best_pts = list(pts)
            else:
                # Optional: try to repair? No, just skip.
                pass
                
        return best_pts

    # Strategy:
    # 1. Run several greedy initializations.
    # 2. Pick the best one.
    # 3. Run local search on it.
    
    best_initial = []
    max_initial_size = 0
    
    # Try a few greedy seeds
    for seed in range(5):
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        pts = greedy_initial(seed)
        if len(pts) > max_initial_size:
            max_initial_size = len(pts)
            best_initial = pts

    if not best_initial:
        best_initial = []

    # Run local search
    if time.time() - start_time < TIME_LIMIT - 10:
        remaining_time = TIME_LIMIT - 15 - (time.time() - start_time)
        if remaining_time > 5:
            final_pts = local_search(best_initial, remaining_time)
            if len(final_pts) > len(best_initial):
                best_initial = final_pts

    return best_initial

if __name__ == "__main__":
    construct()
