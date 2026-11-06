import time
import random
import numpy as np

N = 64
TIME_LIMIT = 300.0

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 15.0
    
    # Precompute coordinates
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    total = N * N
    
    # Precompute squared distances
    # dist[i, j] = (xi-xj)^2 + (yi-yj)^2
    # Using int32 to save memory if possible, but int64 is safer for 64x64 grid
    # Max dist is (63^2 + 63^2) = 7938, fits in int16 even, but let's use int32 for speed in ops
    dx = (xs[:, None] - xs[None, :]).astype(np.int32)
    dy = (ys[:, None] - ys[None, :]).astype(np.int32)
    dist = dx * dx + dy * dy
    
    def can_add_vectorized(candidate, current_indices):
        """
        Check if candidate can be added to current_indices without forming an isosceles triangle.
        Returns True if valid.
        
        Conditions for NO isosceles triangle involving candidate and current points:
        1. Apex at candidate: All distances from candidate to current points must be unique.
        2. Apex at current point i: For any pair (i, j) in current, dist(candidate, i) != dist(i, j) 
           AND dist(candidate, j) != dist(i, j).
           Wait, the condition is: Triangle (c, i, j) is isosceles if:
           - d(c,i) == d(c,j)  [Apex at c] -> Covered by condition 1 (unique dists from c)
           - d(c,i) == d(i,j)  [Apex at i]
           - d(c,j) == d(i,j)  [Apex at j]
           
           So we need to check:
           A. len(unique(d(c, current))) == len(current)
           B. For all i, j in current (i<j): d(c,i) != d(i,j) and d(c,j) != d(i,j)
           
           Vectorized check for B:
           Let D_c = dist[c, current_indices] (shape K)
           Let D_sub = dist[np.ix_(current_indices, current_indices)] (shape KxK)
           
           We need to ensure that for any pair (i,j), D_c[i] != D_sub[i,j] and D_c[j] != D_sub[i,j].
           This is equivalent to:
           For each i, D_c[i] is not in the row D_sub[i, :] (excluding self? no, dist(i,i)=0, D_c[i] > 0 usually)
           Actually, simpler:
           Construct a boolean matrix M where M[i,j] = (D_c[i] == D_sub[i,j]) or (D_c[j] == D_sub[i,j]).
           If any M[i,j] is True, it's invalid.
           
           However, building KxK matrix for every candidate is expensive if K is large.
           But K is likely small (< 50-100) in the early/mid game, and we can break early.
           
           Alternative efficient check:
           1. Check unique distances from candidate.
              d_c = dist[c, current_indices]
              if len(np.unique(d_c)) < len(d_c): return False
              
           2. Check base cases.
              For each current point i, let d_i = d_c[i].
              We need to check if d_i appears in dist[i, current_indices] (excluding i itself? 
              dist[i,i]=0. d_i is distance from c to i. c != i, so d_i > 0. 
              So we just check if d_i is in dist[i, current_indices].
              Note: dist[i, current_indices] contains distances from i to all current points.
              If d_i == dist[i, j] for some j != i, then triangle (c,i,j) has d(c,i)==d(i,j).
              This forms an isosceles triangle with apex at i.
              
              So the condition simplifies to:
              For all i in current_indices:
                  d_c[i] is NOT in dist[i, current_indices]
                  
              Wait, is that sufficient?
              Triangle (c, i, j).
              Sides: d(c,i), d(c,j), d(i,j).
              Isosceles if:
              1. d(c,i) == d(c,j) -> Checked by unique d_c.
              2. d(c,i) == d(i,j) -> Checked by "d_c[i] in dist[i, current_indices]"
              3. d(c,j) == d(i,j) -> Checked by "d_c[j] in dist[j, current_indices]"
              
              Yes! So we just need to check for each i, if d_c[i] is in the row i of the sub-distance matrix.
              
              Vectorized:
              d_c = dist[c, current_indices]  # shape (K,)
              # We want to know for each i, if d_c[i] exists in dist[i, current_indices]
              # dist[i, current_indices] is a row vector.
              # We can create a matrix of comparisons?
              # Or loop over K? If K is small, loop is fine. If K is large, we need vectorization.
              
              Let's try vectorizing the "in row" check.
              # Create a boolean mask for each i: (dist[i, current_indices] == d_c[i])
              # This is a (K, K) boolean matrix B where B[i,j] = (dist[i, current_indices][j] == d_c[i])
              # We need to check if any B[i, :] is True.
              
              # dist[i, current_indices] for all i is dist[np.ix_(current_indices, current_indices)]
              D_sub = dist[np.ix_(current_indices, current_indices)]
              # d_c is shape (K,). We want to compare D_sub[i, :] with d_c[i].
              # D_sub[i, j] == d_c[i]
              # This is equivalent to: D_sub == d_c[:, None]
              matches = (D_sub == d_c[:, None])
              # If any element in matches is True, then for some i, there exists j such that d(i,j) == d(c,i).
              # Note: j can be i? dist[i,i]=0. d_c[i] > 0 (since c != i). So j != i automatically.
              if np.any(matches):
                  return False
                  
              return True
        """
        K = len(current_indices)
        if K < 2:
            return True
            
        # 1. Check unique distances from candidate
        d_c = dist[candidate, current_indices]
        if len(np.unique(d_c)) < K:
            return False
            
        # 2. Check base cases: d(c,i) != d(i,j) for all i, j
        # D_sub[i,j] = dist(i, current_indices[j])
        D_sub = dist[np.ix_(current_indices, current_indices)]
        # Compare D_sub[i,j] with d_c[i]
        # d_c is (K,). d_c[:, None] is (K, 1).
        # D_sub is (K, K).
        # D_sub == d_c[:, None] broadcasts to (K, K).
        # Element (i,j) is True if dist(i, current_indices[j]) == dist(candidate, current_indices[i])
        # This covers the case where apex is at current_indices[i].
        if np.any(D_sub == d_c[:, None]):
            return False
            
        return True

    def greedy_random():
        """Greedily add points in random order using vectorized check."""
        current_mask = np.zeros(total, dtype=bool)
        current_indices = []
        remaining = np.arange(total)
        random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline:
                break
            if can_add_vectorized(idx, current_indices):
                current_mask[idx] = True
                current_indices.append(idx)
        return list(current_indices)

    def local_search(initial_indices, time_budget):
        """
        Local search: try to swap out a point and swap in a better one, or just add more.
        Since we start with a greedy solution, it's often "maximal" (can't add any more).
        So we need swaps.
        
        Strategy:
        1. Pick a random point to remove.
        2. Try to find a point to add that fits.
        3. If found, accept.
        4. Repeat.
        """
        current = list(initial_indices)
        current_mask = np.zeros(total, dtype=bool)
        for idx in current:
            current_mask[idx] = True
            
        best = list(current)
        best_size = len(current)
        end_time = time.time() + time_budget
        
        # Precompute list of all indices for sampling
        all_indices = np.arange(total)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Try a batch of random swaps
            n_attempts = 50
            for _ in range(n_attempts):
                if time.time() > end_time:
                    break
                if not current:
                    break
                    
                # Pick a point to remove
                i = random.randrange(len(current))
                idx_remove = current[i]
                
                # Temporarily remove it
                temp_indices = [x for x in current if x != idx_remove]
                
                # Try to find a candidate to add
                # Sample some candidates from the remaining points
                # To be efficient, we can sample from all points not in current
                # But checking if a point is in current_mask is O(1)
                
                # Heuristic: Sample 100 random points
                candidates = random.sample(range(total), 100)
                
                found = False
                for cand in candidates:
                    if current_mask[cand]:
                        continue
                    if can_add_vectorized(cand, temp_indices):
                        # Accept swap
                        current[i] = cand
                        current_mask[idx_remove] = False
                        current_mask[cand] = True
                        improved = True
                        found = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best = list(current)
                        break
                        
                if found:
                    # After a successful swap, try to add more points greedily
                    # This might increase size
                    remaining_pts = [j for j in range(total) if not current_mask[j]]
                    random.shuffle(remaining_pts)
                    for add in remaining_pts[:20]: # Limit to 20 tries to save time
                        if time.time() > end_time:
                            break
                        if can_add_vectorized(add, current):
                            current_mask[add] = True
                            current.append(add)
                            if len(current) > best_size:
                                best_size = len(current)
                                best = list(current)
                    break # Restart the outer loop logic or just continue swapping
                    
        return best

    # Phase 1: Multiple greedy restarts
    best_indices = []
    best_size = 0
    n_restarts = 0
    
    # Estimate time per greedy run to decide number of restarts
    # Greedy run is O(N^2 * K * check_cost). Check cost is O(K^2) vectorized.
    # So O(N^2 * K^3) roughly? No, K grows.
    # Let's just run until deadline - 60s.
    
    while time.time() < deadline - 60:
        n_restarts += 1
        result = greedy_random()
        if len(result) > best_size:
            best_size = len(result)
            best_indices = result
            
        # If we found a very large set, we might break early? 
        # Max possible is ~ N^2 / something. 
        # For now, keep going.

    # Phase 2: Local search on best solution
    if best_size > 0 and time.time() < deadline - 20:
        remaining_time = max(10, deadline - 20 - time.time())
        improved_set = local_search(best_indices, remaining_time)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_indices = improved_set

    # Phase 3: Final greedy pass to see if we can add any more
    if time.time() < deadline - 5:
        current_mask = np.zeros(total, dtype=bool)
        for idx in best_indices:
            current_mask[idx] = True
        remaining_pts = [j for j in range(total) if not current_mask[j]]
        random.shuffle(remaining_pts)
        for idx in remaining_pts:
            if time.time() > deadline - 5:
                break
            if can_add_vectorized(idx, best_indices):
                current_mask[idx] = True
                best_indices.append(idx)

    # Convert indices to (x, y) tuples
    result = [(int(xs[idx]), int(ys[idx])) for idx in best_indices]
    return result

if __name__ == "__main__":
    # Quick self-test
    pts = construct()
    print(f"Found {len(pts)} points")
