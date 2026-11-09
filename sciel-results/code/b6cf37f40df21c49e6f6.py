import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    N = 64
    M = N * N
    
    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute squared distances between all pairs
    # dist[i][j] = squared distance between points[i] and points[j]
    # Using numpy for speed
    coords = np.array(points, dtype=np.int32)
    
    # Compute distance matrix (M x M)
    # diff = coords[:, None, :] - coords[None, :, :]
    # dist = np.sum(diff**2, axis=-1)
    # This is 4096x4096x2 which is ~268MB. Let's try to be more memory efficient.
    # Actually, 4096*4096*4 bytes = 67MB for the distance matrix. It's fine.
    
    # However, accessing numpy arrays in a tight loop in Python can be slow.
    # Let's stick to pure Python sets but optimize the logic.
    
    # Strategy:
    # 1. Greedy construction with multiple random restarts.
    # 2. Local search (swap and add) to improve.
    
    # Precompute distance lookup for fast access
    # dist_map[(x1,y1,x2,y2)] = squared distance
    # Actually, let's precompute distance from each point to all others
    # dist_from[i] = list of (j, dist_ij) for all j
    
    # For memory efficiency, let's compute distances on the fly but cache them
    # Actually, let's use a 2D array for distances
    dist_arr = np.zeros((M, M), dtype=np.int32)
    for i in range(M):
        x1, y1 = points[i]
        for j in range(M):
            x2, y2 = points[j]
            dist_arr[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    def try_add(S, p):
        """
        Try to add point p to set S.
        S is a set of indices.
        Returns True if added successfully, False otherwise.
        """
        # Check distance from p to all points in S
        # If any distance repeats, or if that distance already exists in S's neighbor map, fail.
        
        # We need to know if any two points in S are at the same distance from p.
        # AND if any point q in S has another point r in S at the same distance d(p,q).
        
        # Efficient check:
        # 1. Compute distances from p to all q in S.
        # 2. If any distance repeats among these, return False (p, q1, q2 is isosceles with apex p).
        # 3. For each unique distance d, check if there is a pair (q, r) in S such that dist(q,r) == d.
        #    This is the hard part. We need to know if S contains an isosceles triangle with base d?
        #    No, we need to know if adding p creates an isosceles triangle.
        #    The triangles involving p are (p, q, r).
        #    Isosceles conditions:
        #    a) dist(p,q) == dist(p,r)  -> Checked by step 2.
        #    b) dist(p,q) == dist(q,r)  -> For each q in S, check if there exists r in S such that dist(q,r) == dist(p,q).
        #    c) dist(p,r) == dist(q,r)  -> Symmetric to b.
        
        # So, for each q in S:
        #    d_pq = dist(p,q)
        #    We need to check if there exists r in S (r != q) such that dist(q,r) == d_pq.
        #    This is equivalent to: does q have a neighbor in S at distance d_pq?
        
        # To make this fast, we can precompute for each point in S, the set of distances to other points in S.
        # But S changes.
        
        # Alternative: 
        # Maintain for each point s in S, a set of distances to other points in S? 
        # That's O(|S|^2) memory and update time.
        
        # Let's use a simpler check for the greedy phase:
        # For candidate p:
        #   dists_p = {}
        #   for q in S:
        #       d = dist(p,q)
        #       if d in dists_p: return False
        #       dists_p[d] = q
        #   
        #   for q in S:
        #       d_pq = dist(p,q)
        #       # Check if any other r in S has dist(q,r) == d_pq
        #       # This requires iterating over S again? O(|S|^2)
        #       for r in S:
        #           if r == q: continue
        #           if dist(q,r) == d_pq:
        #               return False
        #   return True
        
        # This is O(|S|^2) per candidate.
        # If |S| ~ 50, |S|^2 = 2500.
        # Candidates ~ 4000.
        # Total ops ~ 10M. This is acceptable for a few iterations.
        
        # Let's implement this.
        
        x1, y1 = points[p]
        dists_from_p = {}
        
        for q in S:
            d = dist_arr[p, q]
            if d in dists_from_p:
                return False
            dists_from_p[d] = q
        
        # Now check condition b/c
        for q in S:
            d_pq = dist_arr[p, q]
            
            # Check if any r in S has dist(q,r) == d_pq
            for r in S:
                if r == q:
                    continue
                d_qr = dist_arr[q, r]
                if d_qr == d_pq:
                    return False
        
        return True

    def greedy_construct():
        S = set()
        candidates = list(range(M))
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if try_add(S, p):
                S.add(p)
        
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            # Try swaps: remove one, add one
            # Or just try to add points not in S
            # Let's try adding first
            candidates = [i for i in range(M) if i not in S]
            random.shuffle(candidates)
            
            for p in candidates:
                if time.time() > deadline - 10:
                    break
                if try_add(S, p):
                    S.add(p)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no addition, try a swap
            # Remove a random point, then try to add a different one
            if S_list:
                q_to_remove = random.choice(S_list)
                S.discard(q_to_remove)
                
                candidates = [i for i in range(M) if i not in S]
                random.shuffle(candidates)
                
                added = False
                for p in candidates:
                    if time.time() > deadline - 10:
                        break
                    if p == q_to_remove:
                        continue
                    if try_add(S, p):
                        S.add(p)
                        added = True
                        break
                
                if not added:
                    # Revert
                    S.add(q_to_remove)
                    # Try another removal? Or break?
                    # To avoid infinite loop, let's just break if no swap works in a few tries
                    # For now, break to restart greedy
                    break
            else:
                break
                
        return S

    best_S = set()
    
    # Multiple restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        S = greedy_construct()
        S = local_search(S)
        
        if len(S) > len(best_S):
            best_S = S.copy()
        
        # If we found a large set, we can reduce remaining restarts or time per restart
        # Simple heuristic: if size > 15, we are doing well.
        
    # Final polish on the best set found
    if time.time() < deadline - 5:
        best_S = local_search(best_S)

    return [points[i] for i in best_S]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
