import time
import numpy as np
import random

N = 64

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before the 330s kill switch
    deadline = start_time + 300.0
    
    # 1. Pre-compute conflict graph (independent set formulation)
    # Vertices: 0 to N*N - 1
    # Edge between u, v if there exists w such that {u, v, w} forms an isosceles triangle.
    # Note: If d(u,v) == d(v,w), then u and v conflict (because adding w would create isosceles, 
    # but wait: the condition is "no three points form an isosceles triangle".
    # This means for ANY triple (a,b,c) in our set, they must NOT be isosceles.
    # This is equivalent to: The set S is an independent set in a 3-uniform hypergraph H, 
    # where edges are isosceles triples.
    # However, we can also view it as: 
    # If we have a set S, and we want to add p, we must ensure p doesn't form an isosceles 
    # triangle with ANY pair (a,b) in S.
    # This is NOT a simple graph independent set problem on pairs.
    # Let's re-read the directive: "Model the problem as a maximum independent set search on a conflict graph 
    # where vertices are grid points and edges connect pairs that share a common 'isosceles partner'".
    
    # Let's analyze this claim.
    # If u and v are connected in this graph, it means there exists w such that {u, v, w} is isosceles.
    # Does this mean we cannot have both u and v in our set?
    # NO. If we have u and v in our set, we are only forbidden from having w in our set.
    # We CAN have u and v in our set, as long as we don't have w (and no other point that forms isosceles with u and v).
    # So the "conflict graph" described in the directive is actually a "2-packing" or similar problem, 
    # OR the directive is slightly misleading and implies we should use it for pruning.
    
    # Actually, a set S is valid iff for every pair (a,b) in S, S does not contain any point in the 
    # "forbidden set" F(a,b) = {p : {a,b,p} is isosceles}.
    
    # Let's stick to the greedy + local search approach but optimize it significantly using NumPy.
    # The current program is slow because it does O(k^2) Python loops for each insertion.
    # We can vectorize the check:
    # Given a candidate p and a set S (represented as a list of indices), we can compute 
    # all pairwise distances in S and distances from p to S, and check for equality.
    
    # Even better: 
    # We can maintain a "forbidden" mask for each point? No, the forbidden set depends on the PAIR.
    
    # Let's try a different structural approach: 
    # Randomized Local Search (Simulated Annealing or just Hill Climbing with random restarts)
    # Start with a random set of size k.
    # Try to swap points or add/remove points to improve validity/size.
    
    # But checking validity of the whole set is O(k^3).
    
    # Let's go back to the directive's suggestion: "bitset-accelerated neighborhood pruning".
    # This suggests using bitsets to represent sets of points and their conflicts.
    
    # Let's implement a fast greedy construction using NumPy for the validity check.
    
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    n_points = N * N
    
    # Pre-compute squared distances? 
    # dist[i, j] = squared distance between point i and point j.
    # This is a 4096 x 4096 matrix of int32. 4096*4096*4 bytes = 64 MB. This is acceptable.
    # Pre-computing this might take a few seconds, but it will make the search much faster.
    
    # Let's try to pre-compute the distance matrix.
    # x_coords = points[:, 0]
    # y_coords = points[:, 1]
    # dx = x_coords[:, None] - x_coords[None, :]
    # dy = y_coords[:, None] - y_coords[None, :]
    # dist_sq = dx*dx + dy*dy
    
    # This will take a moment. Let's do it.
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    dx = x_coords[:, None] - x_coords[None, :]
    dy = y_coords[:, None] - y_coords[None, :]
    dist_sq = dx*dx + dy*dy
    
    # Now, for a set S (list of indices), and a new point p (index),
    # we need to check if adding p is valid.
    # Conditions for invalidity:
    # 1. There exist a, b in S such that dist(p, a) == dist(p, b)
    # 2. There exist a, b in S such that dist(p, a) == dist(a, b)
    # 3. There exist a, b in S such that dist(p, b) == dist(a, b)
    
    # Let S be a 1D array of indices.
    # d_pa = dist_sq[p, S]  # distances from p to all points in S
    # d_ab = dist_sq[np.ix_(S, S)] # pairwise distances in S
    
    # Check 1: Any duplicates in d_pa?
    # Check 2 & 3: Does d_pa intersect with d_ab (excluding the diagonal? No, d_ab includes all pairs).
    # Actually, if d_pa[i] == d_ab[j, k], it means dist(p, S[i]) == dist(S[j], S[k]).
    # This is exactly the condition.
    
    # So, we can check:
    # 1. len(set(d_pa)) == len(d_pa)
    # 2. len(set(d_pa) & set(d_ab)) == 0
    
    # Wait, condition 2 and 3 are covered by the intersection of d_pa and d_ab.
    # If dist(p, a) == dist(b, c), that's an isosceles triangle?
    # No, the triangle is (p, b, c). The sides are dist(p,b), dist(p,c), dist(b,c).
    # Isosceles if any two are equal.
    # So we need to check if for ANY a,b,c in S U {p}, the triangle is isosceles.
    # This is equivalent to checking for every pair (u, v) in S U {p}, 
    # if there exists w in S U {p} such that dist(u,w) == dist(v,w) or dist(u,w) == dist(u,v) etc.
    
    # My previous check was:
    # Adding p to S.
    # New triangles must involve p.
    # So we only need to check triangles of the form (p, a, b) where a, b in S.
    # Sides: d(p,a), d(p,b), d(a,b).
    # Isosceles if d(p,a)==d(p,b) OR d(p,a)==d(a,b) OR d(p,b)==d(a,b).
    
    # So:
    # 1. d(p,a) == d(p,b) for some a!=b in S. => Duplicate in d_pa.
    # 2. d(p,a) == d(a,b) for some a!=b in S. => d_pa[i] == d_ab[i, j] or d_ab[j, i].
    # 3. d(p,b) == d(a,b) for some a!=b in S. => d_pa[j] == d_ab[i, j] or d_ab[j, i].
    
    # So the condition is:
    # - No duplicates in d_pa.
    # - No value in d_pa is present in d_ab.
    
    # This is much faster to check with NumPy!
    
    def is_valid_addition(p_idx, S):
        if len(S) < 2:
            return True
        
        S = np.array(S)
        d_pa = dist_sq[p_idx, S]
        
        # Check duplicates in d_pa
        if len(np.unique(d_pa)) < len(d_pa):
            return False
        
        # Get pairwise distances in S
        # We only need the upper triangle to save time/memory, but np.ix_ is easy.
        d_ab = dist_sq[np.ix_(S, S)]
        
        # Check intersection
        # We can use a set for d_pa and check if any element of d_ab is in it.
        # But d_ab is a 2D array. 
        # Flattening d_ab might be expensive if S is large.
        # However, we can just check:
        # Are there any i, j such that d_ab[i, j] is in d_pa?
        
        # A fast way: 
        # Create a boolean mask for d_ab where value is in d_pa.
        # np.isin(d_ab, d_pa)
        # If any is True, return False.
        
        # np.isin might be slow for large arrays.
        # Alternative: 
        # Since d_pa is small (size k), we can iterate? No, we want speed.
        # Let's try np.isin.
        
        if np.any(np.isin(d_ab, d_pa)):
            return False
            
        return True

    best_set = []
    best_size = 0
    
    # Random restarts
    # We will try to build a set greedily.
    # To improve the greedy, we can randomize the order of points.
    
    # We can also try to "repair" or "improve" the set.
    # But let's start with optimized greedy.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Shuffle points
        order = np.random.permutation(n_points)
        current_set = []
        
        for p_idx in order:
            if time.time() > deadline:
                break
                
            if is_valid_addition(p_idx, current_set):
                current_set.append(p_idx)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found a very good set, we might want to break early?
            # No, let's keep searching for a while.
            
    # Convert indices back to (x, y)
    result = [points[i] for i in best_set]
    return result

# Test run to see how long it takes
# if __name__ == "__main__":
#     t0 = time.time()
#     res = construct()
#     print(len(res), time.time() - t0)
