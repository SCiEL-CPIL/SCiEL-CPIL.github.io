"""No-isosceles set construction on [0, N)^2 grid."""
import time
import random
import math
import numpy as np
from collections import Counter

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    # Pre-compute all grid points and indices
    all_points = [(x, y) for x in range(N) for y in range(N)]
    M = len(all_points)
    
    # Pre-compute squared distance matrix
    coords = np.array(all_points)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    D = (dx * dx + dy * dy).astype(np.uint16)
    
    # Pre-compute counters for each row for fast conflict counting
    # D_counters[i] is a Counter of distances from point i to all points
    D_counters = [Counter(D[i]) for i in range(M)]
    
    def get_conflicts(p, S):
        """
        Calculate the number of isosceles triangles formed by adding point p to set S.
        S is a list of indices.
        """
        if not S:
            return 0
            
        # Distances from p to all points in S
        d_pa = D[p, S]
        
        # 1. Triangles with apex at p: d(p, a) == d(p, b)
        # Count duplicates in d_pa
        # Since d_pa is an array, we can use a set to find unique count
        # But d_pa can be large.
        # Using Counter is O(N)
        c_pa = Counter(d_pa)
        conf_apex_p = sum((count - 1) for count in c_pa.values() if count > 1)
        
        # 2. Triangles with apex at a in S: d(p, a) == d(a, b)
        # For each a in S, check how many b in S satisfy d(a, b) == d(p, a)
        # This is D_counters[a][d(p, a)]
        # Note: d(p, a) is the distance.
        # We need to sum these counts for all a in S.
        
        conf_apex_other = 0
        # d_pa is a numpy array of uint16.
        # We iterate over indices in S and corresponding distances
        for i, a in enumerate(S):
            dist_val = d_pa[i]
            # Count how many points b in S are at this distance from a
            # D_counters[a] contains counts for all points in the grid, 
            # but we only care about b in S.
            # Wait. D_counters[a][dist_val] counts ALL points b in the grid such that d(a,b)=dist_val.
            # We only want b in S.
            # So this approach is WRONG if S is a subset.
            # We need to count b in S such that d(a,b) == dist_val.
            
            # This requires checking each b in S.
            # This makes it O(|S|^2) again.
            # Is there a faster way?
            # If |S| is small, O(|S|^2) is fine.
            # Let's revert to O(|S|^2) but optimized.
            pass
            
        # Let's re-implement get_conflicts with O(|S|^2) but careful constants.
        
        # 1. Apex at p
        # d_pa is array of length K
        # conf_apex_p = K - len(np.unique(d_pa))
        conf_apex_p = len(d_pa) - len(np.unique(d_pa))
        
        # 2. Apex at a in S
        # For each a in S, we want count of b in S such that d(a,b) == d(p,a)
        # Let d_val = d(p,a)
        # We can pre-compute d_a_s = D[a, S] for all a in S?
        # That is a KxK matrix.
        # Then for each row i (corresponding to a_i), we want count of d_val_i in row i.
        
        # Let's extract the submatrix D_S = D[np.array(S)][:, np.array(S)]
        # And d_pa = D[p, S]
        
        S_arr = np.array(S)
        D_S = D[S_arr][:, S_arr]
        
        # For each i, d_val = d_pa[i]
        # Count of d_val in D_S[i, :]
        # Note: D_S[i, i] is 0. d_val > 0.
        
        conf_apex_other = 0
        for i in range(len(S)):
            d_val = d_pa[i]
            # Count occurrences of d_val in D_S[i, :]
            # np.sum(D_S[i, :] == d_val)
            conf_apex_other += np.sum(D_S[i, :] == d_val)
            
        return conf_apex_p + conf_apex_other

    # This is O(K^2) with numpy overhead in the loop.
    # For K=20, 400 ops. 100k steps -> 40M ops.
    # NumPy sum on small arrays is slow due to overhead.
    # Let's try to vectorize the "Apex at a" part.
    
    # D_S is KxK.
    # d_pa is Kx1.
    # We want sum over i of (D_S[i, :] == d_pa[i]).sum()
    # This is sum of diagonal elements of (D_S == d_pa[:, None])
    
    def get_conflicts_fast(p, S):
        if not S:
            return 0
        S_arr = np.array(S)
        d_pa = D[p, S_arr]
        
        # Apex at p
        conf_apex_p = len(d_pa) - len(np.unique(d_pa))
        
        # Apex at a in S
        D_S = D[S_arr][:, S_arr]
        # Comparison matrix: D_S[i, j] == d_pa[i]
        # We want sum over i of count of True in row i
        # Which is sum of (D_S == d_pa[:, None]).sum(axis=1)
        # Or just .sum() of the whole boolean matrix?
        # Yes, each True corresponds to a pair (a_i, b_j) such that d(a_i, b_j) == d(p, a_i).
        # This forms an isosceles triangle (p, a_i, b_j) with apex a_i.
        # So we just sum all True values.
        
        # Note: Does this count the triangle twice?
        # Triangle (p, a, b).
        # Apex a: d(p,a) == d(a,b).
        # Apex b: d(p,b) == d(b,a).
        # These are distinct conditions.
        # If d(p,a) == d(a,b) AND d(p,b) == d(b,a), it's an equilateral triangle (if d(p,a)=d(p,b)=d(a,b)).
        # In that case, it's counted in conf_apex_p (apex p) AND in conf_apex_other (apex a and apex b).
        # Is that correct?
        # An equilateral triangle is an isosceles triangle.
        # We just need to count the number of isosceles triangles.
        # If a triangle is equilateral, it is 1 isosceles triangle.
        # But my counting method counts:
        # - Apex p: 1
        # - Apex a: 1
        # - Apex b: 1
        # Total 3.
        # This is overcounting.
        
        # We need to count DISTINCT triangles.
        # A triangle is defined by the set of 3 points {p, a, b}.
        # It is isosceles if ANY of the 3 apex conditions hold.
        
        # This makes the "conflict count" harder to define for soft constraints if we just sum conditions.
        # However, for the purpose of "is it valid?", if count > 0, it's invalid.
        # For optimization, we want to minimize the number of isosceles triangles.
        # If we overcount equilateral triangles, the penalty is heavier, which is fine for search.
        # It just means the landscape is steeper.
        
        # Let's stick to the sum of conditions. It's a valid heuristic for "badness".
        
        # Vectorized count:
        # D_S == d_pa[:, None] creates KxK boolean matrix.
        # .sum() gives total matches.
        
        # Memory: KxK boolean. For K=100, 10k bytes. Fine.
        
        matches = (D_S == d_pa[:, None]).sum()
        
        return conf_apex_p + matches

    # Local search / SA
    def simulated_annealing():
        # Start with a small random valid set?
        # Or just random set?
        # Let's start with random set of size 10.
        initial_size = 10
        current_set = random.sample(range(M), initial_size)
        current_conflicts = get_conflicts_fast(current_set[0], current_set[1:]) # This is not total conflicts
        # We need total conflicts of the set.
        # Total conflicts = sum of conflicts introduced by each point?
        # No, that's not additive in a simple way because of double counting.
        # However, for SA, we can just recalculate total conflicts from scratch?
        # That would be O(K^3) or O(K^2 * K) = O(K^3).
        # For K=20, 8000 ops. 100k steps -> 800M ops. Too slow.
        
        # We need to maintain total_conflicts incrementally.
        # When adding p: new_conflicts = old_conflicts + get_conflicts_fast(p, S)
        # When removing p: new_conflicts = old_conflicts - get_conflicts_fast(p, S \ {p})
        
        # So we need a function to calculate conflicts of p given S.
        # get_conflicts_fast(p, S) does this.
        
        # Let's implement the SA loop with incremental updates.
        
        S = list(current_set)
        in_set = set(S)
        
        # Calculate initial total conflicts
        # Sum of get_conflicts_fast(p, S \ {p}) for all p in S?
        # This is O(K^3) initially. Acceptable for K=10.
        total_conflicts = 0
        for p in S:
            S_minus_p = [x for x in S if x != p]
            total_conflicts += get_conflicts_fast(p, S_minus_p)
            # Note: This counts each triangle multiple times (once for each vertex involved in the isosceles condition).
            # As discussed, this is a valid "badness" metric.
            
        best_set = list(S)
        best_conflicts = total_conflicts
        best_size = len(S)
        
        T = 100.0
        cooling = 0.999
        steps = 50000
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Choose move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                candidates = [i for i in range(M) if i not in in_set]
                if not candidates: continue
                p = random.choice(candidates)
                delta_c = get_conflicts_fast(p, S)
                delta_f = 10000 - delta_c # Size +1, Conflicts +delta_c
                # Fitness = 10000*Size - 1000*Conflicts?
                # Let's use Fitness = 10000*Size - Conflicts
                # delta_f = 10000 - delta_c
                
                if delta_f > 0 or random.random() < math.exp(delta_f / T):
                    S.append(p)
                    in_set.add(p)
                    total_conflicts += delta_c
                    if total_conflicts == 0 and len(S) > best_size:
                        best_set = list(S)
                        best_size = len(S)
                        best_conflicts = 0
                    elif total_conflicts < best_conflicts:
                        best_set = list(S)
                        best_size = len(S)
                        best_conflicts = total_conflicts
                        
            elif move_type == 'remove':
                if not S: continue
                p = random.choice(S)
                S_minus_p = [x for x in S if x != p]
                delta_c = get_conflicts_fast(p, S_minus_p)
                # Removing p decreases conflicts by delta_c
                # Size decreases by 1
                # delta_f = -10000 + delta_c
                delta_f = delta_c - 10000
                
                if delta_f > 0 or random.random() < math.exp(delta_f / T):
                    S.remove(p)
                    in_set.discard(p)
                    total_conflicts -= delta_c
                    if total_conflicts == 0 and len(S) > best_size:
                        best_set = list(S)
                        best_size = len(S)
                        best_conflicts = 0
                    elif total_conflicts < best_conflicts:
                        best_set = list(S)
                        best_size = len(S)
                        best_conflicts = total_conflicts
                        
            else: # swap
                if not S: continue
                p = random.choice(S)
                candidates = [i for i in range(M) if i not in in_set]
                if not candidates: continue
                q = random.choice(candidates)
                
                S_minus_p = [x for x in S if x != p]
                # Conflicts caused by p in S
                c_p = get_conflicts_fast(p, S_minus_p)
                # Conflicts caused by q in S_minus_p
                c_q = get_conflicts_fast(q, S_minus_p)
                
                # New total conflicts = total_conflicts - c_p + c_q
                # Size unchanged
                # delta_f = - (c_q - c_p) = c_p - c_q
                delta_f = c_p - c_q
                
                if delta_f > 0 or random.random() < math.exp(delta_f / T):
                    # Perform swap
                    S[S.index(p)] = q
                    in_set.discard(p)
                    in_set.add(q)
                    total_conflicts = total_conflicts - c_p + c_q
                    if total_conflicts == 0 and len(S) > best_size:
                        best_set = list(S)
                        best_size = len(S)
                        best_conflicts = 0
                    elif total_conflicts < best_conflicts:
                        best_set = list(S)
                        best_size = len(S)
                        best_conflicts = total_conflicts
                        
            T *= cooling
            
        return best_set

    best_overall = []
    best_size_overall = 0
    
    num_restarts = 5 # SA is slower, fewer restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        random.seed(restart * 12345 + 67890)
        
        result = simulated_annealing()
        if len(result) > best_size_overall:
            # Verify validity?
            # SA might end with conflicts > 0.
            # We should only accept if conflicts == 0?
            # Or repair?
            # Let's check if it's valid.
            # For now, assume best_set tracked is the best.
            # But best_set might have conflicts > 0.
            # We want a VALID set.
            # So we should only update best_overall if the set is valid.
            
            # Check validity of result
            # Re-calculate conflicts for the whole set
            # This is O(K^3)
            valid = True
            if len(result) > 0:
                total_c = 0
                res_list = list(result)
                for p in res_list:
                    S_minus_p = [x for x in res_list if x != p]
                    total_c += get_conflicts_fast(p, S_minus_p)
                if total_c != 0:
                    valid = False
            
            if valid and len(result) > best_size_overall:
                best_overall = result
                best_size_overall = len(result)
                
    return [all_points[i] for i in best_overall]
