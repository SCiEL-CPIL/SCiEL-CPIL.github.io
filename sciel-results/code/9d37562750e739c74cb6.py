import numpy as np
import time
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
coords = np.array(POINTS)

# Precompute squared distances matrix D
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    best_size = float('inf')
    best_set_indices = []
    
    # Helper function to check if a point can be added to a set
    # S_indices: list of indices of points in the current set
    # p_idx: index of the candidate point
    def can_add(p_idx, S_indices):
        if len(S_indices) < 2:
            return True
            
        S_arr = np.array(S_indices)
        
        # 1. Distances from p to all points in S
        d_p_s = D[p_idx, S_arr]
        
        # Check 1: Isosceles with vertex at p
        # If any distance from p to S is equal to another distance from p to S, 
        # then p, s_i, s_j form an isosceles triangle with vertex p.
        # This happens if d_p_s has duplicates.
        if len(np.unique(d_p_s)) < len(d_p_s):
            return False
            
        # 2 & 3. Isosceles with vertex at s_i or s_j
        # We need to check if d(p, s_i) == dist(s_i, s_j) or d(p, s_j) == dist(s_i, s_j)
        # for any pair (i, j) in S.
        
        # Get the submatrix of distances between points in S
        # D_S[i, j] = dist(S[i], S[j])
        D_S = D[np.ix_(S_arr, S_arr)]
        
        # d_p_s is shape (K,)
        # D_S is shape (K, K)
        
        # Condition A: d(p, s_i) == dist(s_i, s_j)
        # This corresponds to D_S[i, j] == d_p_s[i]
        # We can check this by comparing D_S columns with d_p_s?
        # Actually, for each i, we check if D_S[i, :] contains d_p_s[i].
        # Or more efficiently: 
        # Create a mask where D_S[i, j] == d_p_s[i]
        # Note: D_S[i, i] = 0, d_p_s[i] > 0 (since p not in S), so diagonal won't match unless d_p_s[i]=0 which is impossible.
        
        # Vectorized check for Condition A:
        # d_p_s[:, None] is (K, 1), D_S is (K, K)
        # We want to know if any element in row i of D_S equals d_p_s[i]
        # match_A = (D_S == d_p_s[:, None]).any(axis=1)
        # If any match_A is True, return False.
        
        # Vectorized check for Condition B:
        # d(p, s_j) == dist(s_i, s_j)
        # This corresponds to D_S[i, j] == d_p_s[j]
        # match_B = (D_S == d_p_s[None, :]).any(axis=0)
        # If any match_B is True, return False.
        
        # Combine:
        # Note: D_S is symmetric.
        # If D_S[i, j] == d_p_s[i], it's an isosceles triangle (p, s_i, s_j) with vertex s_i.
        # If D_S[i, j] == d_p_s[j], it's an isosceles triangle (p, s_i, s_j) with vertex s_j.
        
        # We can just check if any element in D_S matches the corresponding d_p_s value.
        # Let's do it in one go to save overhead.
        
        # Create a boolean matrix for A
        # A[i, j] is True if D_S[i, j] == d_p_s[i]
        # B[i, j] is True if D_S[i, j] == d_p_s[j]
        
        # Optimization: 
        # Instead of creating two full boolean matrices, we can check:
        # 1. Does any row i of D_S contain d_p_s[i]?
        # 2. Does any col j of D_S contain d_p_s[j]?
        
        # Since D_S is symmetric, checking rows is sufficient for "vertex at s_i" type?
        # No.
        # Triangle (p, u, v).
        # Case 1: pu = pv. (Vertex p). Checked by unique(d_p_s).
        # Case 2: pu = uv. (Vertex u). d_p_s[u] == D_S[u, v].
        # Case 3: pv = uv. (Vertex v). d_p_s[v] == D_S[u, v].
        
        # So we need to check if for any pair (u, v) in S:
        # d_p_s[u] == D_S[u, v] OR d_p_s[v] == D_S[u, v]
        
        # This is equivalent to checking if the set of values {d_p_s[u] for u in S} 
        # intersects with the set of values {D_S[u, v] for u, v in S, u!=v}
        # Wait, no. The indices matter.
        # d_p_s[u] must equal D_S[u, v] for the SAME u.
        
        # Let's use the vectorized approach from inspiration 2 which worked well.
        
        # Check A: D_S[i, j] == d_p_s[i]
        # This is true if any column j in row i equals d_p_s[i].
        # We can flatten this check.
        
        # Efficient check:
        # match_a = (D_S == d_p_s[:, None]).any(axis=1)
        # if match_a.any(): return False
        
        # match_b = (D_S == d_p_s[None, :]).any(axis=0)
        # if match_b.any(): return False
        
        # To reduce overhead, we can combine:
        # If there is any (i,j) such that D_S[i,j] == d_p_s[i] OR D_S[i,j] == d_p_s[j]
        
        # Let's just do the two checks. For K < 50, this is very fast in NumPy.
        
        if (D_S == d_p_s[:, None]).any():
            return False
            
        if (D_S == d_p_s[None, :]).any():
            return False
            
        return True

    # Main search loop: Random Greedy Restarts
    # We try to construct a maximal set by adding points in a random order.
    # If at any point the size exceeds best_size, we abort.
    
    while time.time() - start_time < deadline:
        # Random permutation of all points
        perm = np.random.permutation(N_POINTS)
        
        S_indices = []
        
        for p_idx in perm:
            # Pruning: if we already have more points than our best, stop this run
            if len(S_indices) >= best_size:
                break
                
            if can_add(int(p_idx), S_indices):
                S_indices.append(int(p_idx))
                
        # After the loop, check if this set is better
        if len(S_indices) < best_size:
            best_size = len(S_indices)
            best_set_indices = S_indices
            
            # Optional: If we find a very small set (e.g., < 20), we could try to optimize further,
            # but for now, we just keep searching for smaller ones.
            
    # Convert best indices to (x, y) tuples
    result = [POINTS[i] for i in best_set_indices]
    
    return result

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found set of size {len(res)}")
