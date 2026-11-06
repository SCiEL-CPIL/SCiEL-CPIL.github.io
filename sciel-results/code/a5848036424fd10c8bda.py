import numpy as np
import time
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
# Precompute squared distances
coords = np.array(POINTS)
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

def construct():
    start_time = time.time()
    deadline = 280  # Safety margin for 300s limit
    
    best_set = None
    best_size = float('inf')
    
    # Strategy:
    # 1. Fast vectorized greedy construction to find a small maximal set.
    # 2. Local search (removal) to try to shrink the set while maintaining maximality.
    #    A set S is maximal if for every p not in S, S U {p} is NOT no-isosceles.
    #    Equivalently, for every p not in S, there exists a pair (a, b) in S such that
    #    (a, b, p) forms an isosceles triangle.
    #    If we remove a point q from S, S' = S \ {q}.
    #    S' is maximal if for every p not in S', S' U {p} is not no-isosceles.
    #    Note that p can be q or any other point not in S.
    #    If p = q, we just need S' U {q} = S to be no-isosceles (which it is).
    #    Wait, the condition for maximality is: adding ANY point p not in S' creates an isosceles triangle.
    #    If we remove q, the new set is S'. The points not in S' are (not in S) U {q}.
    #    For p in (not in S): S U {p} was not no-isosceles (since S was maximal).
    #    Does S' U {p} remain not no-isosceles?
    #    The isosceles triangle in S U {p} might have involved q. If it did, removing q might break it.
    #    So we must check if S' is maximal.
    
    def is_no_isosceles(S_indices):
        """Check if set of indices is no-isosceles."""
        if len(S_indices) < 3:
            return True
        S_arr = np.array(S_indices)
        D_S = D[np.ix_(S_arr, S_arr)]
        # Upper triangle
        i, j = np.triu_indices(len(S_arr), k=1)
        d_ik = D_S[i, j]
        d_jk = D_S[i, j] # Wait, need to be careful with indexing
        # Let's just check all triples? No, that's slow.
        # Vectorized check:
        # For each pair (i, j), check if there exists k such that d(i,k) == d(i,j) or d(j,k) == d(i,j) or d(i,k) == d(j,k)
        # Actually, simpler:
        # For each point i, distances to others must be unique. (Catches vertex at i)
        # And for each pair (i, j), d(i,j) must not equal d(i,k) or d(j,k) for any k != i, j.
        
        # Check 1: Vertex at i. Distances from i to all others must be unique.
        # D_S rows are distances from row i to all others.
        # Exclude self (0).
        for idx in range(len(S_arr)):
            dists = D_S[idx, :]
            dists = np.delete(dists, idx)
            if len(np.unique(dists)) < len(dists):
                return False
        
        # Check 2 & 3: Vertex at i or j, base at k? No.
        # Isosceles triangle (a, b, c) means two sides equal.
        # d(a,b) == d(a,c) -> vertex a. Covered by Check 1 if a is the vertex.
        # d(a,b) == d(b,c) -> vertex b. Covered by Check 1 if b is the vertex.
        # d(a,c) == d(b,c) -> vertex c. Covered by Check 1 if c is the vertex.
        
        # Wait, does Check 1 cover ALL cases?
        # If d(a,b) == d(a,c), then from perspective of a, distances to b and c are equal. So Check 1 fails for a.
        # If d(a,b) == d(b,c), then from perspective of b, distances to a and c are equal. So Check 1 fails for b.
        # If d(a,c) == d(b,c), then from perspective of c, distances to a and b are equal. So Check 1 fails for c.
        
        # So Check 1 is sufficient to detect ANY isosceles triangle?
        # Yes. An isosceles triangle has a vertex where two sides are equal.
        # That vertex will see two points at the same distance.
        # So if NO point sees two others at the same distance, there are NO isosceles triangles.
        
        return True

    def is_maximal(S_indices, all_indices):
        """Check if S is maximal. S is maximal if for all p not in S, S U {p} is NOT no-isosceles."""
        S_set = set(S_indices)
        for p in all_indices:
            if p in S_set:
                continue
            # Check if S U {p} is no-isosceles
            # If it IS no-isosceles, then S is NOT maximal (we could add p)
            # So we need to check if S U {p} creates an isosceles triangle.
            # This is equivalent to: does p form an isosceles triangle with any pair in S?
            # Or does p create equal distances from some point in S to two other points in S?
            
            # Efficient check:
            # 1. Distances from p to all points in S must be unique? No, that's for S U {p} being no-isosceles.
            #    If S U {p} is no-isosceles, then:
            #    a) Distances from p to S are unique.
            #    b) For any i, j in S, d(i,j) != d(p,i) and d(i,j) != d(p,j).
            #    c) S itself is no-isosceles (already true).
            
            # So S is NOT maximal if there exists p not in S such that:
            #    - Distances from p to S are all unique.
            #    - For all i, j in S, d(i,j) != d(p,i) and d(i,j) != d(p,j).
            
            # We want S to BE maximal, so for ALL p not in S, the above condition must FAIL.
            # i.e., for all p not in S, either:
            #    - Distances from p to S are NOT unique (some d(p,i) == d(p,j)).
            #    - OR there exists i, j in S such that d(i,j) == d(p,i) or d(i,j) == d(p,j).
            
            S_list = list(S_indices)
            if len(S_list) == 0:
                continue
                
            d_p_s = D[p, S_list]
            
            # Condition 1: d_p_s has duplicates?
            if len(np.unique(d_p_s)) < len(d_p_s):
                continue # This p cannot be added (creates isosceles with vertex p). S is "safe" for this p.
                
            # Condition 2: d(i,j) == d(p,i) or d(i,j) == d(p,j) for some i, j in S?
            # D_S is distances between points in S.
            D_S = D[np.ix_(S_list, S_list)]
            
            # Check if any value in D_S (excluding diagonal) matches any value in d_p_s
            # d_p_s is distances from p to S_list.
            # d_p_s[i] is distance p to S_list[i].
            # D_S[i, j] is distance S_list[i] to S_list[j].
            
            # We need to check if D_S[i, j] == d_p_s[i] (vertex at S_list[i])
            # or D_S[i, j] == d_p_s[j] (vertex at S_list[j])
            
            # Vectorized:
            # D_S == d_p_s[:, None] -> matches d_p_s[i] for all j?
            # D_S == d_p_s[None, :] -> matches d_p_s[j] for all i?
            
            # We only care about i != j.
            # But if D_S[i, i] == 0, and d_p_s[i] is never 0 (p not in S), it won't match 0 unless d_p_s[i]==0.
            
            if np.any(D_S == d_p_s[:, None]):
                continue # This p cannot be added (creates isosceles with vertex in S). S is "safe" for this p.
                
            if np.any(D_S == d_p_s[None, :]):
                continue # This p cannot be added (creates isosceles with vertex in S). S is "safe" for this p.
                
            # If we reach here, p CAN be added. S is NOT maximal.
            return False
            
        return True

    def greedy_construction(perm):
        S = []
        S_set = set()
        
        for p_idx in perm:
            if len(S) >= best_size:
                break
            
            if len(S) == 0:
                S.append(p_idx)
                S_set.add(p_idx)
                continue
            if len(S) == 1:
                S.append(p_idx)
                S_set.add(p_idx)
                continue
                
            S_list = list(S_set)
            d_p_s = D[p_idx, S_list]
            
            # Check 1: Distances from P to S are unique
            if len(np.unique(d_p_s)) < len(d_p_s):
                continue
                
            # Check 2 & 3
            D_S = D[np.ix_(S_list, S_list)]
            
            if np.any(D_S == d_p_s[:, None]):
                continue
            
            if np.any(D_S == d_p_s[None, :]):
                continue
            
            S.append(p_idx)
            S_set.add(p_idx)
            
        return S

    # Phase 1: Random Greedy Restarts
    while time.time() - start_time < deadline - 60:
        perm = np.random.permutation(N_POINTS)
        S = greedy_construction(perm)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = [POINTS[i] for i in S]
            
            # Try to improve with local search
            S_indices = list(S)
            all_indices = list(range(N_POINTS))
            
            improved = True
            while improved and time.time() - start_time < deadline - 10:
                improved = False
                # Try removing each point
                for i in range(len(S_indices)):
                    new_S = S_indices[:i] + S_indices[i+1:]
                    if len(new_S) >= best_size:
                        continue
                    
                    # Check if new_S is no-isosceles (it is, since subset of no-isosceles)
                    # Check if new_S is maximal
                    if is_maximal(new_S, all_indices):
                        S_indices = new_S
                        best_size = len(S_indices)
                        best_set = [POINTS[idx] for idx in S_indices]
                        improved = True
                        break
                        
    # Phase 2: If we have time, do more random restarts or refine
    if best_set is None:
        # Fallback
        perm = np.random.permutation(N_POINTS)
        S = greedy_construction(perm)
        best_set = [POINTS[i] for i in S]
        
    return best_set
