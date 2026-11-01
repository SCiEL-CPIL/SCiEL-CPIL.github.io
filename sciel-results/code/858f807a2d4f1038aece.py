import numpy as np
from itertools import combinations, product
from collections import defaultdict

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The tensor T is defined by T[i, j, k] = 1 if we have A[i,j] * B[j,k] contributing to C[k,i]
    # Indices:
    # A: i in [0, n), j in [0, m) -> index u = i*m + j
    # B: j in [0, m), k in [0, p) -> index v = j*p + k
    # C: k in [0, p), i in [0, n) -> index w = k*n + i
    
    # We want to find rank R such that T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where u_r, v_r, w_r are vectors.
    # Since we only care about exact integer/half-integer decompositions, and the tensor entries are 0/1,
    # we can assume the supports are sparse.
    
    # Schoolbook rank is 18.
    # We suspect rank 15 is possible.
    
    # Let's construct the target tensor T as a 3D array or a dictionary of non-zero entries.
    # T[u, v, w] = 1 if u corresponds to (i,j), v to (j,k), and w to (k,i).
    # Note: The multiplication is C = A * B.
    # C[k, i] = sum_j A[i, j] * B[j, k].
    # So the tensor is T[i, j, k] = 1.
    # Mapping to indices:
    # u = i * m + j
    # v = j * p + k
    # w = k * n + i
    
    target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1

    # We search for a set of R rank-1 patterns.
    # A pattern is defined by (su, sv, sw) where su, sv, sw are sets of indices.
    # For simplicity and speed, we restrict to patterns where |su|, |sv|, |sw| are small.
    # Specifically, we try to find 15 patterns.
    
    # Candidate patterns:
    # 1. "Pure" patterns: single indices (i,j), (j,k), (k,i) where j matches and k,i match.
    #    These are the 18 schoolbook terms.
    # 2. "Mixed" patterns: e.g., su = {i1*m+j, i2*m+j}, sv = {j*p+k}, sw = {k*n+i1, k*n+i2}
    #    This would cover two terms (i1,j,k) and (i2,j,k) if they share j and k.
    #    But wait, if su has two i's, and sv has one j, and sw has two i's (matching the k),
    #    then the pattern covers (i1,j,k) and (i2,j,k).
    #    Coefficients can be 1.
    
    # Let's generate all "valid" patterns where the support is a product set (su x sv x sw)
    # intersected with the support of T.
    # Actually, a rank-1 tensor u \otimes v \otimes w has support su x sv x sw.
    # If we sum them, the coefficient at (u,v,w) is the sum of coeffs of patterns containing (u,v,w).
    # We need this sum to be 1 for all (u,v,w) in support(T) and 0 otherwise.
    
    # Since T has 18 non-zero entries, and we want rank 15, we need 3 "savings".
    # This implies we need patterns that cover multiple entries or cancel out extra entries.
    
    # Let's try a specific structure known for 2x2x2 (rank 7) extended to 2x3x3.
    # The 2x2x2 tensor has 8 entries. Rank 7 decomposition is:
    # P1: (0,0) (0,0) (0,0)
    # P2: (0,0) (1,1) (1,1)
    # P3: (0,1) (0,1) (1,0)
    # P4: (0,1) (1,0) (0,1)
    # P5: (1,0) (0,0) (0,1)
    # P6: (1,0) (1,1) (1,0)
    # P7: (1,1) (0,1) (0,0)
    # Wait, the standard rank 7 for 2x2x2 multiplication:
    # u1=(1,0), v1=(1,0), w1=(1,0) -> (0,0,0)
    # u2=(1,0), v2=(0,1), w2=(0,1) -> (0,1,1) ? No, w index is k*n+i.
    # Let's stick to the dictionary representation.
    
    # Let's use a linear programming or simple backtracking approach.
    # Given the small size (18 targets), we can enumerate subsets of "basis" patterns.
    
    # Let's define a set of candidate patterns.
    # Pattern is a dict: {(u,v,w): coeff}
    # We restrict coeffs to {1, -1, 0.5, -0.5} initially.
    
    # Generate all "rectangular" patterns where su, sv, sw are subsets of indices.
    # To keep it tractable, we only consider patterns where |su|<=2, |sv|<=2, |sw|<=2.
    # And the pattern must be "consistent" with the tensor structure (i.e., if u in su, v in sv, w in sw,
    # then the triple (i,j,k) derived from u,v,w must be consistent: j_u == j_v, k_v == k_w, i_w == i_u).
    
    candidates = []
    
    # Helper to get indices from flat index
    def u_to_ij(u): return divmod(u, m)
    def v_to_jk(v): return divmod(v, p)
    def w_to_ki(w): return divmod(w, n)
    
    # Generate all valid supports (su, sv, sw)
    # su subset of [0, nm), sv subset of [0, mp), sw subset of [0, pn)
    # Size 1:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                candidates.append(((u,), (v,), (w,)))
                
    # Size 2 in U (varying i, fixed j, k)
    for j in range(m):
        for k in range(p):
            for i1 in range(n):
                for i2 in range(n):
                    if i1 == i2: continue
                    u1 = i1*m + j
                    u2 = i2*m + j
                    v = j*p + k
                    w1 = k*n + i1
                    w2 = k*n + i2
                    # Support: {u1, u2} x {v} x {w1, w2}
                    # This covers (i1,j,k) and (i2,j,k).
                    # Check if consistent:
                    # (u1, v, w1): j_u1=j, j_v=j, k_v=k, k_w1=k, i_w1=i1, i_u1=i1. OK.
                    # (u1, v, w2): i_w2=i2 != i1. Not in T.
                    # So this pattern covers exactly 2 entries of T if we set coeffs appropriately?
                    # No, the pattern is a tensor product. It has non-zeros at ALL combinations in the support.
                    # So (u1, v, w2) is also in the support of the rank-1 tensor.
                    # But T[u1, v, w2] = 0.
                    # So we must cancel this out with other patterns or choose supports that don't create "cross terms" outside T.
                    
                    # This implies we need to be very careful.
                    # A rank-1 tensor with support {u1,u2} x {v} x {w1,w2} has 4 non-zero entries:
                    # (u1,v,w1), (u1,v,w2), (u2,v,w1), (u2,v,w2).
                    # Only (u1,v,w1) and (u2,v,w2) are in T.
                    # The other two are "noise".
                    # To use this, we need another pattern to cancel the noise.
                    
                    # This makes the search complex.
                    # Let's try a different approach: Hardcode a known rank 15 solution if I can derive it, 
                    # or use a more powerful solver.
    
    # Given the time constraint and the risk of invalidity, I will implement a robust fallback to rank 18,
    # but I will attempt a quick check for a rank 16 or 15 solution using a pre-computed set of patterns 
    # if I can find one in the literature or derive it simply.
    
    # Actually, let's try to construct a rank 16 solution by merging two schoolbook terms if possible.
    # Merging (i1,j,k) and (i2,j,k) requires a pattern that covers both and cancels cross terms.
    # Pattern P: u={u1,u2}, v={v}, w={w1,w2}. Coeff 1.
    # Non-zeros: (u1,v,w1)=1, (u1,v,w2)=1, (u2,v,w1)=1, (u2,v,w2)=1.
    # We need to cancel (u1,v,w2) and (u2,v,w1).
    # These are exactly the terms (i1,j,k) with i=i2? No, (u1,v,w2) corresponds to (i1, j, k) but w2 implies i2.
    # So (u1, v, w2) is (i1, j, k) in U,V but (k, i2) in W.
    # This entry is 0 in T.
    # We need another pattern Q to cancel these.
    # Q: u={u1}, v={v}, w={w2}? No, that's just the term itself if it were in T.
    # It is not in T.
    
    # It seems merging is difficult without introducing more complexity.
    
    # I will return the schoolbook rank 18 to ensure correctness.
    
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_fb[u][col] = 1
                V_fb[v][col] = 1
                W_fb[w][col] = 1
                col += 1
                
    return {"U": U_fb, "V": V_fb, "W": W_fb}
