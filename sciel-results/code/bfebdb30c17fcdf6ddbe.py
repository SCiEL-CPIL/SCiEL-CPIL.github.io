"""Construct a kissing configuration in dimension 11 using the D11 lattice shortest vectors."""
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate all shortest vectors of the D11 lattice.
    # D_n = { x in Z^n : sum(x_i) is even }
    # The shortest vectors have squared norm 4, i.e., norm 2.
    # They are of two types:
    # 1. Two coordinates are +1 or -1, rest are 0 (with even sum, which is always 2 or -2, so even).
    #    Actually, sum is +/-2 which is even. So all such vectors are in D_n.
    #    Count: C(n,2) * 4 = 55 * 4 = 220 for n=11.
    # 2. All coordinates are +/-1. Sum is even iff the number of -1s is even.
    #    Count: 2^(n-1) = 2^10 = 1024.
    # Total: 220 + 1024 = 1244? Wait, let me recheck.
    
    # Actually, for D_n, the minimum norm squared is 2 (vectors with two 1s and rest 0).
    # But the task says "minimum pairwise distance is at least the maximum vector norm".
    # If we normalize to norm 2, then squared norm is 4.
    # Let's re-read: "Normalizing each vector to norm 2 certifies a kissing configuration".
    # So we need integer vectors. The condition is: min pairwise distance >= max norm.
    # If all vectors have the same norm R, then min pairwise distance >= R.
    # For D_n lattice, the shortest vectors have norm sqrt(2). 
    # If we scale by sqrt(2), they have norm 2. But they must be integer vectors.
    # The vectors with two +/-1s have norm sqrt(2). They are integer vectors.
    # Pairwise distance between two such vectors: 
    #   If they share no indices: distance squared = 4 (each has norm sqrt(2), angle 90 deg? No.)
    #   Let's compute: v = e_i + e_j, w = e_k + e_l with {i,j} != {k,l} and disjoint.
    #   ||v-w||^2 = ||v||^2 + ||w||^2 - 2 v.w = 2 + 2 - 0 = 4. So distance = 2.
    #   Max norm = sqrt(2). So 2 >= sqrt(2). OK.
    #   If they share one index: v = e_i + e_j, w = e_i + e_k (j!=k).
    #   ||v-w||^2 = ||e_j - e_k||^2 = 2. Distance = sqrt(2).
    #   Max norm = sqrt(2). So sqrt(2) >= sqrt(2). OK.
    #   If they are the same: distance 0. But we need distinct vectors.
    #   If v = e_i + e_j and w = -(e_i + e_j) = -e_i - e_j.
    #   ||v-w||^2 = ||2e_i + 2e_j||^2 = 8. Distance = 2*sqrt(2) >= sqrt(2). OK.
    #   What about v = e_i + e_j and w = e_j + e_i? Same vector.
    #   What about v = e_i - e_j and w = e_i + e_j?
    #   ||v-w||^2 = ||-2e_j||^2 = 4. Distance = 2 >= sqrt(2). OK.
    
    # So all vectors with exactly two non-zero entries (each +/-1) work?
    # Let's check v = e_i + e_j and w = e_i - e_j.
    #   v.w = 1 - 1 = 0. ||v-w||^2 = 2+2-0 = 4. OK.
    #   v = e_i + e_j, w = -e_i + e_j. v.w = -1+1=0. ||v-w||^2=4. OK.
    #   v = e_i + e_j, w = -e_i - e_j. v.w = -1-1=-2. ||v-w||^2 = 2+2-2(-2)=8. OK.
    
    # Now the all-ones type vectors: x in {+1,-1}^n with even number of -1s.
    # Norm squared = n = 11. Norm = sqrt(11).
    # These have larger norm than the two-nonzero vectors.
    # If we mix them, the max norm is sqrt(11).
    # Distance between two all-ones vectors x, y:
    #   ||x-y||^2 = sum (x_i - y_i)^2. Each term is 0 or 4.
    #   Let k be the number of positions where x_i != y_i. Then ||x-y||^2 = 4k.
    #   Since x and y both have even parity (even number of -1s), the number of positions where they differ must be even.
    #   So k is even. Minimum k=2 (if they differ in exactly 2 positions).
    #   Then ||x-y||^2 = 8, distance = 2*sqrt(2) ≈ 2.828.
    #   Max norm = sqrt(11) ≈ 3.317.
    #   2.828 < 3.317. So this FAILS.
    
    # So we cannot mix the two types. We must use only one type.
    
    # Option 1: Only two-nonzero vectors. Norm = sqrt(2).
    #   Count: C(11,2) * 4 = 55 * 4 = 220.
    #   We verified min distance >= norm.
    
    # Option 2: Only all-ones vectors with even parity. Norm = sqrt(11).
    #   Count: 2^10 = 1024.
    #   But min distance between two such vectors is 2*sqrt(2) ≈ 2.828 < sqrt(11) ≈ 3.317.
    #   So this does NOT satisfy the condition.
    
    # Wait, did I make a mistake? Let's re-examine.
    # The condition is: minimum pairwise distance >= maximum vector norm.
    # For the all-ones vectors, max norm = sqrt(11).
    # Min pairwise distance: we need min ||x-y|| over all distinct x,y in the set.
    # We found min ||x-y||^2 = 8, so min distance = 2*sqrt(2) ≈ 2.828.
    # 2.828 < 3.317. So indeed, it fails.
    
    # Therefore, only the two-nonzero vectors work, giving 220 vectors.
    
    # But the prompt mentions "kissing number of D11 is exactly 2304". 
    # Let me re-read the prompt's strategy hint:
    # "generate all 2304 shortest vectors (squared norm 4) from the D11 lattice definition"
    # Squared norm 4 means norm 2. 
    # In D_n, the shortest vectors have squared norm 2. 
    # If we consider the lattice 2*D_n or some scaling?
    # Or maybe the prompt is referring to a different lattice?
    # The D11 lattice has 2304 minimal vectors? Let me check.
    # Number of minimal vectors in D_n: 2*n*(n-1) for the type with two non-zeros? No.
    # For D_n, minimal vectors are:
    #   Type 1: two coordinates are +/-1, rest 0. Count: 2 * C(n,2) * 2 = 2n(n-1)? 
    #   Wait, for each pair (i,j), we have 4 vectors: (1,1), (1,-1), (-1,1), (-1,-1).
    #   So count = 4 * C(n,2) = 2n(n-1). For n=11: 2*11*10 = 220.
    #   Type 2: all coordinates +/-1 with even sum. Count: 2^(n-1). For n=11: 1024.
    #   Total: 220 + 1024 = 1244. Not 2304.
    
    # Maybe the prompt is referring to the Leech lattice or something else? 
    # Or maybe D11 is not the right lattice?
    # The kissing number in dimension 11 is known to be between 2304 and 2336? 
    # Actually, the kissing number in dimension 11 is not known exactly, but it is at least 2304.
    # 2304 = 2 * 11 * 10 + 2^10? No, 220 + 1024 = 1244.
    # 2304 = 2^8 * 9 = 256 * 9.
    # Maybe it's the number of vectors in some other construction.
    
    # Let's just use the 220 vectors from the two-nonzero type, since they satisfy the condition.
    # We can also try to find more vectors that work.
    
    # Actually, wait. Can we scale the vectors? The task says "integer vectors".
    # If we take the two-nonzero vectors and multiply by 1, they have norm sqrt(2).
    # If we take them and multiply by k, they have norm k*sqrt(2).
    # But they must be integer vectors. So k must be such that k*1 is integer, which is always true.
    # But the condition is min distance >= max norm.
    # If we scale by k, both distance and norm scale by k, so the condition is invariant.
    # So we can just use k=1.
    
    # Let's generate the 220 vectors.
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)
    
    # Verify the condition
    arr = np.array(vectors, dtype=np.int64)
    norms = np.sqrt(np.sum(arr**2, axis=1))
    max_norm = np.max(norms)
    
    # Compute pairwise distances
    # This is O(k^2 * D) which is 220^2 * 11 = 532,400 operations. Fast.
    diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Check condition
    if min_dist < max_norm - 1e-9:
        # This should not happen, but just in case
        pass
    
    return arr

# EVOLVE-BLOCK-END
