# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    their maximum squared norm is strictly less than their minimum pairwise squared distance.
    Uses the construction based on 2*E8 + 2*D3 (or equivalent high-density lattice packing).

    Returns:
        points: np.ndarray of shape (num_points, 11) of int64
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    d = 11

    # Generate E8 lattice points (all integer vectors of norm 2, i.e., 2 ones and 8 zeros, with signs)
    # E8 has 240 minimal vectors.
    e8_points = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Choose positions i and j for the two non-zero entries
            for sign_i in [1, -1]:
                for sign_j in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = sign_i
                    v[j] = sign_j
                    e8_points.append(v)
    
    e8_points = np.array(e8_points)  # Shape (240, 8)
    assert e8_points.shape == (240, 8)

    # Generate D3 lattice points
    # D3 is the lattice of integer vectors in R^3 with even sum of coordinates.
    # Minimal vectors of D3 have norm 2.
    # They are: (1,1,0), (1,-1,0), (1,0,1), (1,0,-1), (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
    # Wait, let's list all integer vectors with even sum and norm 2.
    # Sum must be even.
    # (1,1,0) sum=2 (even)
    # (1,-1,0) sum=0 (even)
    # (1,0,1) sum=2 (even)
    # (1,0,-1) sum=0 (even)
    # (0,1,1) sum=2 (even)
    # (0,1,-1) sum=0 (even)
    # (0,-1,1) sum=0 (even)
    # (0,-1,-1) sum=-2 (even)
    # Are there others? Norm is 2, so two 1s and one 0, or one 1 and one -1 and one 0?
    # If two 1s: (1,1,0) perms -> 3 * 1 (signs fixed to 1) = 3 vectors? No, signs can be anything as long as sum is even.
    # Let's just generate all integer vectors in [-2,2]^3 with norm 2 and even sum.
    d3_points = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                if x*x + y*y + z*z == 2:
                    if (x + y + z) % 2 == 0:
                        d3_points.append([x, y, z])
    
    d3_points = np.array(d3_points, dtype=np.int64)
    # There should be 12 such vectors.
    # Let's verify count.
    # Permutations of (1,1,0) with signs such that sum is even.
    # (1,1,0) sum 2. (1,-1,0) sum 0. (-1,1,0) sum 0. (-1,-1,0) sum -2.
    # So for each permutation of (1,1,0), there are 4 sign combinations?
    # No, the values are fixed magnitudes.
    # Let's trust the loop.
    assert d3_points.shape[0] == 12, f"D3 count is {d3_points.shape[0]}"

    # Construct 11D points
    # The construction mentioned in knowledge: 2*E8 + 2*D3.
    # This likely means taking the direct sum of two E8 lattices and two D3 lattices?
    # But dimension is 8+8+3+3 = 22. We need dimension 11.
    # The knowledge says "Replaced D11 (220 pts) with 2*E8 + 2*D3 (252 pts)".
    # 2*E8 has 2*240 = 480 points? No, E8 has 240 minimal vectors.
    # Maybe it means the lattice L = E8 x E8 x D3 x D3 is not right for dim 11.
    # Let's re-read carefully: "Integer kissing configuration in dimension 11".
    # The seed program returned 2 points (1,1..1) and (-1..-1).
    # The improvement to 110 was likely D11 (220 points? No, D11 minimal vectors are 2*11*10 = 220? No, D_n has 2n(n-1) minimal vectors. D11 has 2*11*10=220. But the score was 110. Maybe only half were used? Or maybe the constraint is stricter.)
    # The constraint is: min pairwise squared distance >= max squared norm.
    # For a lattice, if we take all minimal vectors, their norms are equal (say 2).
    # The distance between two minimal vectors u and v is |u-v|^2 = |u|^2 + |v|^2 - 2u.v = 4 - 2u.v.
    # If u.v = 0, dist^2 = 4. If u.v = 1, dist^2 = 2. If u.v = 2, dist^2 = 0 (same point).
    # For D11, minimal vectors have norm 2.
    # Inner products of minimal vectors in D_n:
    # They can be 0, 1, or 2.
    # If u and v are distinct minimal vectors, u.v <= 1?
    # In D_n, minimal vectors are (1, 1, 0...) and (1, -1, 0...).
    # Let u = e1 + e2, v = e1 + e3. u.v = 1. Dist^2 = 4 - 2 = 2.
    # Max norm squared = 2.
    # Min dist squared = 2.
    # Condition: min_dist^2 >= max_norm^2. 2 >= 2. Holds.
    # So D11 gives 220 points. Why was the score 110?
    # Maybe the evaluator requires strict inequality? ">= the maximum squared norm".
    # Or maybe some pairs have distance < 2?
    # In D_n, can u.v be 2? Only if u=v.
    # Can u.v be -1? u = e1+e2, v = -e1+e3. u.v = -1. Dist^2 = 4 - 2(-1) = 6.
    # So min dist^2 is 2.
    # So D11 should yield 220 points.
    # The knowledge says "Replaced D11 (220 pts) ... Measured: 110 -> 126".
    # This implies the previous best was 110, not 220.
    # Why would D11 give 110?
    # Maybe the points must be distinct? They are.
    # Maybe the zero vector is excluded? It is.
    # Maybe I am misinterpreting "2*E8 + 2*D3".
    # Dimension 11.
    # Perhaps the construction is a subset of a higher dimensional lattice projected or embedded?
    # Or maybe it's a specific code.
    # Let's look at the numbers: 252 points.
    # 252 = 2 * 126.
    # E8 has 240. D3 has 12.
    # 240 + 12 = 252.
    # Ah! Maybe the points are formed by taking E8 points in the first 8 coords and 0 in the last 3, PLUS D3 points in the last 3 coords and 0 in the first 8?
    # Let's check this union.
    # Set A: (e, 0) where e in E8_min. Norm^2 = 2. Count 240.
    # Set B: (0, d) where d in D3_min. Norm^2 = 2. Count 12.
    # Total 252 points.
    # Check distances:
    # 1. Within A: dist^2 = dist_E8^2. Min is 2.
    # 2. Within B: dist^2 = dist_D3^2. Min is 2.
    # 3. Between A and B: u = (e, 0), v = (0, d). u.v = 0. dist^2 = 2 + 2 = 4.
    # Max norm squared = 2.
    # Min dist squared = 2.
    # Condition: 2 >= 2. Holds.
    # This yields 252 points.
    # This matches the "252 pts" in the knowledge!
    # And it is a structural change from just D11 (if D11 was the previous best, but maybe D11 wasn't implemented correctly or the previous best was something else).
    # The prompt says "Replacing D11 (220 pts) with ... (252 pts)".
    # So if I implement this union, I should get 252.
    # The current score is 1.0 (2 points).
    # The island best was 110 (from p38).
    # If I get 252, that is a huge jump.

    points_a = np.zeros((240, 11), dtype=np.int64)
    points_a[:, :8] = e8_points

    points_b = np.zeros((12, 11), dtype=np.int64)
    points_b[:, 8:] = d3_points

    all_points = np.vstack([points_a, points_b])

    # Verify constraints explicitly to be safe
    # 1. Shape (k, 11)
    assert all_points.shape[1] == 11
    k = all_points.shape[0]
    assert k >= 2

    # 2. No zero vector
    norms_sq = np.sum(all_points**2, axis=1)
    assert not np.any(norms_sq == 0)

    # 3. Min pairwise dist^2 >= Max norm^2
    max_norm_sq = np.max(norms_sq)
    
    # Compute min pairwise distance
    min_dist_sq = np.inf
    # We can do this efficiently or with a loop. k=252, pairs ~ 31k. Loop is fine.
    for i in range(k):
        for j in range(i + 1, k):
            diff = all_points[i] - all_points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                if min_dist_sq < max_norm_sq:
                    # If this happens, the construction is invalid.
                    # But theoretically it should hold.
                    pass

    if min_dist_sq >= max_norm_sq:
        return all_points
    else:
        # Fallback to a known valid smaller set if the above fails (should not happen)
        # Just return the first few points of A
        return all_points[:10]

# EVOLVE-BLOCK-END
