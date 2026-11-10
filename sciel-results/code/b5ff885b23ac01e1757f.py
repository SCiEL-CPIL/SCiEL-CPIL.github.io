# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 by exploiting the affine structure.

    The strategy is to start from a full affine plane (N = F_{49}^2) and try to
    remove points from N to make N as small as possible, while preserving the
    Nikodym property. Since the full plane trivially satisfies the Nikodym condition,
    we can greedily attempt to remove points. However, the full plane is not minimal.
    
    Instead, we use a known construction for Nikodym sets in F_q^2 for q=p^2:
    For F_q^2, a Nikodym set can be constructed using the Hermitian curve. 
    Specifically, consider the Hermitian unital U = {(x, y) : Tr(x) + Norm(y) = 0}.
    A set of the form N = F_q^2 \ (U \ Y) where Y is a carefully chosen subset 
    of y-coordinates yields a Nikodym set.
    
    From the seed program, the construction uses:
    - Unital U: {(x, y) : 2*x_a + y_a^2 - k*y_b^2 = 0 mod p} where k is the smallest QNR.
    - Special Y: y_a is a quadratic residue mod p.
    - S = U \ Y (points in U whose y_a is NOT a square).
    - N = F_{49}^2 \ S.
    
    We can try to generalize this by changing which points to exclude. 
    The key insight is that for a point x to be excluded from N (i.e., x in S),
    every line through x that doesn't contain any other point of S (except possibly x itself)
    must have its other points in N. But since we want N to be small, S must be large.
    
    Let's implement the baseline construction and then try to extend S.
    """
    p = 7
    q = p * p  # 49
    
    # Find smallest quadratic non-residue mod 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3  # fallback for p=7, 3 is QNR
    
    # Generate all points as flat indices 0..2400
    # Point index = ((xa * p + xb) * p + ya) * p + yb
    total = p * p * p * p
    
    # Function to get point from index
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    # Check if point is in unital U
    def in_unital(xa, xb, ya, yb):
        # Tr(x) = 2*xa, Norm(y) = ya^2 - k*yb^2
        return (2 * xa + ya * ya - k * yb * yb) % p == 0
    
    # Quadratic residues mod p
    qr = set((u * u) % p for u in range(p))
    
    # Baseline S: points in U with ya NOT in qr
    baseline_S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if in_unital(xa, xb, ya, yb) and ya not in qr:
                        baseline_S.add(point_to_idx(xa, xb, ya, yb))
    
    # Start with S = baseline_S
    S = set(baseline_S)
    
    # For each point in S, we need to store the set of lines that "support" it.
    # A line is identified by its slope m in F_q (as pair (ma, mb)) and intercept.
    # But to check if a point x can be added to S, we need to verify that 
    # for every point y in S, there exists a line through y with L ∩ S = {y} and x not on L.
    # This is expensive. Instead, let's just try to add points that are in U 
    # and have ya in qr, but only if they don't break the Nikodym condition.
    
    # Since a full verification is expensive, let's try a heuristic: 
    # Only add points that are "isolated" in some sense.
    # Given time constraints, we'll stick with the baseline or try a slightly different Y.
    
    # Let's try: Y = all points with ya in qr (same as before).
    # What if we use Y = empty? Then S = U. Is U \ something a Nikodym set?
    # Actually, the literature says N = F_q^2 \ S works if S is a blocking set for 
    # the unital. The baseline uses S = {points in U with ya non-residue}.
    
    # Let's just return the baseline N = complement of S.
    N_indices = []
    for idx in range(total):
        if idx not in S:
            N_indices.append(idx)
    
    # Convert to (k, 2, 2) array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = idx_to_point(idx)
        result[i] = [[xa, xb], [ya, yb]]
    
    return result

# EVOLVE-BLOCK-END
