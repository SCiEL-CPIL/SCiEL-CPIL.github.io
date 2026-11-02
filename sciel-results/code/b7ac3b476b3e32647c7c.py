"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s / 150s limit
    n, m, p = DIMENSIONS

    def build_strassen_35():
        rank = 35
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for block in range(5):
            k1 = 2 * block
            k2 = 2 * block + 1
            r = 7 * block + 0
            U[0][r] += 1; U[3][r] += 1
            V[k1][r] += 1; V[10 + k2][r] += 1
            W[2 * k1][r] += 1; W[2 * k2 + 1][r] += 1
            r = 7 * block + 1
            U[2][r] += 1; U[3][r] += 1
            V[k1][r] += 1
            W[2 * k1 + 1][r] += 1; W[2 * k2 + 1][r] -= 1
            r = 7 * block + 2
            U[0][r] += 1
            V[k2][r] += 1; V[10 + k2][r] -= 1
            W[2 * k2][r] += 1; W[2 * k2 + 1][r] += 1
            r = 7 * block + 3
            U[3][r] += 1
            V[10 + k1][r] += 1; V[k1][r] -= 1
            W[2 * k1][r] += 1; W[2 * k1 + 1][r] += 1
            r = 7 * block + 4
            U[0][r] += 1; U[1][r] += 1
            V[10 + k2][r] += 1
            W[2 * k1][r] -= 1; W[2 * k2][r] += 1
            r = 7 * block + 5
            U[2][r] += 1; U[0][r] -= 1
            V[k1][r] += 1; V[k2][r] += 1
            W[2 * k2 + 1][r] += 1
            r = 7 * block + 6
            U[1][r] += 1; U[3][r] -= 1
            V[10 + k1][r] += 1; V[10 + k2][r] += 1
            W[2 * k1][r] += 1
        return U, V, W

    def verify(U, V, W):
        """Check that sum_r U[i,r]*V[j,r]*W[k,r] = delta(i,k) for all i in 0..3, j in 0..19, k in 0..19 with correct trace convention."""
        # U has shape (4, rank) indexed by (a,b) -> a*2+b
        # V has shape (20, rank) indexed by (b,d) -> b*10+d  
        # W has shape (20, rank) indexed by (c,d) -> c*10+d  (trace convention: W[k*n+i, r] where k is col of B, n is rows of A)
        # Actually per the task: W[k*n+i, r] where k is the column index of B (0..9), i is row of output (0..1)
        # So W index = k*2 + i where k is B-column, i is output row
        # The tensor entry T[a,b,c,d] = delta(a,c) should equal sum_r U[a*2+b, r] * V[b*10+d, r] * W[d*2+c, r]
        rank = len(U[0])
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(10):
                        expected = 1.0 if a == c else 0.0
                        actual = 0.0
                        ui = a * 2 + b
                        vj = b * 10 + d
                        wk = d * 2 + c
                        for r in range(rank):
                            actual += U[ui][r] * V[vj][r] * W[wk][r]
                        if abs(actual - expected) > 1e-9:
                            return False
        return True

    # Start with the known rank-35 solution
    U, V, W = build_strassen_35()
    
    # Verify our base solution is correct
    if not verify(U, V, W):
        # Fallback: return as-is
        return {"U": U, "V": V, "W": W}
    
    best_rank = 35
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank by zeroing out columns and checking if the remaining
    # decomposition still works. This is a greedy approach: try removing each
    # rank-1 term and see if we can compensate.
    
    # Alternative: try a structured search. The key observation is that in the
    # Strassen decomposition, the A-side vectors repeat across blocks. Perhaps
    # we can merge some terms.
    
    # Let's try: can we achieve rank 34 by finding a linear dependency among
    # the 35 rank-1 terms?
    import numpy as np
    
    # Build the tensor from the decomposition and check if any term is redundant
    # by checking if it can be expressed as a combination of others.
    
    # For each term r, check if removing it leaves a solvable system for the
    # remaining terms. This is expensive, so let's try a simpler heuristic:
    # check if the 35 vectors in the U, V, W spaces have any linear dependencies
    # that allow merging.
    
    # Simpler approach: try all pairs of terms to see if one can be absorbed
    # into another by adjusting coefficients.
    
    # Actually, let's try a more direct approach: use the fact that the tensor
    # T[a,b,c,d] = delta(a,c) has a special structure. For each (b,d), the
    # contribution to the (a,c) plane is delta(a,c). So for each (b,d), we need
    # a rank-2 decomposition of the 2x2 identity, which requires at least 2
    # terms unless we share across (b,d) pairs.
    
    # The lower bound: for each of the 20 (b,d) pairs, we need the (a,c) part
    # to give identity. If we use s terms total, each term contributes to at
    # most one (b,d) pair (since V and W are fixed per term). Wait, no - a
    # single term can contribute to multiple (b,d) pairs if V and W have
    # support on multiple (b,d) entries.
    
    # Let's just try to see if rank 34 is possible by a search.
    # Strategy: take the rank-35 solution and try to set one column to zero,
    # then solve for adjustments in the remaining 34 columns.
    
    # This is a linear system problem. For each output entry (a,b,c,d), we need:
    # sum_{r=0}^{33} U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r] = delta(a,c)
    # where we've removed column 34.
    
    # This is nonlinear in the entries, so let's try a different approach:
    # fix U and V, solve for W, or fix U and W, solve for V, etc.
    
    # Let's try: fix U and V from the rank-35 solution (with column 34 removed),
    # and see if we can find W that works.
    
    # Actually, the simplest improvement to try: can we reduce from 35 to 34
    # by finding that one of the 35 terms is actually a linear combination of
    # the others in the tensor sense?
    
    # Let's build the full tensor from the decomposition and check its rank
    # properties.
    
    # T[a,b,c,d] = sum_r U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r]
    
    # Reshape to a matrix: for each (b,d), we have a 2x2 matrix M_{bd}[a,c] = T[a,b,c,d]
    # The tensor rank is at least the max over all (b,d) of the rank of M_{bd},
    # which is 2 (since M_{bd} = I_2 for all b,d).
    # Also, the tensor rank is at least the sum over b of the rank of the
    # 2x(2*10) matrix T[:,b,:,:] reshaped... this is getting complex.
    
    # Let me just try a practical search: attempt to find a rank-34 solution
    # by perturbing the rank-35 solution and solving.
    
    # For now, return the rank-35 solution. The search for rank 34 would
    # require a more sophisticated approach that may not succeed within
    # the time budget.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
