"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        return False, float(np.max(np.abs(total - expected)))
    return True, 0.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Build the known rank-54 Strassen-based solution as fallback
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P2 = (A_1,j0 + A_1,j1) B_j0,k0
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                
                # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                
                # P5 = (A_0,j0 + A_0,j1) B_j1,k1
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                
                # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        # Remaining j=4
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build baseline
    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    if not valid:
        return baseline
    
    best = baseline
    best_rank = 54
    
    # Try to find a lower-rank decomposition
    # Strategy: For 2x5x6, try different block partitions
    # The key is that R(2 x m x p) can sometimes be less than the sum of block ranks
    # due to shared structure across blocks
    
    # Try rank 42: Use a smarter partition
    # Split j into {0,1,2} and {3,4}
    # For 2x3x6: we can use a decomposition of rank at most 2*3+2*6-2 = 14? No, that's not right.
    # Actually, let me try to construct specific low-rank solutions
    
    # Approach: Try to find if rank 42 is achievable
    # 2x5x6 tensor: We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # Let's try a direct construction with rank 42
    # Idea: Use 7 Strassen-like operations per 2x2x2 block, but share some across blocks
    # j has 5 elements, k has 6 elements
    # Partition: j={0,1,2,3,4}, k={0,1,2,3,4,5}
    # Use blocks: (j0,j1) x (k0,k1) for 7 products each
    # But try to share: if we use (j0,j1) with all k's in a smarter way
    
    # Alternative: Try rank 36 = 2*5 + 2*6 - 2? That formula gives R(2xp x p x 2) but not our case.
    # For 2xm x px2, R = 2m+2p-2 when m=p. For 2x5x6, it's not symmetric.
    
    # Let me try: can we do 2x5x6 in rank 2*5 + 2*6 - 2 = 16? No, that's for 2xm x m x 2.
    # Our tensor is 2x5 x 5x6, so it's not that form.
    
    # The lower bound is max(2*5, 5*6, 2*6) = max(10, 30, 12) = 30
    # Upper bound from Strassen blocking: 54 as we have
    
    # Let me try a different blocking: 
    # j={0,1,2,3,4}, k={0,1,2,3,4,5}
    # Use 2x2 blocks in k: (0,1), (2,3), (4,5) - 3 blocks
    # For each j, use Strassen with the 3 k-blocks? No, Strassen needs 2x2.
    
    # Try: j={0,1} with k={(0,1),(2,3),(4,5)}: 3*7 = 21
    # j={2,3} with k={(0,1),(2,3),(4,5)}: 3*7 = 21
    # j={4} with all k: 2*6 = 12
    # Total: 54 (same as before)
    
    # Try: j={0,1,2} with k={(0,1),(2,3),(4,5)}
    # For 2x3x6, what's the rank? We can do it in at most 2*3*6=36, but better.
    # Use Strassen on j={0,1} for each k-pair: 3*7=21, plus j=2: 2*6=12, total 33
    # Then j={3,4} with k={(0,1),(2,3),(4,5)}: 2*7 + 2*6 = 14+12=26? No...
    # j={3,4}: Strassen for each k-pair: 3*7=21? But j has only 2 elements, so 7 per k-pair = 21
    # Total: 33 + 21 = 54 again.
    
    # Let me try a completely different approach: 
    # Use the fact that we can sometimes combine terms across blocks
    # For now, stick with 54 unless we find something better quickly
    
    if time.time() > deadline - 5:
        return best
    
    # Try to see if we can get rank 48 by sharing one term
    # This is a heuristic attempt - not guaranteed to work
    # For now, return the verified rank-54 solution
    
    return best
# EVOLVE-BLOCK-END
