import time
import random
import copy
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition 
    for multiplying a 3x3 matrix by a 3x5 matrix.
    
    Strategy:
    1. Split the 3x3x5 problem into 3x3x3 and 3x3x2.
    2. Search for a low-rank decomposition of the 3x3x3 part (target rank 23).
    3. Use schoolbook decomposition for the 3x3x2 part (rank 18).
    4. Combine the decompositions.
    5. If no low-rank 3x3x3 is found, fall back to schoolbook 3x3x5 (rank 45).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 5s safety margin

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    def check_decomposition(U, V, W, n, m, p, tol=1e-10):
        """
        Check if U, V, W form an exact tensor decomposition for n x m x p.
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def combine_decompositions(decomp1, n1, m1, p1, decomp2, n2, m2, p2):
        """
        Combine two decompositions.
        decomp1 is for n1 x m1 x p1.
        decomp2 is for n2 x m2 x p2.
        Assumes n1==n2, m1==m2, and p = p1 + p2.
        """
        U1, V1, W1 = decomp1['U'], decomp1['V'], decomp1['W']
        U2, V2, W2 = decomp2['U'], decomp2['V'], decomp2['W']
        
        # U and V are concatenated horizontally (rank adds)
        # W is concatenated horizontally? No, W depends on k.
        # W1 has rows indexed by k in [0, p1).
        # W2 has rows indexed by k in [0, p2).
        # The combined W should have rows indexed by k in [0, p1+p2).
        # For k < p1, use W1[k, :]. For k >= p1, use W2[k-p1, :].
        
        U_combined = np.hstack([U1, U2])
        V_combined = np.hstack([V1, V2])
        
        # W construction
        # W1 shape: (p1 * n, rank1)
        # W2 shape: (p2 * n, rank2)
        # We need to interleave or append?
        # The tensor is T[i,j,k].
        # For k in 0..p1-1, the term is from decomp1.
        # For k in p1..p1+p2-1, the term is from decomp2.
        # So W_combined rows:
        # Rows 0..p1*n - 1 correspond to k in 0..p1-1.
        # Rows p1*n .. (p1+p2)*n - 1 correspond to k in p1..pp1+p2-1.
        
        W_combined = np.zeros((p1 * n + p2 * n, U1.shape[1] + U2.shape[1]), dtype=float)
        
        # Fill W1 part
        # W1 is indexed by k1, i. Index = k1 * n + i.
        # In W_combined, this is the same index.
        W_combined[:p1 * n, :U1.shape[1]] = W1
        
        # Fill W2 part
        # W2 is indexed by k2, i. Index = k2 * n + i.
        # In W_combined, this is index (p1 * n) + k2 * n + i.
        W_combined[p1 * n:, U1.shape[1]:] = W2
        
        return {"U": U_combined, "V": V_combined, "W": W_combined}

    # Search for 3x3x3 decomposition
    n3, m3, p3 = 3, 3, 3
    n2, m2, p2 = 3, 3, 2
    
    best_decomp_333 = None
    best_rank_333 = 27 # If we don't find < 27, combined rank >= 45
    
    # Try ranks from 23 to 26
    for rank in range(23, 27):
        if time.time() > deadline:
            break
        
        # Random search for 3x3x3
        # Use small integer/half-integer entries
        # Strassen-like decompositions often use -1, 0, 1
        # Let's try a structured random search
        num_attempts = 1000
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Generate random matrices with entries in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
            # Or just integers {-1, 0, 1} first
            U = np.random.choice([-1.0, 0.0, 1.0], (n3 * m3, rank))
            V = np.random.choice([-1.0, 0.0, 1.0], (m3 * p3, rank))
            W = np.random.choice([-1.0, 0.0, 1.0], (p3 * n3, rank))
            
            if check_decomposition(U, V, W, n3, m3, p3):
                best_decomp_333 = {"U": U, "V": V, "W": W}
                best_rank_333 = rank
                break # Found lowest rank, stop
        
        if best_decomp_333 is not None:
            break

    if best_decomp_333 is not None:
        # Get schoolbook for 3x3x2
        decomp_332 = schoolbook_decomposition(n2, m2, p2)
        # Combine
        combined = combine_decompositions(best_decomp_333, n3, m3, p3, decomp_332, n2, m2, p2)
        # Verify combined
        if check_decomposition(combined['U'], combined['V'], combined['W'], n, m, p):
            return {
                "U": combined['U'].tolist(),
                "V": combined['V'].tolist(),
                "W": combined['W'].tolist()
            }

    # Fall back to schoolbook 3x3x5
    decomp_335 = schoolbook_decomposition(n, m, p)
    return {
        "U": decomp_335['U'].tolist(),
        "V": decomp_335['V'].tolist(),
        "W": decomp_335['W'].tolist()
    }
# EVOLVE-BLOCK-END
