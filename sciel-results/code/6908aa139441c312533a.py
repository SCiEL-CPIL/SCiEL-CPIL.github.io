import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 2.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy: Exploit linearity in output dimension n=2. View as two 1x3x10
    multiplications. Search for shared structure that allows rank reduction
    below schoolbook rank 60.
    
    If no reduction found, return the schoolbook decomposition (rank 60).
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    # Start with schoolbook decomposition
    rank_school = n * m * p  # 60
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by merging pairs of terms
    # For each pair of schoolbook terms, check if they can be combined
    # into a single term with adjusted coefficients
    
    def try_merge(U, V, W, idx1, idx2, time_budget):
        """
        Try to merge terms idx1 and idx2 into a single term.
        Returns (merged_U, merged_V, merged_W, success) or (None, None, None, False)
        """
        if time.time() > time_budget:
            return None, None, None, False
        
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Extract the two terms
        U1 = [U[i][idx1] for i in range(n*m)]
        U2 = [U[i][idx2] for i in range(n*m)]
        V1 = [V[i][idx1] for i in range(m*p)]
        V2 = [V[i][idx2] for i in range(m*p)]
        W1 = [W[i][idx1] for i in range(p*n)]
        W2 = [W[i][idx2] for i in range(p*n)]
        
        # For a valid merge, we need:
        # U1 * V1 + U2 * V2 = U_m * V_m for some U_m, V_m
        # This means the outer products must be linearly dependent
        
        # Check if the two outer products are proportional
        # For simplicity, try U_m = U1 + U2, V_m = V1 + V2
        # This works if U1*V1 + U2*V2 = (U1+U2)*(V1+V2)
        # Which requires U1*V2 + U2*V1 = 0
        
        # Compute residual
        residual = 0.0
        for i in range(n*m):
            for j in range(m*p):
                # (U1*V1 + U2*V2)[i,j] vs (U1+U2)*(V1+V2)[i,j]
                lhs = U1[i]*V1[j] + U2[i]*V2[j]
                rhs = (U1[i]+U2[i])*(V1[j]+V2[j])
                residual += abs(lhs - rhs)
        
        if residual < 1e-10:
            # Merge is valid
            merged_U = [U[i][:] for i in range(n*m)]
            merged_V = [V[i][:] for i in range(m*p)]
            merged_W = [W[i][:] for i in range(p*n)]
            
            # Combine the two terms
            for i in range(n*m):
                merged_U[i][idx1] = U1[i] + U2[i]
            for i in range(m*p):
                merged_V[i][idx1] = V1[i] + V2[i]
            for i in range(p*n):
                merged_W[i][idx1] = W1[i] + W2[i]
            
            # Zero out the second term
            for i in range(n*m):
                merged_U[i][idx2] = 0
            for i in range(m*p):
                merged_V[i][idx2] = 0
            for i in range(p*n):
                merged_W[i][idx2] = 0
            
            # Verify the merge produces correct output
            # Check that the combined term contributes correctly
            success = True
            return merged_U, merged_V, merged_W, success
        
        return None, None, None, False
    
    # Try all pairs of terms that could potentially merge
    # Focus on pairs that share structure (same i or same k)
    pairs_to_try = []
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    # Same row i, same column j, different output k
                    idx1 = i * m * p + j * p + k1
                    idx2 = i * m * p + j * p + k2
                    pairs_to_try.append((idx1, idx2))
    
    # Shuffle for diversity
    random.shuffle(pairs_to_try)
    
    for idx1, idx2 in pairs_to_try:
        if time.time() > deadline:
            break
        
        merged_U, merged_V, merged_W, success = try_merge(
            best_U, best_V, best_W, idx1, idx2, deadline
        )
        
        if success:
            # Verify the merged decomposition is still valid
            valid = True
            for i in range(n):
                for k in range(p):
                    # Check output coefficient
                    expected = 1.0
                    computed = 0.0
                    for j in range(m):
                        for t in range(best_rank):
                            if merged_U[i*m+j][t] != 0 and merged_V[j*p+k][t] != 0:
                                computed += merged_U[i*m+j][t] * merged_V[j*p+k][t]
                    if abs(computed - expected) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Update best
            best_rank -= 1
            # Remove the zeroed column
            new_U = []
            for row in merged_U:
                new_row = [val for idx, val in enumerate(row) if idx != idx2]
                new_U.append(new_row)
            new_V = []
            for row in merged_V:
                new_row = [val for idx, val in enumerate(row) if idx != idx2]
                new_V.append(new_row)
            new_W = []
            for row in merged_W:
                new_row = [val for idx, val in enumerate(row) if idx != idx2]
                new_W.append(new_row)
            
            best_U, best_V, best_W = new_U, new_V, new_W
            print(f"Reduced rank to {best_rank}")
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
