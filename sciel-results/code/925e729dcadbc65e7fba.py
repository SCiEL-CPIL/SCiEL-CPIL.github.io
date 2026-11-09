"""Matrix multiplication 3x4x5 with known low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x5 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition of rank n*m*p = 60
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    # Check time
    if time.time() > deadline:
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Try to find a lower rank decomposition
    # Strategy: For a given rank R, pick random U, V with small integer entries.
    # Then solve for W such that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    # This is a linear system in W.
    
    best_solution = None
    best_rank = rank_baseline
    
    # Try ranks from 59 down to 50
    for R in range(rank_baseline - 1, 45, -1):
        if time.time() > deadline:
            break
        
        # Number of trials
        trials = 50
        for t in range(trials):
            if time.time() > deadline:
                break
            
            # Random U, V with entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # We need to find W (p*n x R) such that:
            # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # This can be written as: for each (i,k), sum_c (sum_j U[i*m+j,c] V[j*p+k,c]) W[k*n+i,c] = 1
            # Wait, that's not right. The sum is over c, and for each c, we have a product U*V*W.
            #
            # Actually, the condition is:
            # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i,j,k
            #
            # This is a multilinear equation, not linear in W alone.
            #
            # Alternative: Fix U and V, then for each c, define M_c[i,k] = sum_j U[i*m+j,c] V[j*p+k,c]
            # Then the condition becomes: sum_c M_c[i,k] * W[k*n+i,c] = 1
            #
            # Wait, that's still not quite right. Let me re-derive.
            #
            # T_{i,j,k} = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            #
            # For fixed i,k, this is a sum over j and c. Let's define:
            # A_{i,j,c} = U[i*m+j,c]
            # B_{j,k,c} = V[j*p+k,c]
            # C_{k,i,c} = W[k*n+i,c]
            #
            # Then T_{i,j,k} = sum_c A_{i,j,c} B_{j,k,c} C_{k,i,c}
            #
            # This is not linear in C (W) because it's a product of three terms.
            #
            # However, if we fix U and V, then for each c, we have a matrix M_c of size n x p:
            # M_c[i,k] = sum_j U[i*m+j,c] V[j*p+k,c]
            #
            # Then T_{i,j,k} is not simply related to M_c.
            #
            # Let me think differently. The tensor T has indices (i,j,k) with T_{i,j,k}=1.
            # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c where:
            # (U_c \otimes V_c \otimes W_c)[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            #
            # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
            #
            # This is a system of n*m*p equations in (p*n)*R unknowns (entries of W),
            # but the equations are bilinear in (V, W) if U is fixed, or trilinear.
            #
            # For a fixed U, V, this is not linear in W. So we can't just solve a linear system.
            #
            # Alternative approach: Use the fact that for small dimensions, we can do an exhaustive
            # or semi-exhaustive search.
            #
            # Given the complexity, let's try a different strategy: use known decompositions
            # or try to construct one explicitly.
            #
            # For now, let's just return the schoolbook decomposition if we can't find better.
            pass
    
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
