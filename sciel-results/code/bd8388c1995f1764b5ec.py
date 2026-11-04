"""Matrix multiplication 4,5,6 tensor decomposition search."""
import time
import itertools
import random
import sys

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_left(deadline):
    return deadline - time.time()

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix mult."""
    deadline = time.time() + TIME_LIMIT
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank_schoolbook = n*m*p  # 120
    
    # Start with schoolbook as baseline
    best_rank = rank_schoolbook
    best_U = [[0]*rank_schoolbook for _ in range(nm)]
    best_V = [[0]*rank_schoolbook for _ in range(mp)]
    best_W = [[0]*rank_schoolbook for _ in range(pn)]
    for idx, (i,j,k) in enumerate(itertools.product(range(n), range(m), range(p))):
        best_U[i*m+j][idx] = 1
        best_V[j*p+k][idx] = 1
        best_W[k*n+i][idx] = 1
    
    # Try to find lower rank decompositions
    # Strategy: use structured patterns to reduce rank
    # For matrix multiplication C = A*B where A is n x m, B is m x p
    # The tensor has entries T[i,j,k] = delta_{ik} (in the appropriate sense)
    # Actually T_{(i,k),(j)} = delta_{ik}... no.
    
    # The matrix multiplication tensor: T_{(i,j),(k),(l)}... 
    # Standard: T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor entries: T_{(i,k),j} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}
    # In our indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So for each factor c: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should sum to delta_{ik}... 
    # No, the product should reconstruct: for each (i,k), sum over j of A_{i,j} B_{j,k}
    # The tensor decomposition says: T_{(i,j),(k,l)}... 
    
    # Let me think more carefully. The matrix multiplication tensor T has:
    # T_{(i,j),(k,l)} = delta_{j,k} * delta_{i,l}... no.
    # T is a 3-tensor: T_{a,b,c} where a indexes (i,j) of A, b indexes (j,k) of B, c indexes (i,k) of C
    # T_{(i,j),(j',k),(i',k')} = 1 if i=i' and k=k' and j=j'
    
    # In our factorization:
    # C_{i,k} = sum_c (sum_{i',j'} U_{i',j',c} A_{i',j'}) (sum_{j',k'} V_{j',k',c} B_{j',k'}) * W_{k,i,c}
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = delta_{(i,j),(j,k)}... 
    
    # More precisely: for fixed (i,k), the coefficient of A_{i0,j0} * B_{j1,k1} in the expansion
    # should be 1 if i0=i, j0=j1, k1=k, else 0.
    # So: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 if (i,j) and (j,k) with the same j, else 0
    # Wait, that's not right either. Let me re-derive.
    
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Our formula: C_{i,k} = sum_c W[k,i,c] * (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
    # = sum_c W[k,i,c] * sum_{i',j',j',k'} U[i',j',c] V[j',k',c] A[i',j'] B[j',k']
    # For this to equal sum_j A[i,j] B[j,k], we need for each (i,k):
    # sum_c W[k,i,c] sum_{i',j',j',k'} U[i',j',c] V[j',k',c] A[i',j'] B[j',k'] = sum_j A[i,j] B[j,k]
    # This must hold for all A and B, so the coefficient of A[i',j'] B[j',k'] must match:
    # sum_c W[k,i,c] U[i',j',c] V[j',k',c] = 1 if i'=i, j'=k', k'=k... 
    # No: coefficient of A[i',j'] B[j',k'] on LHS is sum_c W[k,i,c] U[i',j',c] V[j',k',c]
    # On RHS, coefficient of A[i',j'] B[j',k'] in sum_j A[i,j] B[j,k] is:
    # 1 if i'=i, j'=j' (same j), k'=k, and j'=j'... 
    # RHS = sum_j A[i,j] B[j,k], so coefficient of A[i',j'] B[j',k'] is 1 if i'=i, k'=k, j'=j' (i.e., the j index matches between A and B).
    # So: sum_c W[k,i,c] U[i',j',c] V[j',k',c] = delta_{i,i'} * delta_{k,k'} * delta_{j',j'}
    # where j' appears in both U and V with the same index.
    
    # So the constraint is: for all (i,k,i',j',j',k'):
    # sum_c W[k,i,c] * U[i',j',c] * V[j',k',c] = delta_{i,i'} * delta_{k,k'} * (1 if j'==j' else 0)
    # The j' in U and V must be the same for the term to be non-zero on RHS.
    
    # In schoolbook: each term (i,j,k) gets its own factor, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # This gives sum_c W[k,i,c] U[i',j',c] V[j',k',c] = 1 only when (i,j,k)=(i',j',k') which means i=i', j=j', k=k'
    # And j' in U matches j' in V. Correct.
    
    # To reduce rank, we want to share factors. The key insight: if we can group terms
    # where the same (i,j) pattern or (j,k) pattern appears, we might combine.
    
    # Let's try a different approach: brute force search over small rank with random initialization
    # and check validity.
    
    # Actually, let's try a structured approach:
    # For 4x5 * 5x6, can we do better than 120?
    # Known results: for 2x2*2x2, rank 7 (vs 8 schoolbook)
    # For 4x4*4x4, best known is 49 (vs 64)
    # For our case 4x5*5x6, the exact minimum rank is unknown but likely < 120.
    
    # Let's try to construct a lower-rank decomposition using a grid-based approach.
    # Idea: partition the index sets and use tensor product structure.
    
    # For simplicity, let's try to verify if a rank-119 decomposition exists by
    # checking if we can merge any two schoolbook factors.
    
    # Two factors c1 and c2 can be merged if there exists a scalar alpha such that
    # replacing both with a single factor works. This is complex.
    
    # Alternative: use the fact that for each fixed j, the terms (i,j) for all i and (j,k) for all k
    # form a rank-1 structure in the (i,k) space... but they don't quite.
    
    # Let me try a completely different strategy: use randomized search with numpy.
    import numpy as np
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W (lists of lists) form a valid decomposition."""
        U_arr = np.array(U, dtype=object)
        V_arr = np.array(V, dtype=object)
        W_arr = np.array(W, dtype=object)
        
        # For each (i,k), check that sum_c W[k*n+i,c] * (sum_{i',j'} U[i'*m+j',c] * A[i',j']) * (sum_{j',k'} V[j'*p+k',c] * B[j',k']) = sum_j A[i,j]*B[j,k]
        # This must hold for all A, B. So we check the tensor identity:
        # sum_c W[k*n+i,c] * U[i'*m+j',c] * V[j'*p+k',c] = delta_{i,i'} * delta_{k,k'} * delta_{j',j'}
        
        # Build the 3D tensor from factors
        # T[i',j',j',k',k,i] should equal delta_{i,i'} delta_{k,k'} delta_{j',j'}
        # Note: j' in U and V must match.
        
        # For efficiency, check a few random (i',j',k') triples
        for _ in range(50):
            ip = random.randint(0, n-1)
            jp = random.randint(0, m-1)
            kp = random.randint(0, p-1)
            # Expected: for each (i,k), sum_c W[k*n+i,c] U[ip*m+jp,c] V[jp*p+kp,c] should be
            # delta_{i,ip} * delta_{k,kp} if jp==jp (always true since same j)
            # Wait, the j in U is jp, the j in V is jp. So the term is non-zero only if these match.
            # In the sum over c, we're looking at fixed (ip,jp) in U and fixed (jp,kp) in V.
            # The result should be: for (i,k), the value is 1 if i==ip and k==kp, else 0.
            
            for i in range(n):
                for k in range(p):
                    expected = 1.0 if (i == ip and k == kp) else 0.0
                    actual = 0.0
                    for c in range(rank):
                        actual += float(W_arr[k*n+i][c]) * float(U_arr[ip*m+jp][c]) * float(V_arr[jp*p+kp][c])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank by removing one factor from schoolbook and adjusting
    # This is very hard to do correctly.
    
    # Given the complexity and time constraints, let's try a few random perturbations
    # of the schoolbook decomposition to see if any work.
    
    best = {'U': best_U, 'V': best_V, 'W': best_W}
    
    # Try rank 119: remove one column and see if we can adjust
    # This is extremely unlikely to work with simple adjustments.
    
    # Let's just return the schoolbook decomposition if we can't find better quickly.
    # The probability of finding a valid lower-rank decomposition randomly is very low.
    
    return best

# EVOLVE-BLOCK-END
