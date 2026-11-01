"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 145  # seconds, safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication."""
    deadline = START_TIME + TIME_LIMIT - 5  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (9, rank)  
    # W has shape (p*n, rank) = (6, rank)
    # The decomposition: for each (i,j,k), output (i,j,k) = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # i in [0,n), j in [0,m), k in [0,p)
    
    # Try various ranks from low to high, attempting to find valid decompositions
    best_result = None
    
    # First, try a direct search with rational/integer/half-integer entries
    # For small dimensions, we can use exhaustive or randomized search
    
    for target_rank in range(n*m*p, n*m*p - 1, -1):
        # Try rank from n*m*p down
        # Use randomized approach to find decompositions
        found = _find_decomposition(n, m, p, target_rank)
        if found:
            U, V, W = found
            return {"U": U, "V": W and V or V, "W": W} if False else {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _find_decomposition(n, m, p, rank):
    """Attempt to find a decomposition of given rank using iterative refinement."""
    if rank < 3:
        return None
    
    # The constraint: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other index combinations that don't correspond to valid (i,j,k), the sum should be 0
    # Wait - the output W[k*n+i, r] is indexed by (k,i), so we need:
    # For each output (i, j, k), the product sum should give C_{(i,j),k} which is A[i,:] dot B[:,k]
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = A_{i,:} B_{:,k}
    # But A and B are variable! The decomposition must work for ALL A and B.
    # So T_{ijk} must represent the bilinear form properly.
    
    # Actually, for matrix multiplication C = AB where A is (n x m) and B is (m x p):
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i, j, k) and T_{ijk} = delta_{jk}... no.
    # The multiplication tensor: T = sum_{i,j,k} e_i (in output) x e_j (in A) x e_k (in B)
    # where the output element (i,k) gets contribution from (i,j) in A and (j,k) in B.
    # So T_{(i,k), (i,j), (j,k)} = 1.
    
    # In our factorization with U indexed by (i*m+j), V by (j*p+k), W by (k*n+i):
    # For each term c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if and only if the indices are consistent, i.e., this contributes to output (i,k) from inputs (i,j) and (j,k).
    # So for fixed (i,j,k): the product should be 1 (since A_{ij} * B_{jk} contributes to C_{ik}).
    # And for inconsistent index combinations, it should be 0.
    # But wait - U, V, W are fixed tensors. The product U[a,c]*V[b,c]*W[d,c] gives a coefficient.
    # The full bilinear computation is: for output index (i,k), value = sum over (j) of [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]] * A[i*m+j] * B[j*p+k]
    # So the coefficient of A[i*m+j]*B[j*p+k] in output (k*n+i) must be 1 for valid (i,j,k) pairs.
    # But every (i,j,k) triple IS valid since i in [0,n), j in [0,m), k in [0,p).
    # So actually ALL combinations need coefficient 1!
    # That means: for ALL (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a much simpler constraint! Every triple gets coefficient 1.
    # So we need: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    #   sum_{c=1}^{rank} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is the rank of the n*m*p all-ones tensor under this specific factorization pattern.
    # Wait, that's not right either. Let me re-examine.
    
    # The output is W[k*n+i, r] which represents C[i,k].
    # The computation: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' V[j*p+k',c] * B[j,k']) ... no.
    
    # Let me be more careful. In tensor decomposition of matrix mult:
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    # We write: C_{ik} = sum_c U_i,c * V_k,c * W_{j,c} ... that's not standard.
    
    # Standard: C_{ik} = sum_c U_{i,j,c} ... no, that overcounts.
    
    # The standard tensor decomposition: 
    # C_{ik} = sum_{c=1}^{rank} A_{i, f(c)} * B_{g(c), k} * alpha_c
    # No, that's not right either.
    
    # Correct form: C_{ik} = sum_c (sum_j U^{(c)}_{ij} A_{ij}) (sum_k' V^{(c)}_{jk'} B_{jk'}) ... 
    
    # Actually the cleanest way: 
    # C_{ik} = sum_j A_{ij} B_{jk} = sum_c [sum_j U_c(i,j) A_{ij}] [sum_k' V_c(j,k') B_{jk'}] ... no.
    
    # Let me use the problem's convention directly:
    # U has shape (n*m, rank): U[a, c] where a = i*m + j
    # V has shape (m*p, rank): V[b, c] where b = j*p + k  
    # W has shape (p*n, rank): W[d, c] where d = k*n + i
    
    # The output W[k*n+i, r] means output index d = k*n+i corresponds to C[i,k].
    # The bilinear form: for each output index d = k*n+i (i.e., C[i,k]):
    # C[i,k] = sum_c W[d, c] * (sum over... ) 
    
    # I think the convention is:
    # C_{ik} = sum_c U_{i*, c} * V_{*k, c} ... but U depends on both i and j.
    
    # Let me just go with: the scalar product sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] 
    # must equal 1 for all valid (i,j,k), because A[i*m+j]*B[j*p+k] has coefficient 
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] in C[k*n+i].
    # And that coefficient must be 1.
    
    # So the problem reduces to: find rank-r matrices U (6xr), V (9xr), W (6xr) with half-integer entries
    # such that for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
    
    # With rank = n*m*p = 18, the schoolbook trivially works (one basis vector per (i,j,k)).
    # Can we do better? For the all-ones constraint structure, maybe.
    
    # Try randomized search for lower ranks
    num_attempts = 5000
    rng = random.Random(42)
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization with small half-integer values
        U = [[rng.choice([0, 0.5, 1.0, -0.5, -1.0, 0.25, 0.75]) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.choice([0, 0.5, 1.0, -0.5, -1.0, 0.25, 0.75]) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.choice([0, 0.5, 1.0, -0.5, -1.0, 0.25, 0.75]) for _ in range(rank)] for _ in range(p*n)]
        
        # Check constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to fractions for exactness
            U_frac = [[_to_frac(x) for x in row] for row in U]
            V_frac = [[_to_frac(x) for x in row] for row in V]
            W_frac = [[_to_frac(x) for x in row] for row in W]
            
            # Verify with exact arithmetic
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p, rank):
                return (U_frac, V_frac, W_frac)
    
    return None


def _to_frac(x):
    """Convert float to Fraction, preferring half-integer/half values."""
    f = Fraction(x).limit_denominator(1000000)
    # Round to nearest quarter
    q = Fraction(int(round(float(f) * 4)), 4)
    return q


def _verify_exact(U, V, W, n, m, p, rank):
    """Verify decomposition with exact arithmetic."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
