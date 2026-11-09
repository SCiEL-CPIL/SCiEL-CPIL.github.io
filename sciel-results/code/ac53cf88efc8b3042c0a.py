"""Find rank-minimal exact tensor decomposition for 2x4 by 4x8 matrix multiplication."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # W[k*n + i, r] - trace convention
    
    # The multiplication tensor: for each (i,j,k), we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # to contribute to the (i,k) output element from input (i,j) and (j,k)
    
    # Schoolbook gives rank 64. Let's search for better.
    
    # Known: for matrix multiplication, Strassen-like methods can help.
    # For 2x4x8, let's try systematic search.
    
    # The tensor T has entries T[(i,j), (j,k), (k,i)] = 1 for all i,j,k
    # A rank-r decomposition: sum_{r=1}^{R} U_r[i,j] * V_r[j,k] * W_r[k,i] = T[i,j,k]
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # But actually the tensor rank lower bound is more subtle.
    
    # Let's try a structured approach: use the fact that we can combine some terms.
    
    # Strategy: Try random constructions with small integer/half-integer values
    # and verify exactness.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        R = len(U[0]) if U else 0
        # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # For each (i, k), the output should be sum_j A[i,j]*B[j,k]
        # The tensor entry T[i,j,k] should equal 1 for the "multiplication" pattern
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(R):
                        total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if total != 1:
                        return False
        return True
    
    def verify_full_decomposition(U, V, W, n, m, p):
        """Full verification: check that the decomposition computes matrix multiplication exactly."""
        R = len(U[0]) if U else 0
        
        # For random test matrices, check if C = A*B is computed correctly
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Our decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # Wait, that's not right. Let me reconsider.
        
        # Actually, the standard tensor decomposition for matrix mult:
        # C[i,k] = sum_{r=1}^{R} W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        
        # No wait, that's not standard either. Let me think again.
        
        # The multiplication tensor M satisfies:
        # M[(i,j), (j,k), (k,i)] = 1
        # A rank-1 term is U[i,j] * V[j,k] * W[k,i]
        # So sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] should equal M[(i,j),(j,k),(k,i)] = 1
        
        # But actually for matrix multiplication, we want:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r W_r[k,i] * (sum_j U_r[i,j] A[i,j]) * (sum_j V_r[j,k] B[j,k])
        
        # Hmm, that doesn't seem right dimensionally. Let me use the standard formulation.
        
        # Standard: C[i,k] = sum_{j=1}^{m} A[i,j] B[j,k]
        # Tensor: T[i,j,k] = 1 (for the multiplication pattern)
        # Rank-1: u[i,j] * v[j,k] * w[k,i]
        # We need: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
        
        # This is what verify_decomposition above checks. Let's use that.
        
        return verify_decomposition(U, V, W, n, m, p)
    
    # Try to find a decomposition with rank < 64
    # Start from rank 32 (lower bound) and go up
    
    # First, let's try some known constructions.
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8.
    # For 2x4x8, we might be able to do better than 64.
    
    # Let's try a systematic random search with verification.
    
    random.seed(42)
    
    best_rank = n * m * p  # start with schoolbook
    best_decomp = None
    
    # Generate schoolbook as fallback
    def schoolbook():
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
    
    # Try ranks from 32 to 63
    for target_rank in range(32, best_rank):
        if time.time() > deadline:
            break
        
        # Try several random attempts for this rank
        attempts = 0
        max_attempts = 50  # limit attempts per rank
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random decomposition with small integer/half-integer values
            U = []
            V = []
            W = []
            
            for _ in range(n * m):
                row = []
                for _ in range(target_rank):
                    # Use small values: -1, 0, 1, 0.5, -0.5
                    val = random.choice([-1, 0, 0, 1, 0.5, -0.5])
                    row.append(val)
                U.append(row)
            
            for _ in range(m * p):
                row = []
                for _ in range(target_rank):
                    val = random.choice([-1, 0, 0, 1, 0.5, -0.5])
                    row.append(val)
                V.append(row)
            
            for _ in range(p * n):
                row = []
                for _ in range(target_rank):
                    val = random.choice([-1, 0, 0, 1, 0.5, -0.5])
                    row.append(val)
                W.append(row)
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_decomp = {"U": U, "V": V, "W": W}
                break  # Found one for this rank, try lower
    
    # If no better found, return schoolbook
    if best_decomp is not None:
        return best_decomp
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
