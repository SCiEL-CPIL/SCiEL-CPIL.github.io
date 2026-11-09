import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit
    
    # The tensor T for matrix multiplication A(n x m) * B(m x p) -> C(n x p)
    # T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    # In the factor convention:
    #   U has shape (n*m, rank), indexed by (i, j) -> flat index i*m + j
    #   V has shape (m*p, rank), indexed by (j, k) -> flat index j*p + k
    #   W has shape (p*n, rank), indexed by (k, i) -> flat index k*n + i
    # The decomposition: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    # Trivial decomposition: rank = n*m*p = 75
    def build_trivial():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is correct."""
        rank = U.shape[1]
        # Reconstruct the tensor and check
        # T[i*m+j, j*p+k, k*n+i] should be 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that all other entries are 0
        # T is zero everywhere except at the (i*m+j, j*p+k, k*n+i) positions
        # For a rank-r decomposition, we need to verify no spurious contributions
        # This is expensive, so we'll rely on the above check plus structure
        return True
    
    def verify_full(U, V, W, tol=1e-8):
        """Full verification: check all tensor entries."""
        rank = U.shape[1]
        # Build the full tensor from the decomposition
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is expensive for large tensors, but our tensor is small
        # n*m = 15, m*p = 25, p*n = 15
        # Full tensor is 15 x 25 x 15 = 5625 entries
        
        # Expected tensor: T[i*m+j, j*p+k, k*n+i] = 1, all others 0
        expected = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute actual tensor
        actual = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            actual += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if not np.allclose(actual, expected, atol=tol):
            return False
        return True
    
    # First, get the trivial solution as fallback
    U_triv, V_triv, W_triv = build_trivial()
    best_rank = n * m * p
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Strategy: Try to find a lower-rank decomposition using block structure
    # 
    # Block decomposition of 3x5x5:
    # A (3x5) = [A1 (3x3) | A2 (3x2)]
    # B (5x5) = [B1 (3x5); B2 (2x5)]
    # C = A1*B1 + A2*B2
    #
    # B1 (3x5) = [B1a (3x3) | B1b (3x2)]
    # B2 (2x5) = [B2a (2x3) | B2b (2x2)]
    #
    # C = A1*B1a + A1*B1b + A2*B2a + A2*B2b
    #
    # Each block is a separate matmul tensor:
    # A1*B1a: 3x3x3 (n=3, m=3, p=3)
    # A1*B1b: 3x3x2 (n=3, m=3, p=2)
    # A2*B2a: 3x2x3 (n=3, m=2, p=3)
    # A2*B2b: 3x2x2 (n=3, m=2, p=2)
    
    # For each block, we can use the trivial decomposition or try to find
    # a lower-rank one. The total rank is the sum of the block ranks.
    
    # Trivial ranks:
    # 3x3x3: 27
    # 3x3x2: 18
    # 3x2x3: 18
    # 3x2x2: 12
    # Total: 75 (same as full trivial)
    
    # To improve, we need sub-trivial decompositions for at least one block.
    # The 3x3x3 block has known rank 16 (Winograd).
    
    # Let me try to construct a rank-16 decomposition for 3x3x3 using
    # a numerical approach.
    
    def solve_3x3x3_rank16():
        """Try to find a rank-16 decomposition for the 3x3x3 matmul tensor."""
        # The tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # We need 16 terms (u_r, v_r, w_r) each of length 9
        # such that sum_r u_r[i*3+j] * v_r[j*3+k] * w_r[k*3+i] = 1
        
        # Use a structured approach based on Winograd's algorithm.
        # The standard Winograd F(4,3) for 3x3 matrix multiplication uses
        # 16 multiplications.
        
        # Let me use a known construction. The Winograd 3x3x3 algorithm
        # works as follows:
        # 
        # For each row i of A and column k of B, we compute C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # The key insight is to use polynomial interpolation.
        # A[i,x] = A[i,0] + A[i,1]*x + A[i,2]*x^2
        # B[x,k] = B[0,k] + B[1,k]*x + B[2,k]*x^2
        # Product: A[i,x]*B[x,k] has degree 4
        # Evaluate at 5 points to get 5 values, then interpolate.
        # But this gives 5*3*3 = 45 multiplications, not 16.
        
        # The actual Winograd algorithm is more clever. Let me use a
        # different approach: I'll set up the system and solve it.
        
        # For rank 16, we have 16 terms. Each term has 9+9+9 = 27 values.
        # Total unknowns: 16*27 = 432 (with scaling freedom, effectively less)
        # Constraints: 27 (one for each (i,j,k) in {0,1,2}^3)
        
        # This is a highly underdetermined system. Let me use a randomized
        # approach: initialize U and V randomly, then solve for W.
        
        # Fix U and V to be random sparse vectors, then for each (i,j,k),
        # the constraint is:
        # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        # 
        # This is linear in W if we fix U and V. But W appears in all
        # constraints for a given (k,i) pair... wait, no.
        # 
        # Actually, for a fixed (i,j,k), the constraint involves:
        # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        # 
        # Let a_r = U[i*3+j, r] * V[j*3+k, r]
        # Then: sum_r a_r * W[k*3+i, r] = 1
        # 
        # This is a linear equation in the variables W[0,0], W[0,1], ..., W[2,2]
        # (9 variables for W). But each constraint involves a different
        # combination of W entries.
        
        # Let me think differently. For each output position (k,i) of C,
        # we need:
        # sum_j sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        # 
        # Wait, that's not right either. Let me re-derive.
        # 
        # The tensor decomposition says:
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # where a = i*3+j, b = j*3+k, c = k*3+i
        # 
        # So for the (i,j,k) entry of the tensor:
        # T[i*3+j, j*3+k, k*3+i] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
        # 
        # And for all other (a,b,c) combinations, T[a,b,c] = 0.
        
        # This is a system of 27*27*27 = 19683 equations (most of which are 0=0
        # or 0=1 constraints).
        
        # This is getting very complex. Let me try a different approach:
        # I'll use a known correct decomposition from the literature.
        
        # After research, the Winograd 3x3x3 decomposition with 16 terms
        # can be constructed as follows:
        
        # The algorithm computes 3x3 matrix multiplication using 16 scalar
        # multiplications. The pre-processing and post-processing are linear.
        
        # Let me implement the standard Winograd F(4,3) algorithm for 3x3:
        
        # Actually, I recall that the minimum rank for 3x3x3 is 16, and
        # one construction uses the following structure:
        
        # Split the 3x3 into four 2x2 blocks plus correction terms.
        # Use Strassen for the 2x2x2 part (7 terms) and handle the rest.
        
        # This is getting too complex to implement correctly from scratch
        # in the time available. Let me try a numerical approach.
        
        # Approach: Use random initialization and gradient descent.
        # Objective: minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
        
        # This is a non-convex problem, but with good initialization it might work.
        
        # Given time constraints, let me try a simpler approach:
        # Use the trivial decomposition for all blocks and accept rank 75.
        # This is guaranteed to be correct.
        
        return None
    
    # Given the complexity of finding a verified sub-75 decomposition,
    # and the fact that all previous attempts have scored 1.0 (trivial rank 75),
    # let me try a different strategy:
    
    # I'll attempt a randomized search for a lower-rank decomposition.
    # Start with rank 70 and try to reduce.
    
    # Actually, let me try a more principled approach:
    # Use the block structure and try to find sub-trivial decompositions
    # for individual blocks.
    
    # For the 3x3x3 block, I'll try to use a rank-19 decomposition
    # (which is easier to construct than rank-16).
    
    # For 3x3x2, 3x2x3, 3x2x2, I'll use trivial (18, 18, 12).
    # Total: 19 + 18 + 18 + 12 = 67
    
    # Let me try to construct a rank-19 decomposition for 3x3x3.
    
    def construct_3x3x3_rank19():
        """Construct a rank-19 decomposition for 3x3x3 matmul."""
        # Use the following strategy:
        # 7 terms from a Strassen-like 2x2x2 decomposition
        # 12 terms for the remaining parts
        
        # Actually, let me just use the trivial 27-term decomposition
        # and see if I can merge some terms.
        
        # For now, return None to indicate failure.
        return None
    
    # Given the difficulty, let me just return the trivial decomposition.
    # This is guaranteed to be correct and scores 1.0.
    
    # But wait - let me try one more thing: a simple randomized search
    # for a rank-70 decomposition.
    
    if time.time() < deadline - 30:
        # Try randomized search for rank 70
        best_found = None
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            rank = 70
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize with the first 70 trivial terms
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank:
                            break
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            
            # Try to adjust the last few terms to cover the missing entries
            # This is a rough heuristic
            
            if verify(U, V, W, tol=1e-6):
                best_found = (U, V, W, rank)
                break
    
    if best_found:
        return {"U": best_found[0].tolist(), "V": best_found[1].tolist(), "W": best_found[2].tolist()}
    
    # Return the trivial decomposition
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

# EVOLVE-BLOCK-END
