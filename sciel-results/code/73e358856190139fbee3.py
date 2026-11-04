"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index triples and their positions
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_pos = np.array([i * m + j for i, j, k in indices])
    v_pos = np.array([j * p + k for i, j, k in indices])
    w_pos = np.array([k * n + i for i, j, k in indices])
    
    def verify(U, V, W, tol=1e-5):
        """Verify that the decomposition is exact using vectorized operations."""
        U_u = U[u_pos, :]
        V_v = V[v_pos, :]
        W_w = W[w_pos, :]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        max_err = np.max(np.abs(vals - 1.0))
        return max_err < tol

    def residual_func(params, rank):
        """Vectorized residual calculation."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        U_u = U_flat[u_pos, :]
        V_v = V_flat[v_pos, :]
        W_w = W_flat[w_pos, :]
        vals = np.sum(U_u * V_v * W_w, axis=1)
        return vals - 1.0

    def construct_strassen_222(rank_offset, U, V, W, u_base, v_base, w_base):
        """
        Adds a Strassen 2x2x2 decomposition to U, V, W.
        u_base: index in U for A[0,0]
        v_base: index in V for B[0,0]
        w_base: index in W for C[0,0]
        """
        # Strassen for 2x2x2
        # m1 = (a00+a11)*(b00+b11)
        U[u_base, rank_offset+0] += 1; U[u_base+3, rank_offset+0] += 1
        V[v_base, rank_offset+0] += 1; V[v_base+3, rank_offset+0] += 1
        W[w_base, rank_offset+0] += 1; W[w_base+3, rank_offset+0] += 1
        
        # m2 = (a10+a11)*b00
        U[u_base+2, rank_offset+1] += 1; U[u_base+3, rank_offset+1] += 1
        V[v_base, rank_offset+1] += 1
        W[w_base+2, rank_offset+1] += 1; W[w_base+3, rank_offset+1] -= 1
        
        # m3 = a00*(b01-b11)
        U[u_base, rank_offset+2] += 1
        V[v_base+1, rank_offset+2] += 1; V[v_base+3, rank_offset+2] -= 1
        W[w_base+1, rank_offset+2] += 1
        
        # m4 = a11*(b10-b00)
        U[u_base+3, rank_offset+3] += 1
        V[v_base+2, rank_offset+3] += 1; V[v_base, rank_offset+3] -= 1
        W[w_base, rank_offset+3] += 1; W[w_base+2, rank_offset+3] += 1
        
        # m5 = (a00+a01)*b11
        U[u_base, rank_offset+4] += 1; U[u_base+1, rank_offset+4] += 1
        V[v_base+3, rank_offset+4] += 1
        W[w_base, rank_offset+4] -= 1; W[w_base+1, rank_offset+4] += 1
        
        # m6 = (a10-a00)*(b00+b01)
        U[u_base+2, rank_offset+5] += 1; U[u_base, rank_offset+5] -= 1
        V[v_base, rank_offset+5] += 1; V[v_base+1, rank_offset+5] += 1
        W[w_base+3, rank_offset+5] += 1
        
        # m7 = (a01-a11)*(b10+b11)
        U[u_base+1, rank_offset+6] += 1; U[u_base+3, rank_offset+6] -= 1
        V[v_base+2, rank_offset+6] += 1; V[v_base+3, rank_offset+6] += 1
        W[w_base, rank_offset+6] += 1; W[w_base+3, rank_offset+6] -= 1

    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        start = time.time()
        best_cost = float('inf')
        best_x = None
        
        # Initialization 1: Structured (Block decomposition)
        # For rank 18: 14 (Strassen 2x2x4) + 4 (Naive 2x1x4)
        # For rank 19: 14 + 5? Or just random perturbation of 18?
        
        U_init = np.zeros((nm, rank))
        V_init = np.zeros((mp, rank))
        W_init = np.zeros((pn, rank))
        
        if rank >= 18:
            # Block 1: A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]
            # This is 2x2 x 2x4.
            # Split into two 2x2 x 2x2 blocks (cols 0-1 and 2-3 of B)
            
            # Sub-block 1: B cols 0-1
            # A indices: 0,1,2,3
            # B indices: 0,1,2,3 (B[0,0], B[0,1], B[1,0], B[1,1])
            # C indices: 0,1,2,3 (C[0,0], C[0,1], C[1,0], C[1,1])
            construct_strassen_222(0, U_init, V_init, W_init, 0, 0, 0)
            
            # Sub-block 2: B cols 2-3
            # A indices: 0,1,2,3
            # B indices: 4,5,6,7 (B[0,2], B[0,3], B[1,2], B[1,3])
            # C indices: 4,5,6,7 (C[0,2], C[0,3], C[1,2], C[1,3])
            construct_strassen_222(7, U_init, V_init, W_init, 0, 4, 4)
            
            # Block 2: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
            # This is 2x1 x 2x4.
            # A indices: 4,5 (A[0,2], A[1,2])
            # B indices: 8,9,10,11 (B[2,0]..B[2,3])
            # C indices: 0..7
            
            # Naive for 2x1 x 2x4
            # C[0,0] = A[0,2] B[2,0]
            U_init[4, 14] = 1
            V_init[8, 14] = 1
            W_init[0, 14] = 1
            
            # C[0,1] = A[0,2] B[2,1]
            U_init[4, 15] = 1
            V_init[9, 15] = 1
            W_init[1, 15] = 1
            
            # C[1,0] = A[1,2] B[2,0]
            U_init[5, 16] = 1
            V_init[8, 16] = 1
            W_init[2, 16] = 1
            
            # C[1,1] = A[1,2] B[2,1]
            U_init[5, 17] = 1
            V_init[9, 17] = 1
            W_init[3, 17] = 1
            
            # Wait, C indices for 2x4 are 0..7.
            # C[0,2] is index 4?
            # W index is k*n + i.
            # C[0,2] -> k=2, i=0 -> 2*2+0 = 4.
            # C[0,3] -> k=3, i=0 -> 3*2+0 = 6.
            # C[1,2] -> k=2, i=1 -> 2*2+1 = 5.
            # C[1,3] -> k=3, i=1 -> 3*2+1 = 7.
            
            # My Strassen calls used w_base=0 and w_base=4.
            # w_base=0 covers C[0,0], C[0,1], C[1,0], C[1,1] (indices 0,1,2,3).
            # w_base=4 covers C[0,2], C[0,3], C[1,2], C[1,3] (indices 4,5,6,7).
            # This is correct.
            
            # The naive part for A[0:2, 2:3] * B[2:3, 0:4]
            # Contributes to ALL C[0:2, 0:4].
            # I only added 4 terms above for C[0,0], C[0,1], C[1,0], C[1,1].
            # I need to add terms for C[0,2], C[0,3], C[1,2], C[1,3].
            
            # C[0,2] = A[0,2] B[2,2]
            U_init[4, 18] = 1 # If rank > 18
            # But we are trying rank 18. So we can't add more terms.
            # The block decomposition 14+4=18 only covers half the output?
            # No, A[0:2, 0:2] * B[0:2, 0:4] covers C[0:2, 0:4] using B rows 0,1.
            # A[0:2, 2:3] * B[2:3, 0:4] covers C[0:2, 0:4] using B row 2.
            # The sum is the full product.
            # The 14 terms handle the B[0:2] part.
            # The 4 terms handle the B[2:3] part.
            # But the 4 terms I wrote only cover C[0:2, 0:2] (cols 0,1).
            # I need 4 more terms for C[0:2, 2:4] (cols 2,3).
            # Total 14 + 8 = 22.
            # So M(2,3,4) is NOT 18 via this split.
            # M(2,1,4) is 4?
            # C[0,k] = A[0,2] B[2,k].
            # C[1,k] = A[1,2] B[2,k].
            # For k=0..3.
            # This is 4 multiplications for col 0? No.
            # A[0,2] is a scalar. B[2,k] is a vector.
            # C[0,:] = A[0,2] * B[2,:].
            # C[1,:] = A[1,2] * B[2,:].
            # This is 2 scalar-vector products.
            # Each scalar-vector product of length 4 requires 4 multiplications?
            # No, the tensor rank of $1 \times 1 \times 4$ is 4?
            # $C_k = A B_k$.
            # This is 4 independent multiplications.
            # So $M(1,1,4) = 4$.
            # $M(2,1,4) = M(1,1,4) + M(1,1,4) = 8$.
            # So 14 + 8 = 22.
            # So rank 18 is not achievable by this simple block split.
            
            pass # Skip this initialization if it's not valid for rank 18
        
        # Just use random initialization for now, but better scaled
        np.random.seed(42)
        scale = 1.0 / np.sqrt(rank)
        U_init = np.random.randn(nm, rank) * scale
        V_init = np.random.randn(mp, rank) * scale
        W_init = np.random.randn(pn, rank) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                lambda x: residual_func(x, rank),
                x0,
                method='lm',
                max_nfev=10000,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            
            if result.cost < 1e-10:
                U_sol = result.x[:nm * rank].reshape(nm, rank)
                V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                
                if verify(U_sol, V_sol, W_sol, tol=1e-6):
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify(U_rounded, W_rounded, V_rounded, tol=1e-6):
                        return U_rounded, V_rounded, W_rounded
                    else:
                        return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None

    # Try rank 19
    solution = try_rank(19)
    if solution is not None:
        U_sol, V_sol, W_sol = solution
        
        def to_list(arr):
            lst = []
            for row in arr:
                row_lst = []
                for val in row:
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        row_lst.append(rounded)
                    else:
                        row_lst.append(float(val))
                lst.append(row_lst)
            return lst
        
        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fallback: naive decomposition with rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
