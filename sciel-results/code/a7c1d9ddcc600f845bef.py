import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # Build the rank-36 schoolbook decomposition
    rank0 = n * m * p  # 36
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, idx] = 1.0
                V0[j * p + k, idx] = 1.0
                W0[k * n + i, idx] = 1.0
                idx += 1

    def verify_decomp(U, V, W, n, m, p, rank):
        """Verify that the tensor decomposition exactly matches matrix multiplication."""
        # Use multiple random tests for robustness
        for trial in range(3):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                u_col = U[:, r]  # shape (n*m,)
                v_col = V[:, r]  # shape (m*p,)
                w_col = W[:, r]  # shape (p*n,)
                # u_val for each (i): sum_j U[i*m+j, r] * A[i,j]
                # v_val for each (k): sum_j V[j*p+k, r] * B[j,k]
                # C[i,k] += W[k*n+i, r] * u_val_i * v_val_k
                u_vals = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        u_vals[i] += u_col[i * m + j] * A[i, j]
                v_vals = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        v_vals[k] += v_col[j * p + k] * B[j, k]
                # Add contribution
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += w_col[k * n + i] * u_vals[i] * v_vals[k]
            if not np.allclose(C_expected, C_decomp, atol=1e-10):
                return False
        return True

    # Rank reduction: try to remove pairs of terms
    # A term r can be removed if its column is in the span of the other columns
    # For a bilinear form, removing term r means we need to find coefficients
    # alpha such that U[:,r] = sum_{r' != r} alpha[r'] * U[:,r'], etc.
    # Actually for tensor rank reduction, we need all three simultaneously.
    # A simpler approach: try removing one term at a time by checking if
    # the 3-tensor slice is in the span of the others.
    
    # Better structural approach: try to remove pairs of terms.
    # If terms r1, r2 can be combined/removed, the rank drops by 2.
    
    # Let's use a greedy approach:
    # 1. Start with all 36 terms
    # 2. For each pair (r1, r2), check if removing both is valid
    #    (i.e., the decomposition still holds)
    # 3. If valid, remove them and continue
    
    # Actually, the most effective structural approach for this specific problem:
    # The 2x2x2 sub-problem has rank 7 (Strassen). 
    # For 2x2 x 2x9, we can group the 9 columns into 4 pairs + 1 single.
    # Each pair of columns can be handled with rank 7 (like 2x2 x 2x2).
    # The single column needs rank 4.
    # Total: 4*7 + 4 = 32.
    
    # But we can do better: try to find linear dependencies across blocks.
    # Let's construct the rank-32 solution first, then try to reduce further.
    
    # Construct rank-32 solution (4 pairs of rank 7 + 1 single of rank 4)
    def construct_rank32():
        rank = 32
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # Strassen-like rank-7 for 2x2 x 2x2
        # A = [[a,b],[c,d]], B = [[x,y],[z,w]]
        # C = A@B
        # 7 products:
        # m1 = (a+d)(x+w)
        # m2 = (a+d)(z+w) - wait, let me use the standard Strassen for 2x2
        # Actually for 2x2x2, Strassen uses 7 mults:
        # m1 = a(x+y)
        # m2 = (a+b)(z+w)  -- no, standard is:
        # m1 = a(x+y)
        # m2 = (a+b)(z+w) -- not quite
        # Let me use the correct 7-term decomposition for 2x2x2:
        # C[0,0] = a*x + b*z
        # C[0,1] = a*y + b*w
        # C[1,0] = c*x + d*z
        # C[1,1] = c*y + d*w
        #
        # 7-term:
        # m1 = (a+d)(x+w)
        # m2 = (a+d)(z+w)  -- no
        # 
        # Standard Strassen 7:
        # m1 = a(x+y)
        # m2 = (a+b)(z+w)  -- this gives a*z+a*w+b*z+b*w, not clean
        #
        # Let me just use the known rank-7 for 2x2x2:
        # The 7 terms are:
        # t1 = (a+d)(x+w)
        # t2 = (a+d)(z+w)  -- hmm, let me verify
        #
        # Actually, I'll use a verified 7-term scheme:
        # C00 = a*x + b*z
        # C01 = a*y + b*w
        # C10 = c*x + d*z
        # C11 = c*y + d*w
        #
        # 7 mults:
        # m1 = (a+d)(x+w)   -> a*x+a*w+d*x+d*w
        # m2 = (a+d)(z+w)   -> a*z+a*w+d*z+d*w
        # m3 = (a+b)(x+z)   -> a*x+a*z+b*x+b*z
        # m4 = (c+d)(y+w)   -> c*y+c*w+d*y+d*w
        # m5 = (a-b)(x-z)   -> a*x-a*z-b*x+b*z
        # m6 = (c-d)(y-w)   -> c*y-c*w-d*y+d*w
        # m7 = (c+d)(x-w)   -> c*x-c*w+d*x-d*w
        #
        # Let me check C00 = a*x + b*z:
        # From m1: a*x
        # From m5: a*x - b*x + b*z - a*z  -> contributes a*x - b*x + b*z - a*z
        # From m7: d*x - c*w + c*x - d*w  -> no b*z here
        # This is getting complicated. Let me just use the seed's approach which was verified.
        
        # Use the exact same Strassen-like terms as the seed program (which verified)
        # For pair (k1, k2):
        # m1: (a+d)(x+w) -> U: a,d; V: x=B[0,k1], w=B[1,k2]; W: C[0,k1],C[0,k2],C[1,k1],C[1,k2] all +1
        # m2: (a+d)(z+w) -> U: a,d; V: z=B[1,k1], w=B[1,k2]; W: C[0,k1]-1,C[0,k2]+1,C[1,k1]-1,C[1,k2]+1
        # m3: (a+b)(x+z) -> U: a,b; V: x=B[0,k1], z=B[1,k1]; W: C[1,k2]+1
        # m4: (c+d)(y+w) -> U: c,d; V: y=B[0,k2], w=B[1,k2]; W: C[1,k1]+1, C[1,k2]-1
        # m5: (a-b)(x-z) -> U: a,-b; V: x=B[0,k1], -z=-B[1,k1]; W: C[0,k1]+1
        # m6: (c-d)(y-w) -> U: c,-d; V: y=B[0,k2], -w=-B[1,k2]; W: C[1,k1]+1, C[1,k2]-1
        # m7: (c+d)(x-w) -> U: c,d; V: x=B[0,k1], -w=-B[1,k2]; W: C[0,k1]-1,C[0,k2]+1,C[1,k1]-1
        
        for pair_idx in range(4):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1
            
            # m1: (a+d)(x+w)
            U[0, col] = 1; U[3, col] = 1
            V[0 * p + k1, col] = 1; V[1 * p + k2, col] = 1
            W[k1 * n + 0, col] = 1; W[k2 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1; W[k2 * n + 1, col] = 1
            col += 1
            
            # m2: (a+d)(z+w)
            U[0, col] = 1; U[3, col] = 1
            V[1 * p + k1, col] = 1; V[1 * p + k2, col] = 1
            W[k1 * n + 0, col] = -1; W[k2 * n + 0, col] = 1
            W[k1 * n + 1, col] = -1; W[k2 * n + 1, col] = 1
            col += 1
            
            # m3: (a+b)(x+z)
            U[0, col] = 1; U[1, col] = 1
            V[0 * p + k1, col] = 1; V[1 * p + k1, col] = 1
            W[k2 * n + 1, col] = 1
            col += 1
            
            # m4: (c+d)(y+w)
            U[2, col] = 1; U[3, col] = 1
            V[0 * p + k2, col] = 1; V[1 * p + k2, col] = 1
            W[k1 * n + 1, col] = 1; W[k2 * n + 1, col] = -1
            col += 1
            
            # m5: (a-b)(x-z)
            U[0, col] = 1; U[1, col] = -1
            V[0 * p + k1, col] = 1; V[1 * p + k1, col] = -1
            W[k1 * n + 0, col] = 1
            col += 1
            
            # m6: (c-d)(y-w)
            U[2, col] = 1; U[3, col] = -1
            V[0 * p + k2, col] = 1; V[1 * p + k2, col] = -1
            W[k1 * n + 1, col] = 1; W[k2 * n + 1, col] = -1
            col += 1
            
            # m7: (c+d)(x-w)
            U[2, col] = 1; U[3, col] = 1
            V[0 * p + k1, col] = 1; V[1 * p + k2, col] = -1
            W[k1 * n + 0, col] = -1; W[k2 * n + 0, col] = 1
            W[k1 * n + 1, col] = -1
            col += 1
        
        # Last single column k=8: rank-4 schoolbook
        k = 8
        U[0, col] = 1; V[0 * p + k, col] = 1; W[k * n + 0, col] = 1; col += 1
        U[1, col] = 1; V[1 * p + k, col] = 1; W[k * n + 0, col] = 1; col += 1
        U[2, col] = 1; V[0 * p + k, col] = 1; W[k * n + 1, col] = 1; col += 1
        U[3, col] = 1; V[1 * p + k, col] = 1; W[k * n + 1, col] = 1; col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W, rank

    U, V, W, rank = construct_rank32()
    
    # Verify the rank-32 solution
    if not verify_decomp(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        U, V, W, rank = U0, V0, W0, rank0
    
    # Now try to reduce rank by removing pairs of terms
    # Key insight: if two terms r1, r2 have the property that their
    # combined contribution can be replicated by adjusting other terms,
    # we can remove them.
    
    # Structural reduction: try removing one term at a time by solving
    # for its representation in terms of others.
    # For a term r to be removable, we need:
    # U[:,r] in span(U[:,r'] for r' != r)
    # V[:,r] in span(V[:,r'] for r' != r)  
    # W[:,r] in span(W[:,r'] for r' != r)
    # AND the bilinear consistency must hold.
    
    # Simpler: try removing pairs (r1, r2) and check if the resulting
    # decomposition is still valid.
    
    # Build the set of active terms
    active = list(range(rank))
    
    # Try single-term removal first
    for r_remove in list(active):
        if time.time() > deadline - 30:
            break
        remaining = [r for r in active if r != r_remove]
        if len(remaining) < rank - 1:
            continue
        U_test = U[:, remaining]
        V_test = V[:, remaining]
        W_test = W[:, remaining]
        if verify_decomp(U_test, V_test, W_test, n, m, p, len(remaining)):
            active = remaining
            rank = len(remaining)
            U = U_test
            V = V_test
            W = W_test
    
    # Try pair-term removal
    changed = True
    while changed:
        changed = False
        for r1, r2 in combinations(list(active), 2):
            if time.time() > deadline - 20:
                break
            remaining = [r for r in active if r != r1 and r != r2]
            U_test = U[:, remaining]
            V_test = V[:, remaining]
            W_test = W[:, remaining]
            if verify_decomp(U_test, V_test, W_test, n, m, p, len(remaining)):
                active = remaining
                rank = len(remaining)
                U = U_test
                V = V_test
                W = W_test
                changed = True
                break  # restart with new active set
    
    # Final verification
    if not verify_decomp(U, V, W, n, m, p, rank):
        # Fallback
        U, V, W, rank = U0, V0, W0, rank0
    
    # Convert to lists for the interface
    U_list = [[U[i, r] for r in range(rank)] for i in range(n * m)]
    V_list = [[V[i, r] for r in range(rank)] for i in range(m * p)]
    W_list = [[W[i, r] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
