"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = 3, 5, 7

    def time_ok():
        return (time.time() - start_time) < deadline

    # Try to find a decomposition of rank 99 using Strassen blocks on 2x2x2 sub-cubes.
    # The idea: partition the n x m x p grid into 2x2x2 blocks where possible,
    # and use Strassen (rank 7) for each such block, schoolbook (rank 8) for the rest.
    
    # For 3x5x7:
    # We can partition into blocks along each dimension:
    # n=3: blocks of size 2 and 1 (i.e., rows 0-1 as block, row 2 separate)
    # m=5: blocks of size 2, 2, and 1 (cols 0-1, 2-3, 4)
    # p=7: blocks of size 2, 2, 2, and 1 (outputs 0-1, 2-3, 4-5, 6)
    
    # Actually, let's think about this more carefully.
    # The tensor T[i,j,k] = 1 if the (i,k) entry of AB depends on A[i,j] * B[j,k].
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    # So the full tensor is all-ones in the valid index range.
    
    # A rank-1 tensor in this context is u_i * v_j * w_k for all (i,j,k).
    # The schoolbook decomposition uses one rank-1 term for each (i,j,k) triple.
    
    # Strassen's algorithm for 2x2x2:
    # C = A*B where A is 2x2, B is 2x2, C is 2x2.
    # The 7 products are:
    # M1 = A00*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B10+B11)
    # M7 = (A00-A11)*(B00+B01)
    #
    # In terms of the tensor decomposition:
    # For a 2x2x2 block with local indices (i,j,k) in {0,1}^3,
    # each Strassen term contributes to specific (i,j,k) positions.
    
    # Let me define the Strassen terms for a 2x2x2 block:
    # Each term is a triple (U_local, V_local, W_local) where each is a vector in {0,1}^3
    # with entries in {-1, 0, 1}.
    
    strassen_terms = [
        # M1 = A00 * (B00 + B11) -> contributes to C00 and C11
        ({(0,0): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1}),
        # Wait, I need to be more careful. Let me use the standard formulation.
        
        # Actually, let me use a different approach. I'll directly construct
        # the Strassen decomposition for a 2x2x2 block.
    ]
    
    # Let me use a well-known Strassen decomposition for 2x2x2:
    # The 7 rank-1 terms for multiplying two 2x2 matrices:
    
    def make_strassen_terms():
        """Return list of (U, V, W) for each of the 7 Strassen terms.
        U is a dict from (i,j) to coefficient for A
        V is a dict from (j,k) to coefficient for B
        W is a dict from (k,i) to coefficient for C
        """
        # Using the standard Strassen algorithm:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a10)*b11
        # m6 = (a01-a10)*(b10+b11)
        # m7 = (a00-a11)*(b00+b01)
        
        # C00 = m1 + m4 - m5 - m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m7
        
        terms = []
        
        # m1 = (a00+a11)*(b00+b11)
        # U: a00 coeff 1, a11 coeff 1 -> {(0,0):1, (1,1):1}
        # V: b00 coeff 1, b11 coeff 1 -> {(0,0):1, (1,1):1}
        # W: C00 coeff 1, C11 coeff 1 -> {(0,0):1, (1,1):1}
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        
        # m2 = (a10+a11)*b00
        # U: a10 coeff 1, a11 coeff 1 -> {(1,0):1, (1,1):1}
        # V: b00 coeff 1 -> {(0,0):1}
        # W: C10 coeff 1, C11 coeff -1 -> {(0,1):1, (1,1):-1}
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        
        # m3 = a00*(b01-b11)
        # U: a00 coeff 1 -> {(0,0):1}
        # V: b01 coeff 1, b11 coeff -1 -> {(0,1):1, (1,1):-1}
        # W: C01 coeff 1, C11 coeff 1 -> {(1,0):1, (1,1):1}
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        
        # m4 = a11*(b10-b00)
        # U: a11 coeff 1 -> {(1,1):1}
        # V: b10 coeff 1, b00 coeff -1 -> {(1,0):1, (0,0):-1}
        # W: C00 coeff 1, C10 coeff 1 -> {(0,0):1, (0,1):1}
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        
        # m5 = (a00+a10)*b11
        # U: a00 coeff 1, a10 coeff 1 -> {(0,0):1, (1,0):1}
        # V: b11 coeff 1 -> {(1,1):1}
        # W: C00 coeff -1, C01 coeff 1 -> {(0,0):-1, (1,0):1}
        terms.append((
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        
        # m6 = (a01-a10)*(b10+b11)
        # U: a01 coeff 1, a10 coeff -1 -> {(0,1):1, (1,0):-1}
        # V: b10 coeff 1, b11 coeff 1 -> {(1,0):1, (1,1):1}
        # W: C01 coeff -1, C10 coeff 1 -> {(1,0):-1, (0,1):1}
        terms.append((
            {(0,1): 1, (1,0): -1},
            {(1,0): 1, (1,1): 1},
            {(1,0): -1, (0,1): 1}
        ))
        
        # m7 = (a00-a11)*(b00+b01)
        # U: a00 coeff 1, a11 coeff -1 -> {(0,0):1, (1,1):-1}
        # V: b00 coeff 1, b01 coeff 1 -> {(0,0):1, (0,1):1}
        # W: C00 coeff -1, C11 coeff 1 -> {(0,0):-1, (1,1):1}
        terms.append((
            {(0,0): 1, (1,1): -1},
            {(0,0): 1, (0,1): 1},
            {(0,0): -1, (1,1): 1}
        ))
        
        return terms
    
    strassen_2x2x2 = make_strassen_terms()
    
    # Verify Strassen decomposition correctness
    def verify_strassen():
        """Verify that the Strassen terms correctly decompose 2x2x2 matrix multiplication."""
        # For each (i,k) in {0,1}x{0,1}, the sum over j and terms should give 1
        # For each (i,j,k), the sum over terms of U[i,j]*V[j,k]*W[k,i] should be 1
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for (U_term, V_term, W_term) in strassen_2x2x2:
                        u_val = U_term.get((i,j), 0)
                        v_val = V_term.get((j,k), 0)
                        w_val = W_term.get((k,i), 0)
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False, (i,j,k), total
        return True, None, None
    
    ok, info, total = verify_strassen()
    if not ok:
        # Fall back to schoolbook if Strassen verification fails
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
    
    # Now build the full 3x5x7 decomposition using Strassen blocks.
    # Strategy: partition the 3x5x7 tensor into 2x2x2 blocks where possible.
    # 
    # i dimension (size 3): blocks at i=0,1 and i=2 (or i=0,1 and i=2)
    # j dimension (size 5): blocks at j=0,1; j=2,3; j=4
    # k dimension (size 7): blocks at k=0,1; k=2,3; k=4,5; k=6
    #
    # We can form 2x2x2 blocks for:
    # (i in {0,1}, j in {0,1}, k in {0,1})
    # (i in {0,1}, j in {0,1}, k in {2,3})
    # (i in {0,1}, j in {0,1}, k in {4,5})
    # (i in {0,1}, j in {2,3}, k in {0,1})
    # (i in {0,1}, j in {2,3}, k in {2,3})
    # (i in {0,1}, j in {2,3}, k in {4,5})
    # (i in {0,1}, j in {4},   k in {0,1})  -> can't form 2x2x2 since j only has 1
    # ...
    
    # Actually, let me think about this differently.
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k).
    # We want to decompose this into rank-1 tensors.
    #
    # A 2x2x2 sub-block with i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # has 8 entries, all equal to 1.
    # We can decompose this 2x2x2 all-ones tensor using Strassen with rank 7.
    #
    # The remaining entries (those not in any 2x2x2 block) must be handled
    # with schoolbook terms (rank 1 each).
    
    # Let me define the partition:
    # i blocks: [0,1], [2]
    # j blocks: [0,1], [2,3], [4]
    # k blocks: [0,1], [2,3], [4,5], [6]
    
    i_blocks = [(0,1), (2,)]
    j_blocks = [(0,1), (2,3), (4,)]
    k_blocks = [(0,1), (2,3), (4,5), (6,)]
    
    rank_terms = []  # list of (U_dict, V_dict, W_dict) where keys are global indices
    
    for i_block in i_blocks:
        for j_block in j_blocks:
            for k_block in k_blocks:
                # Check if this block is 2x2x2
                if len(i_block) == 2 and len(j_block) == 2 and len(k_block) == 2:
                    # Use Strassen
                    i0, i1 = i_block
                    j0, j1 = j_block
                    k0, k1 = k_block
                    
                    for (U_loc, V_loc, W_loc) in strassen_2x2x2:
                        U_global = {}
                        V_global = {}
                        W_global = {}
                        
                        for (li, lj), coeff in U_loc.items():
                            gi = i0 + li
                            gj = j0 + lj
                            U_global[(gi, gj)] = coeff
                        
                        for (lj, lk), coeff in V_loc.items():
                            gj = j0 + lj
                            gk = k0 + lk
                            V_global[(gj, gk)] = coeff
                        
                        for (lk, li), coeff in W_loc.items():
                            gk = k0 + lk
                            gi = i0 + li
                            W_global[(gk, gi)] = coeff
                        
                        rank_terms.append((U_global, V_global, W_global))
                else:
                    # Schoolbook for this block
                    for i in i_block:
                        for j in j_block:
                            for k in k_block:
                                U_global = {(i, j): 1}
                                V_global = {(j, k): 1}
                                W_global = {(k, i): 1}
                                rank_terms.append((U_global, V_global, W_global))
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_global, V_global, W_global) in enumerate(rank_terms):
        for (i, j), coeff in U_global.items():
            if coeff != 0:
                U[i * m + j][col] = coeff
        for (j, k), coeff in V_global.items():
            if coeff != 0:
                V[j * p + k][col] = coeff
        for (k, i), coeff in W_global.items():
            if coeff != 0:
                W[k * n + i][col] = coeff
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(len(U[i * m + j])):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False, (i, j, k), total
        return True, None, None
    
    if not time_ok():
        # Return what we have (should be valid)
        return {"U": U, "V": V, "W": W}
    
    ok, info, total = verify_decomposition(U, V, W, n, m, p)
    
    if not ok:
        # Fall back to schoolbook
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
