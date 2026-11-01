"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try grouped Strassen: pair up columns (0,1), (2,3), ..., (12,13) as 7 pairs,
    # plus column 14 handled separately with 2 products.
    # Each 2x2 block uses Strassen rank 7, last column uses rank 2.
    # Total rank = 7*7 + 2 = 51
    
    if time.time() - _start_time > 10:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    num_pairs = p // 2  # 7 pairs
    leftover = p % 2    # 1 leftover column
    target_rank = 7 * num_pairs + 2 * leftover
    
    U = [[0] * target_rank for _ in range(n * m)]  # 4 x 51
    V = [[0] * target_rank for _ in range(m * p)]  # 30 x 51
    W = [[0] * target_rank for _ in range(p * n)]  # 30 x 51
    
    # Strassen for 2x2 x 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # Strassen products:
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 - b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    # For each pair of columns (k0, k1), we treat B_pair = [[b[0,k0], b[0,k1]], [b[1,k0], b[1,k1]]]
    # Output: C_pair = [[c[0,k0], c[0,k1]], [c[1,k0], c[1,k1]]]
    
    def add_strassen_block(base_idx, k0, k1):
        """Add Strassen decomposition for columns k0, k1."""
        # P1 = (a00 + a11) * (b[0,k0] + b[1,k1])
        # Wait, I need to be careful with the B indexing.
        # B is m x p = 2 x 15. B[j, k] where j is row of B, k is column.
        # For the pair (k0, k1):
        # B_pair = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        # So b00 = B[0,k0], b01 = B[0,k1], b10 = B[1,k0], b11 = B[1,k1]
        
        # A is n x m = 2 x 2. A[i, j] where i is row, j is column.
        # a00 = A[0,0], a01 = A[0,1], a10 = A[1,0], a11 = A[1,1]
        
        # C[i, k] = sum_j A[i,j] * B[j,k]
        # c[0,k0] = A[0,0]*B[0,k0] + A[0,1]*B[1,k0]
        # c[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1]
        # c[1,k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0]
        # c[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1]
        
        # In Strassen notation for this 2x2x2:
        # "b00" in Strassen = B[0,k0] (row 0, col 0 of B_pair)
        # "b01" in Strassen = B[0,k1] (row 0, col 1 of B_pair)
        # "b10" in Strassen = B[1,k0] (row 1, col 0 of B_pair)
        # "b11" in Strassen = B[1,k1] (row 1, col 1 of B_pair)
        
        # P1 = (a00 + a11) * (b00 + b11) = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # P2 = (a10 + a11) * b00 = (A[1,0]+A[1,1]) * B[0,k0]
        # P3 = a00 * (b01 - b11) = A[0,0] * (B[0,k1]-B[1,k1])
        # P4 = a11 * (b10 - b00) = A[1,1] * (B[1,k0]-B[0,k0])
        # P5 = (a00 + a01) * b11 = (A[0,0]+A[0,1]) * B[1,k1]
        # P6 = (a10 - a00) * (b00 + b01) = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # P7 = (a01 - a11) * (b10 + b11) = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        
        # c00 = P1 + P4 - P5 + P7  -> C[0,k0]
        # c01 = P3 + P5             -> C[0,k1]
        # c10 = P2 + P4             -> C[1,k0]
        # c11 = P1 - P2 + P3 + P6  -> C[1,k1]
        
        # Now map to our tensor decomposition:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k)
        
        # For each product P_t, we need:
        # U values: coefficients on A entries (indexed by i*m+j)
        # V values: coefficients on B entries (indexed by j*p+k)
        # W values: coefficients on C entries (indexed by k*n+i)
        
        # P1: U: A[0,0]=1, A[1,1]=1; V: B[0,k0]=1, B[1,k1]=1; W: C[0,k0]=1, C[1,k1]=1
        # But W has one entry per (k,i) pair. We need W[k*n+i, c] for each output.
        # The product contributes to multiple C entries with different coefficients.
        
        # Actually, in the tensor decomposition, each rank-1 term is:
        # sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This must equal 1 for each (i,j,k) triple (since the tensor is the identity for matrix mult).
        
        # Wait, the tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B 
        # contribute to C[i,k]. Actually T[i,j,k] = 1 for all valid (i,j,k) since
        # C[i,k] = sum_j A[i,j]*B[j,k].
        
        # So for each rank-1 component c, we have:
        # U[i*m+j, c] is the coefficient for A[i,j]
        # V[j*p+k, c] is the coefficient for B[j,k]
        # W[k*n+i, c] is the coefficient for C[i,k]
        
        # And sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each (i,j,k).
        
        # For P1 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1]):
        # This product appears in:
        # C[0,k0] with coeff +1
        # C[1,k1] with coeff +1
        # 
        # So W[0*n+0, c] = 1 (for k=k0, i=0) and W[1*n+1, c] = 1 (for k=k1, i=1)
        # But wait, W is indexed by [k*n+i]. So W[k0*2+0, c] = 1 and W[k1*2+1, c] = 1.
        
        # Hmm, but this single product P1 contributes to TWO different C entries.
        # In the tensor decomposition, each rank-1 term is a single product of:
        # (linear form in A) * (linear form in B) * (linear form in C)
        
        # So P1 as a rank-1 term would be:
        # (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1]) * (some C linear form)
        
        # But P1 contributes to both C[0,k0] and C[1,k1] with coefficient 1.
        # So the C linear form would be C[0,k0] + C[1,k1].
        
        # Let me re-derive. The tensor is:
        # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (where e_{i,j} is basis vector for A[i,j])
        
        # A rank-r decomposition is:
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
        
        # And T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
        
        # So for Strassen P1:
        # u_c: A[0,0]=1, A[1,1]=1, others 0
        # v_c: B[0,k0]=1, B[1,k1]=1, others 0
        # w_c: needs to satisfy that for each (i,j,k), the product contributes correctly.
        
        # The product (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1]) = 
        #   A[0,0]*B[0,k0] + A[0,0]*B[1,k1] + A[1,1]*B[0,k0] + A[1,1]*B[1,k1]
        
        # These four terms correspond to:
        # A[0,0]*B[0,k0] -> contributes to C[0,k0] (i=0,j=0,k=k0)
        # A[0,0]*B[1,k1] -> contributes to C[0,k1] (i=0,j=1,k=k1)
        # A[1,1]*B[0,k0] -> contributes to C[1,k0] (i=1,j=0,k=k0)
        # A[1,1]*B[1,k1] -> contributes to C[1,k1] (i=1,j=1,k=k1)
        
        # So w_c should have:
        # w_c[k0*2+0] = 1  (C[0,k0])
        # w_c[k1*2+0] = 1  (C[0,k1])
        # w_c[k0*2+1] = 1  (C[1,k0])
        # w_c[k1*2+1] = 1  (C[1,k1])
        
        # Let me verify: for (i=0,j=0,k=k0):
        # u[0*2+0]*v[0*15+k0]*w[k0*2+0] = 1*1*1 = 1 ✓
        # for (i=0,j=1,k=k1):
        # u[0*2+1]*v[1*15+k1]*w[k1*2+0] = 0*1*1 = 0
        # But we need A[0,0]*B[1,k1] to contribute to C[0,k1], which is (i=0,j=1,k=k1).
        # u[0*2+0] = 1, v[1*15+k1] = 1, w[k1*2+0] = 1
        # So contribution = 1*1*1 = 1 ✓
        
        # OK so this works! Each Strassen product becomes one rank-1 term.
        
        # Let me define the 7 Strassen products for columns k0, k1:
        # idx = base_idx + 0: P1
        # idx = base_idx + 1: P2
        # idx = base_idx + 2: P3
        # idx = base_idx + 3: P4
        # idx = base_idx + 4: P5
        # idx = base_idx + 5: P6
        # idx = base_idx + 6: P7
        
        # P1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: [1,0,0,1] for [A[0,0],A[0,1],A[1,0],A[1,1]]
        # V: [1,0,1,0] for [B[0,k0],B[0,k1],B[1,k0],B[1,k1]]
        # W: [1,1,1,1] for [C[0,k0],C[0,k1],C[1,k0],C[1,k1]]
        c = base_idx
        U[0][c] += 1; U[3][c] += 1  # A[0,0], A[1,1]
        V[0*p+k0][c] += 1; V[1*p+k1][c] += 1  # B[0,k0], B[1,k1]
        W[k0*n+0][c] += 1; W[k1*n+0][c] += 1; W[k0*n+1][c] += 1; W[k1*n+1][c] += 1
        
        # P2 = (A[1,0]+A[1,1]) * B[0,k0]
        # U: [0,0,1,1]
        # V: [1,0,0,0]
        # W: contributes to C[1,k0] with coeff +1
        c = base_idx + 1
        U[2][c] += 1; U[3][c] += 1  # A[1,0], A[1,1]
        V[0*p+k0][c] += 1  # B[0,k0]
        W[k0*n+1][c] += 1  # C[1,k0]
        
        # P3 = A[0,0] * (B[0,k1]-B[1,k1])
        # U: [1,0,0,0]
        # V: [0,1,0,-1]
        # W: contributes to C[0,k1] with coeff +1
        c = base_idx + 2
        U[0][c] += 1  # A[0,0]
        V[0*p+k1][c] += 1; V[1*p+k1][c] += -1  # B[0,k1]-B[1,k1]
        W[k1*n+0][c] += 1  # C[0,k1]
        
        # P4 = A[1,1] * (B[1,k0]-B[0,k0])
        # U: [0,0,0,1]
        # V: [0,0,-1,1]
        # W: contributes to C[0,k0] with +1 and C[1,k1] with +1
        # Wait: A[1,1]*B[1,k0] -> C[1,k0], A[1,1]*(-B[0,k0]) -> -C[1,k0]?
        # No: A[1,1]*B[1,k0] contributes to C[1,k0] (i=1,j=1,k=k0)
        # A[1,1]*(-B[0,k0]) contributes to C[1,k0] (i=1,j=0,k=k0) with coeff -1
        # So both contribute to C[1,k0]!
        # W: C[1,k0] gets +1 - 1 = 0? That can't be right.
        
        # Let me recheck Strassen:
        # P4 = a11 * (b10 - b00) = A[1,1] * (B[1,k0] - B[0,k0])
        # This contributes to:
        # A[1,1]*B[1,k0] -> C[1,k0]
        # A[1,1]*(-B[0,k0]) -> C[1,k0]  (since i=1, j=0, k=k0)
        # So the net contribution to C[1,k0] is A[1,1]*B[1,k0] - A[1,1]*B[0,k0]
        # But in the original matrix mult, C[1,k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0]
        # So P4 alone doesn't give the right answer; it's part of the combination.
        
        # In the tensor decomposition, W[k*n+i, c] is just a coefficient.
        # For P4: the product is A[1,1]*(B[1,k0]-B[0,k0])
        # This is a single scalar product. In the tensor, it contributes to
        # multiple (i,j,k) entries. But in our rank-1 decomposition, each term is
        # u ⊗ v ⊗ w where u, v, w are vectors.
        
        # The issue is that P4 = A[1,1]*(B[1,k0]-B[0,k0]) is NOT a product of a 
        # linear form in A and a linear form in B that maps to a single C entry.
        # It maps to C[1,k0] with the combined effect.
        
        # Actually, I think the way to handle this is:
        # The rank-1 term is: (A[1,1]) * (B[1,k0]-B[0,k0]) * (C[1,k0])
        # But that's not quite right either because the coefficient on C[1,k0] 
        # should reflect both contributions.
        
        # Let me reconsider. In the tensor T[i,j,k], T[i,j,k] = 1 for all (i,j,k).
        # A rank-1 term u⊗v⊗w contributes u[i*m+j]*v[j*p+k]*w[k*n+i] to T[i,j,k].
        
        # For P4 = A[1,1]*(B[1,k0]-B[0,k0]):
        # u = e_{A[1,1]} (only A[1,1] has coeff 1)
        # v = e_{B[1,k0]} - e_{B[0,k0]} (B[1,k0] has coeff 1, B[0,k0] has coeff -1)
        # w = ? 
        # 
        # The contribution to T[i,j,k] is:
        # u[i*m+j]*v[j*p+k]*w[k*n+i]
        # Non-zero when i=1,j=1 (u non-zero) AND (j=1,k=k0 OR j=0,k=k0) (v non-zero)
        # Case 1: i=1,j=1,k=k0: u=1, v=1, w[k0*2+1] -> contributes w[k0*2+1]
        # Case 2: i=1,j=0,k=k0: u=0 (since u only non-zero for j=1) -> no contribution
        # 
        # Wait, u[i*m+j] is non-zero only when i=1,j=1. So only case 1 applies.
        # So P4 contributes w[k0*2+1] to T[1,1,k0].
        # We need T[1,1,k0] = 1, and this is one of several terms contributing to it.
        
        # I think the correct approach is: each Strassen product corresponds to ONE rank-1 term,
        # and the W vector has coefficients that make the overall sum work out.
        
        # Let me just directly construct the rank-1 terms from the Strassen formulas.
        
        # Actually, I realize I should think of it differently. The Strassen algorithm says:
        # C[0,k0] = P1 + P4 - P5 + P7
        # C[0,k1] = P3 + P5
        # C[1,k0] = P2 + P4
        # C[1,k1] = P1 - P2 + P3 + P6
        #
        # Each P_t is a product of a linear form in A and a linear form in B.
        # So P_t = (sum_i alpha_{t,i} A[i]) * (sum_k beta_{t,k} B[k])
        #
        # In the tensor decomposition, each P_t corresponds to a rank-1 term where:
        # u_t[i*m+j] = alpha_{t,i} if j matches, else 0  -- NO, this isn't right.
        
        # I think the cleanest way is to just verify the final result numerically.
        # Let me construct it and check.
        
        pass  # Placeholder - I'll fill in after verification
        
    # This approach is getting complex. Let me just use the known-good 7-pair Strassen
    # with the last column as 2 products, and verify.
    
    # Reset and try a cleaner implementation
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    for pair_idx in range(num_pairs):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        base = 7 * pair_idx
        
        # P1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        c = base
        U[0][c] = 1; U[3][c] = 1
        V[0*p+k0][c] = 1; V[1*p+k1][c] = 1
        W[k0*n+0][c] = 1; W[k1*n+0][c] = 1; W[k0*n+1][c] = 1; W[k1*n+1][c] = 1
        
        # P2 = (A[1,0]+A[1,1]) * B[0,k0]
        c = base + 1
        U[2][c] = 1; U[3][c] = 1
        V[0*p+k0][c] = 1
        W[k0*n+1][c] = 1
        
        # P3 = A[0,0] * (B[0,k1]-B[1,k1])
        c = base + 2
        U[0][c] = 1
        V[0*p+k1][c] = 1; V[1*p+k1][c] = -1
        W[k1*n+0][c] = 1
        
        # P4 = A[1,1] * (B[1,k0]-B[0,k0])
        c = base + 3
        U[3][c] = 1
        V[1*p+k0][c] = 1; V[0*p+k0][c] = -1
        W[k0*n+1][c] = 1
        
        # P5 = (A[0,0]+A[0,1]) * B[1,k1]
        c = base + 4
        U[0][c] = 1; U[1][c] = 1
        V[1*p+k1][c] = 1
        W[k1*n+0][c] = -1; W[k1*n+1][c] = -1
        
        # P6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        c = base + 5
        U[2][c] = 1; U[0][c] = -1
        V[0*p+k0][c] = 1; V[0*p+k1][c] = 1
        W[k1*n+1][c] = 1
        
        # P7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        c = base + 6
        U[1][c] = 1; U[3][c] = -1
        V[1*p+k0][c] = 1; V[1*p+k1][c] = 1
        W[k0*n+0][c] = 1; W[k1*n+1][c] = 1
    
    # Leftover column 14: use 2 products
    if leftover == 1:
        k = 14
        base = 7 * num_pairs
        # P_a = A[0,0] * B[0,k]
        c = base
        U[0][c] = 1
        V[0*p+k][c] = 1
        W[k*n+0][c] = 1
        
        # P_b = A[0,1] * B[1,k]
        c = base + 1
        U[1][c] = 1
        V[1*p+k][c] = 1
        W[k*n+0][c] = 1
        
        # P_c = A[1,0] * B[0,k]
        # P_d = A[1,1] * B[1,k]
        # Wait, I need 4 products for the last column, not 2!
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # That's 4 scalar multiplications.
    
    # I made an error. The last column needs 4 products, not 2.
    # Let me recalculate: 7*7 + 4 = 53.
    
    # Let me restart with the correct rank.
    target_rank = 7 * num_pairs + 4 * leftover
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    for pair_idx in range(num_pairs):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        base = 7 * pair_idx
        
        c = base
        U[0][c] = 1; U[3][c] = 1
        V[0*p+k0][c] = 1; V[1*p+k1][c] = 1
        W[k0*n+0][c] = 1; W[k1*n+0][c] = 1; W[k0*n+1][c] = 1; W[k1*n+1][c] = 1
        
        c = base + 1
        U[2][c] = 1; U[3][c] = 1
        V[0*p+k0][c] = 1
        W[k0*n+1][c] = 1
        
        c = base + 2
        U[0][c] = 1
        V[0*p+k1][c] = 1; V[1*p+k1][c] = -1
        W[k1*n+0][c] = 1
        
        c = base + 3
        U[3][c] = 1
        V[1*p+k0][c] = 1; V[0*p+k0][c] = -1
        W[k0*n+1][c] = 1
        
        c = base + 4
        U[0][c] = 1; U[1][c] = 1
        V[1*p+k1][c] = 1
        W[k1*n+0][c] = -1; W[k1*n+1][c] = -1
        
        c = base + 5
        U[2][c] = 1; U[0][c] = -1
        V[0*p+k0][c] = 1; V[0*p+k1][c] = 1
        W[k1*n+1][c] = 1
        
        c = base + 6
        U[1][c] = 1; U[3][c] = -1
        V[1*p+k0][c] = 1; V[1*p+k1][c] = 1
        W[k0*n+0][c] = 1; W[k1*n+1][c] = 1
    
    if leftover == 1:
        k = 14
        base = 7 * num_pairs
        c = base
        U[0][c] = 1; V[0*p+k][c] = 1; W[k*n+0][c] = 1
        c = base + 1
        U[1][c] = 1; V[1*p+k][c] = 1; W[k*n+0][c] = 1
        c = base + 2
        U[2][c] = 1; V[0*p+k][c] = 1; W[k*n+1][c] = 1
        c = base + 3
        U[3][c] = 1; V[1*p+k][c] = 1; W[k*n+1][c] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
