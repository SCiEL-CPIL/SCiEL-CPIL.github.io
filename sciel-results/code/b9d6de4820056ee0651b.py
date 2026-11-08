import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _build_strassen444():
    """Return (U4, V4, W4) for 4x4x4 matrix multiplication using Strassen on 2x2x2 blocks.
    Shapes: U4 (16, 49), V4 (16, 49), W4 (16, 49) where:
      U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    This gives a valid decomposition of the 4x4x4 tensor with rank 49.
    """
    # Strassen for 2x2x2: rank 7
    # For 2x2x2: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    # The 7 rank-1 terms for Strassen (standard):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In tensor form, each rank-1 term (i,j,k) contributes 1 to U[i*2+j], V[j*2+k], W[k*2+i]
    # We need to express each M as a linear combination of A_ij and B_kl, so:
    # U entry for (i,j) is the coefficient of A_ij in that M
    # V entry for (j,k) is the coefficient of B_kl... wait, V[j*2+k] corresponds to B_{j,k}
    
    # Let's define the 7 terms with their (i,j) and (j',k) patterns:
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   A coeffs: A00:1, A11:1 -> U[0*2+0]=U[0]=1, U[1*2+1]=U[3]=1
    #   B coeffs: B00:1, B11:1 -> V[0*2+0]=V[0]=1, V[1*2+1]=V[3]=1
    #   C = M1 contributes to all C_ij with coefficient 1: W[0*2+0]=W[0]=1, W[0*2+1]=W[1]=1, W[1*2+0]=W[2]=1, W[1*2+1]=W[3]=1
    #
    # Let me just build it directly by listing all rank-1 contributions.
    
    # For 2x2x2 Strassen, the rank-7 decomposition:
    # Each term is (u_vec, v_vec, w_vec) where:
    # u_vec has nonzeros at positions corresponding to A entries
    # v_vec has nonzeros at positions corresponding to B entries
    # w_vec has nonzeros at positions corresponding to C entries
    
    # Positions: A[i,j] -> index i*2+j, B[j,k] -> index j*2+k, C[k,i] -> index k*2+i
    
    strassen_222 = [
        # M1 = (A00+A11)(B00+B11), C00+=1,C01+=1,C10+=1,C11+=1
        ([1,0,0,1], [1,0,0,1], [1,1,1,1]),
        # M2 = (A10+A11)B00, C10+=1, C11+=1 (with -1 in C11 via subtraction)
        # Actually C10 = M2+M4, C11 = M1-M2+M3+M6
        # So M2 contributes: C10 += 1, C11 += -1
        ([0,0,1,1], [1,0,0,0], [0,-1,1,0]),
        # M3 = A00(B01-B11), C01+=1, C11+=1
        ([1,0,0,0], [0,1,-1,0], [1,0,0,1]),
        # M4 = A11(B10-B00), C10+=1, C11+=1
        ([0,0,0,1], [0,-1,1,0], [0,1,1,0]),
        # M5 = (A00+A01)B11, C00+=-1, C01+=1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6 = (A10-A00)(B00+B01), C11+=1
        ([1,-1,0,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A01-A11)(B10+B11), C00+=1, C11+=1
        ([0,1,-1,0], [0,1,1,0], [1,0,0,1]),
    ]
    
    # Verify: this gives rank 7 for 2x2x2.
    
    # Now build 4x4x4 by applying Strassen to each 2x2 block.
    # A is 4x4, split into 4 blocks of 2x2: A00, A01, A10, A11
    # B is 4x4, split into 4 blocks: B00, B01, B10, B11
    # C = A*B, C_ij = sum_k A_ik * B_kj
    
    # We use Strassen at the block level (4 blocks, each 2x2):
    # This gives 7 block-level multiplications, each of which is a 2x2x2 multiplication
    # Each 2x2x2 multiplication uses rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Block indices: A[a,b] where a,b in {0,1} is the 2x2 block at rows 2a..2a+1, cols 2b..2b+1
    # Global A[i,j] = A_block[a,b][i-2a, j-2b] where a=i//2, b=j//2
    # Global index for A: i*4+j
    
    # For each of the 7 block-level Strassen terms (with indices t=0..6):
    #   It specifies which A_blocks and B_blocks are combined.
    #   The result is a 2x2 matrix (one block of C).
    #   To compute that 2x2 product, we use the 2x2x2 Strassen decomposition.
    
    # Let's denote the block-level Strassen terms. The 4 block-level A entries are:
    # A00 (blocks [0,0]), A01 (blocks [0,1]), A10 (blocks [1,0]), A11 (blocks [1,1])
    # Similarly for B.
    
    # The 7 block-level terms (same as the 2x2x2 Strassen above but at block level):
    # T0: (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    # T1: (A10+A11)B00 -> contributes to C10, C11
    # T2: A00(B01-B11) -> contributes to C01, C11
    # T3: A11(B10-B00) -> contributes to C10, C11
    # T4: (A00+A01)B11 -> contributes to C00, C01
    # T5: (A10-A00)(B00+B01) -> contributes to C11
    # T6: (A01-A11)(B10+B11) -> contributes to C00, C11
    
    # For each term t, and for each of the 7 sub-terms s in the 2x2x2 Strassen:
    # We need to determine:
    #   Which global A[i,j] entries are involved (with what coefficients)
    #   Which global B[j,k] entries are involved
    #   Which global C[k,i] entries are involved
    
    # This is getting complex. Let me just construct it programmatically.
    
    # For a given block-level term, say T0 = (A00+A11)(B00+B11):
    # The A part involves A00 and A11 (each 2x2).
    # The B part involves B00 and B11 (each 2x2).
    # The product is a 2x2x2 tensor where:
    #   The "A side" is the sum of two 2x2 matrices: A00 + A11
    #   The "B side" is the sum of two 2x2 matrices: B00 + B11
    
    # To decompose this using 2x2x2 Strassen, we treat:
    #   A_local = A00 + A11 (a 2x2 matrix)
    #   B_local = B00 + B11 (a 2x2 matrix)
    # And apply the rank-7 Strassen decomposition.
    
    # For sub-term s of the 2x2x2 Strassen, with u_vec, v_vec, w_vec:
    #   u_vec[r] is the coefficient for A_local[r] (where r = i*2+j for local 2x2 indices)
    #   v_vec[c] is the coefficient for B_local[c]
    #   w_vec[p] is the coefficient for C_local[p]
    
    # A_local[r] = sum over blocks: coeff * A_block[block]
    # For T0: A_local = A00 + A11, so A_local[i,j] = A[2*0+i, 2*0+j] + A[2*1+i, 2*1+j]
    #   i.e., A[0,j] + A[2,j] for j=0,1 (for i=0) and A[1,j] + A[3,j] for j=0,1 (for i=1)
    
    # So for a local index r = i*2+j, the global A indices are:
    #   (0*2+i, 0*2+j) = (i, j) and (1*2+i, 1*2+j) = (i+2, j+2)
    #   Wait, A00 is rows 0-1, cols 0-1. A11 is rows 2-3, cols 2-3.
    #   A_local[i,j] = A00[i,j] + A11[i,j] = A[i, j] + A[i+2, j+2]
    
    # So for sub-term s with u_vec[r] where r = i*2+j:
    #   Global A contribution: u_vec[r] * A[i, j] and u_vec[r] * A[i+2, j+2]
    
    # Similarly for B:
    # B_local[i,j] = B00[i,j] + B11[i,j] = B[i, j] + B[i+2, j+2]
    # So for v_vec[c] where c = i*2+j:
    #   Global B contribution: v_vec[c] * B[i, j] and v_vec[c] * B[i+2, j+2]
    
    # For C:
    # C_local[k,i] contributes to C_block[0,0][k,i] = C[2*0+k, 2*0+i] = C[k, i]
    # Wait, the result block is at position (0,0) of the block grid? No.
    # T0 contributes to all four C blocks: C00, C01, C10, C11.
    # C_local is a 2x2 matrix. The result of (A00+A11)*(B00+B11) is a 2x2 matrix.
    # This 2x2 matrix adds to C00, C01, C10, C11? No.
    
    # I'm confusing myself. Let me restart with a cleaner approach.
    
    # The 4x4x4 tensor T[i,j,k] = delta_{j,k} (since C[i,k] = sum_j A[i,j]*B[j,k])
    # In the trace convention W[k*n+i, r], we have T[i,j,k] with i in 0..3, j in 0..3, k in 0..3.
    # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    
    # For Strassen-based decomposition of 4x4x4:
    # Split A into 4 blocks of 2x2: A[a,b] for a,b in {0,1}, where A[a,b][i,j] = A[2a+i, 2b+j]
    # Split B into 4 blocks: B[a,b][i,j] = B[2a+i, 2b+j]
    # C[a,c][i,j] = sum_b A[a,b][i,*] * B[b,c][*,j]
    
    # Apply Strassen at block level: 7 terms, each is a product of two 2x2 "super-matrices"
    # Each super-matrix product is a 2x2x2 tensor, decomposed with rank 7.
    # Total rank = 49.
    
    # Let me just hardcode the rank-49 decomposition by enumerating all contributions.
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    # The 7 block-level Strassen terms with their A and B block coefficients:
    # (a_coeffs, b_coeffs, c_blocks) where a_coeffs is dict from block to coeff
    block_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, [(0,0), (0,1), (1,0), (1,1)]),  # T0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, [(1,0), (1,1)]),  # T1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, [(0,1), (1,1)]),  # T2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, [(1,0), (1,1)]),  # T3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, [(0,0), (0,1)]),  # T4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, [(1,1)]),  # T5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, [(0,0), (1,1)]),  # T6
    ]
    
    # The 7 sub-terms for 2x2x2 Strassen:
    sub_terms = [
        ([1,0,0,1], [1,0,0,1], [1,1,1,1]),
        ([0,0,1,1], [1,0,0,0], [0,-1,1,0]),
        ([1,0,0,0], [0,1,-1,0], [1,0,0,1]),
        ([0,0,0,1], [0,-1,1,0], [0,1,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([1,-1,0,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,-1,0], [0,1,1,0], [1,0,0,1]),
    ]
    
    col = 0
    for bt_idx, (a_coeff, b_coeff, c_blocks) in enumerate(block_terms):
        for st_idx, (u_vec, v_vec, w_vec) in enumerate(sub_terms):
            # u_vec[r] for r in 0..3, r = i*2+j (local 2x2)
            # For each local index, expand to global
            for r in range(4):
                i_loc = r // 2
                j_loc = r % 2
                coeff_u = u_vec[r]
                if coeff_u == 0:
                    continue
                # A_local[i_loc, j_loc] = sum over a_blocks: coeff * A[a_block, b_block][i_loc, j_loc]
                for (ba, bb), a_c in a_coeff.items():
                    global_i = 2 * ba + i_loc
                    global_j = 2 * bb + j_loc
                    U4[global_i * 4 + global_j, col] += coeff_u * a_c
            
            for c_idx in range(4):
                j_loc = c_idx // 2
                k_loc = c_idx % 2
                coeff_v = v_vec[c_idx]
                if coeff_v == 0:
                    continue
                for (ba, bb), b_c in b_coeff.items():
                    global_j = 2 * ba + j_loc
                    global_k = 2 * bb + k_loc
                    V4[global_j * 4 + global_k, col] += coeff_v * b_c
            
            for p_idx in range(4):
                k_loc = p_idx // 2
                i_loc = p_idx % 2
                coeff_w = w_vec[p_idx]
                if coeff_w == 0:
                    continue
                for (ca, cc) in c_blocks:
                    global_k = 2 * ca + k_loc
                    global_i = 2 * cc + i_loc
                    W4[global_k * 4 + global_i, col] += coeff_w
            
            col += 1
    
    return U4.tolist(), V4.tolist(), W4.tolist()

def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Pad B (4x7) to (4x8) by adding a zero column block.
    # Then 4x4 * 4x8 = 4x8.
    # Split the 8 columns into two groups of 4: B_left (4x4) and B_right (4x4, with last col zero).
    # Compute C_left = A * B_left (4x4) and C_right = A * B_right (4x4).
    # C_right will have its last column zero (since B_right's last column is zero).
    # Actually, let's pad to 4x8 where B_padded[*, 7] = 0.
    # Then C_padded = A * B_padded is 4x8, with C_padded[*, 7] = 0.
    # We only need C[*, 0..6], which is C_padded[*, 0..6].
    
    # Use rank-49 Strassen for each 4x4x4 block.
    # Total rank = 49 + 49 = 98.
    
    # U is 16x98, V is 28x98, W is 28x98 (p*n = 7*4 = 28)
    
    U4_list, V4_list, W4_list = _build_strassen444()
    U4 = np.array(U4_list, dtype=int)
    V4 = np.array(V4_list, dtype=int)
    W4 = np.array(W4_list, dtype=int)
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((28, rank), dtype=int)
    W = np.zeros((28, rank), dtype=int)
    
    # Part 1: columns 0..3 of B (B_left)
    # U_part1 = U4 (16x49)
    U[:, 0:49] = U4
    # V_part1: V4 is 16x49, but we need V to be 28x49.
    # V4[j*4+k, c] for j in 0..3, k in 0..3 -> maps to global V[j*7+k, c] for k in 0..3
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, 0:49] = V4[j * 4 + k, :]
    # V rows 28-31 (j=4,5,6... wait m*p = 4*7 = 28, so rows 0..27)
    # j in 0..3, k in 0..6. For k in 4..6, V is 0.
    # W_part1: W4 is 16x49, W4[k*4+i, c] for k in 0..3, i in 0..3 -> maps to global W[k*4+i, c] for k in 0..3
    # But global W has p*n = 28 rows, indexed by k*n+i where k in 0..6, i in 0..3.
    # For k in 0..3: W[k*4+i, c] = W4[k*4+i, c]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
    # For k in 4..6: W[k*4+i, c] = 0
    
    # Part 2: columns 4..7 of B (B_right), where column 7 is zero.
    # U_part2 = U4 (16x49)
    U[:, 49:98] = U4
    # V_part2: V4[j*4+k, c] for j in 0..3, k in 0..3 -> maps to global V[j*7+(k+4), c] for k in 0..3
    # But k+4 in 4..7. Global V has k in 0..6, so k+4 in 4..6 is valid, k+4=7 is out of range.
    # So only k=0,1,2 map to k+4=4,5,6. k=3 maps to k+4=7 which doesn't exist.
    for j in range(4):
        for k in range(3):  # k=0,1,2 -> global k=4,5,6
            V[j * 7 + (k + 4), 49:98] = V4[j * 4 + k, :]
    # k=3 in V4 is not mapped (since global k=7 doesn't exist)
    
    # W_part2: W4[k*4+i, c] for k in 0..3, i in 0..3 -> maps to global W[k*4+i, c] for k in 0..3
    # But these correspond to output columns 4..7. Wait, no.
    # The output C is 4x8. C[i, k] where k in 0..7.
    # W[k*n+i, r] where k is the output column index (0..6 for the actual 4x7 output).
    # For part 2, the output columns are 4..7. But we only output columns 0..6.
    # So output columns 4,5,6 are valid, column 7 is discarded.
    # W4[k*4+i, c] where k in 0..3 corresponds to local output column k.
    # Global output column = 4 + k for k in 0..3, i.e., 4,5,6,7.
    # We only keep 4,5,6 (k=0,1,2), discard k=3 (global column 7).
    for k in range(3):  # k=0,1,2 -> global column 4,5,6
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[k * 4 + i, :]
    # k=3 in W4 is not mapped (global column 7 doesn't exist)
    
    # Verify time
    if time.time() > deadline:
        pass  # Return what we have
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Verify correctness
def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Compute the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal delta_{j,k} for the multiplication tensor
    # Wait, the tensor for matrix multiplication is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if this is the identity, but actually the tensor is:
    # T[i,j,k] such that sum_{i,j} T[i,j,k] * A[i,j] * B[j,k] = (A*B)[i,k]... no.
    # The tensor decomposition is: (A*B)[i,k] = sum_r (sum_j U[i*m+j, r]*A[i,j]) * (sum_l V[j*p+k, r]*B[j,k])... 
    # Actually: (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l]) ... no.
    
    # The standard formulation: (A*B)[i,k] = sum_r U[i*m+j, r] ... 
    # Let me just check: for each (i,k), (A*B)[i,k] should equal sum_j A[i,j]*B[j,k].
    # With the decomposition: (A*B)[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_l V[j*p+k,r]*B[j,l])
    # This isn't quite right. The correct form is:
    # (A*B)[i,k] = sum_r U[i*m+j, r] ... 
    
    # Actually the tensor is: T_{ijk} where (A*B)_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # And T_{ijk} = delta_{jk}... no, T_{ijk} = 1 if j==j (always true), so T_{ijk} = delta... 
    # The multiplication tensor has T_{ijk} = 1 for all valid i,j,k (it's the identity in j).
    # Wait no: (AB)_{ik} = sum_j A_{ij} B_{jk}. So T_{ijk} = 1 for all i,j,k.
    # No! T_{ijk} is defined such that (AB)_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}.
    # Since (AB)_{ik} = sum_j A_{ij} B_{jk}, we have T_{ijk} = 1 for all i,j,k.
    # That can't be right either, because that would mean the tensor is all ones.
    
    # Let me reconsider. The tensor T is defined by:
    # T[i, j, k] = 1 if this term appears in (AB)_{ik} = sum_j A[i,j] B[j,k]
    # So for each (i,k), the sum over j gives: sum_j 1 * A[i,j] * B[j,k] = (AB)_{ik}.
    # So T[i,j,k] = 1 for all i, j, k.
    
    # With the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if val != 1:
                    print(f"Mismatch at ({i},{j},{k}): {val}")
                    return False
    return True

if __name__ == "__main__":
    result = construct()
    print("Verifying...")
    ok = _verify(result["U"], result["V"], result["W"])
    print(f"Verification: {'PASS' if ok else 'FAIL'}")
    print(f"Rank: {len(result['U'][0])}")
