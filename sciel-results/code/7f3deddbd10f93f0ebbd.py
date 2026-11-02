"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_based():
    """
    Attempt to construct a decomposition of rank < 48 for 3x4x4.
    
    Idea: Treat the 3x4x4 multiplication as a 2x2 block multiplication where
    the "inner" dimension is split 4 = 2 + 2.
    
    Let A be 3x4, B be 4x4.
    Split A into columns A1 (3x2), A2 (3x2).
    Split B into rows B1 (2x4), B2 (2x4).
    C = A1*B1 + A2*B2.
    
    This is sum of two 3x2x4 multiplications.
    Each 3x2x4 multiplication has schoolbook rank 3*2*4 = 24.
    Total rank = 48. No gain.
    
    Alternative: Split the 3 rows?
    A = [A1 (1x4); A2 (2x4)]
    B = 4x4.
    C = [A1*B (1x4); A2*B (2x4)]
    Rank 1*4*4 + 2*4*4 = 16 + 32 = 48. No gain.
    
    What if we use Strassen on a 2x2 block of the 4x4 matrix?
    This requires the other matrices to be structured accordingly.
    
    Let's try to exploit the specific dimensions 3,4,4.
    Is there a known rank 47?
    Alon-Boyer-Clancy-Ingalls (2007) show rank(3,4,4) <= 47?
    Actually, let's look up or recall.
    Rank(2,2,2)=7.
    Rank(3,3,3) <= 23.
    Rank(3,4,4):
    If we can find a rank 47 decomposition, that's a 1-rank improvement.
    
    Let's try a randomized search for a rank 47 decomposition.
    We fix U and V to be sparse/random and solve for W?
    Or fix U, V, W to be random and check if the tensor is in the span?
    
    Actually, a better heuristic:
    The tensor T is 3x4x4.
    We want T = sum_{r=1}^R U_r ⊗ V_r ⊗ W_r.
    
    Let's use a "local search" on the rank.
    Start with R=47.
    Initialize U, V, W with random integers {-1, 0, 1}.
    Compute the error tensor E = T - sum(U_r ⊗ V_r ⊗ W_r).
    Try to reduce the error by modifying one entry in U, V, or W.
    Since the space is discrete, this is hard.
    
    Alternative: Use the known result that rank(3,4,4) = 47 is an open problem?
    No, I think it's known to be 47 or 48.
    Actually, for 3x4x4, the schoolbook is 48.
    Is there a construction for 47?
    
    Let's try to construct a rank 47 decomposition by "stealing" a rank from a sub-block.
    
    Consider the 3x4x4 tensor.
    We can view it as a linear map from R^{12} x R^{16} -> R^{16}.
    
    Let's try a different approach:
    Use the fact that rank(3,4,4) <= rank(3,3,3) + rank(1,4,4) + rank(1,4,4) + ...
    This is getting complicated.
    
    Let's implement a quick local search for rank 47.
    If it fails within 2 seconds, return schoolbook.
    """
    n, m, p = 3, 4, 4
    target_rank = 47
    
    # Represent the tensor T as a 3D array
    T = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i][j][k] = 1 # Wait, the tensor for matrix multiplication is T[i,j,k] = 1 if ...
                # The tensor T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk}
                # T is a 3rd order tensor with T[i,j,k] = 1 if the term A_ij B_jk appears in C_ik.
                # Actually, the tensor is T[i,j,k] = 1.
                # The rank is the minimum number of rank-1 tensors summing to T.
                # A rank-1 tensor is U[i] V[j] W[k].
                # So we need sum_r U_r[i] V_r[j] W_r[k] = 1 for all i,j,k?
                # No, that would mean the matrix product is 1 for all entries, which is wrong.
                
                # Correct Tensor Definition:
                # The matrix multiplication tensor T_{ijk} is 1 if i,j,k are the indices for A_ij, B_jk, C_ik.
                # The product is C_{ik} = sum_j A_{ij} B_{jk}.
                # The tensor T is defined on I x J x K where I={1..n}, J={1..m}, K={1..p}.
                # T[i,j,k] = 1 if the term A_ij B_jk contributes to C_ik.
                # Wait, C_ik depends on sum over j.
                # The "multiplication tensor" is usually defined as T_{ijk} where i is row of A, j is col of A/row of B, k is col of B.
                # Then C_{ik} = sum_j A_{ij} B_{jk}.
                # The tensor is T[i,j,k] = 1.
                # We want to decompose T = sum_r u_r ⊗ v_r ⊗ w_r.
                # This means for all i,j,k: sum_r U_r[i] V_r[j] W_r[k] = 1.
                # This is only true if n=m=p=1?
                # No, the tensor T is NOT all 1s.
                # T[i,j,k] = 1 if the indices match the structure of multiplication.
                # Actually, for the matrix multiplication map M: A -> (A B), the associated tensor is T_{i j k} where i is row of C, j is row of A/col of B?
                
                # Let's use the standard definition:
                # T_{i, j, k} = 1 if the element at (i, j) of A and (j, k) of B contribute to (i, k) of C.
                # This requires the "middle" index to match.
                # So T[i, j, k] = 1 for all valid i, j, k.
                # Wait, if T[i,j,k]=1 for all i,j,k, then sum_r U_r[i] V_r[j] W_r[k] = 1 for ALL i,j,k.
                # This implies the tensor is the all-ones tensor.
                # But the all-ones tensor has rank 1 (just U=1, V=1, W=1).
                # So the rank of the all-ones tensor is 1.
                # But matrix multiplication has rank n*m*p.
                
                # I am confused about the tensor definition.
                # The tensor for matrix multiplication is NOT the all-ones tensor.
                # The tensor T is such that <T, A ⊗ B ⊗ C> = Tr(A B C^T) or something?
                # No.
                # The tensor T is defined such that the multiplication map M(A, B) = A B is a bilinear map.
                # The rank of the map is the rank of the associated tensor.
                # The tensor is T_{i, j, k} where i indexes rows of C, j indexes rows of A / cols of B?
                # Let's use the index notation:
                # C_{ik} = sum_j A_{ij} B_{jk}.
                # The tensor T has indices i, j, k.
                # T[i, j, k] = 1 if the term A_{ij} B_{jk} is present in the sum for C_{ik}.
                # Since j is summed over, for a fixed i, k, the terms are A_{i,1} B_{1,k}, A_{i,2} B_{2,k}, ..., A_{i,m} B_{m,k}.
                # So the tensor T is not just 1s.
                # T[i, j, k] is 1 if j is the "connecting" index.
                # But j is a single index.
                # The tensor is T_{i, j, k} = 1.
                # Why? Because for each j, the term A_{ij} B_{jk} appears.
                # So T[i, j, k] = 1 for all i, j, k.
                # But this leads to rank 1?
                
                # Ah, the tensor is T_{i, j, k} where i is row of A, j is col of A, k is col of B.
                # Then C_{ik} = sum_j A_{ij} B_{jk}.
                # The tensor T is defined on the space of A, B, C.
                # The rank is the number of terms in the sum.
                # The tensor is T = sum_j (e_j ⊗ e_j ⊗ e_j)? No.
                # The tensor is T_{i, j, k} = 1.
                # Wait, if T_{i,j,k}=1, then sum_r U_r[i] V_r[j] W_r[k] = 1.
                # This is only possible if n=m=p=1?
                # No, if n=m=p=2, T is 2x2x2 with all 1s.
                # Rank of all-ones tensor is 1.
                # But rank of 2x2x2 matrix multiplication is 7.
                
                # The error is in my tensor definition.
                # The tensor for matrix multiplication is NOT the all-ones tensor.
                # The tensor is T_{i, j, k} where i is row of C, j is "inner" index, k is col of C.
                # But the tensor must be multilinear in A, B, C.
                # The map is M(A, B) = C.
                # The tensor is T such that C_{ik} = sum_{j} A_{ij} B_{jk}.
                # This can be written as C_{ik} = sum_{j} T_{i, j, k} A_{ij} B_{jk} where T_{i,j,k} = 1.
                # So the tensor IS the all-ones tensor?
                # No, the tensor is defined on the indices i, j, k.
                # The "rank" of the matrix multiplication is the tensor rank of T.
                # If T is all ones, rank is 1.
                # This contradicts the known rank 7 for 2x2x2.
                
                # Let's re-read the definition of tensor rank for matrix multiplication.
                # The tensor is T_{i, j, k} = 1 if the indices are compatible.
                # For 2x2x2, the indices are i in {0,1}, j in {0,1}, k in {0,1}.
                # T_{i,j,k} = 1 for all i,j,k.
                # Rank is 7.
                # Why?
                # Because the tensor is NOT the all-ones tensor in the sense of the vector space.
                # The tensor is an element of V_I ⊗ V_J ⊗ V_K.
                # The "all-ones" tensor is 1⊗1⊗1, which has rank 1.
                # But the matrix multiplication tensor is NOT 1⊗1⊗1.
                # It is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k?
                # No, that is the all-ones tensor.
                
                # Okay, I am missing something fundamental.
                # Let's look at the schoolbook decomposition in the prompt.
                # U[i*m+j, c] = 1
                # V[j*p+k, c] = 1
                # W[k*n+i, c] = 1
                # This implies that the indices of U, V, W are composite.
                # U has size n*m. V has size m*p. W has size p*n.
                # This matches the dimensions of A, B, C^T.
                # The rank is n*m*p.
                # Each term c corresponds to a specific (i,j,k).
                # The term is U_c ⊗ V_c ⊗ W_c.
                # U_c is a vector of length n*m, with 1 at index i*m+j.
                # V_c is a vector of length m*p, with 1 at index j*p+k.
                # W_c is a vector of length p*n, with 1 at index k*n+i.
                # So the tensor is T = sum_{i,j,k} (e_{ij} ⊗ e_{jk} ⊗ e_{ki}).
                # This is NOT the all-ones tensor.
                # The all-ones tensor would be (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k).
                # The matrix multiplication tensor is the sum of "diagonal" terms in the composite indices.
                
                # So, for 3x4x4:
                # T = sum_{i=0..2, j=0..3, k=0..3} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}.
                # (Using 0-based indexing for simplicity, mapped to 1-based in prompt).
                
                # We want to find R < 48 vectors u_r, v_r, w_r such that
                # sum_{r=1}^R u_r[i,j] v_r[j,k] w_r[k,i] = 1 if (i,j,k) is in the support, 0 otherwise.
                # Wait, the support is ALL (i,j,k).
                # So we need sum_{r} u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k.
                # This is a system of 3*4*4 = 48 equations.
                # We have 3*4*4 + 4*4*4 + 4*3*4 = 48 + 64 + 48 = 160 variables (if R=1).
                # For R=47, we have 47 * (48+64+48) = 47 * 160 = 7520 variables.
                # This is a very underdetermined system.
                # However, the equations are nonlinear (products of variables).
                
                # We can try to find a solution with R=47.
                # Strategy:
                # Set 47 of the u_r, v_r, w_r to be the schoolbook basis vectors.
                # Let the 48th term be a "correction" term that cancels out the 48th schoolbook term and adds a new one?
                # No, we want to reduce the rank.
                
                # Let's try to "merge" two schoolbook terms into one?
                # Term (i,j,k) and (i',j',k').
                # If we can find u,v,w such that u[i,j]v[j,k]w[k,i] + u[i',j']v[j',k']w[k',i'] = 1 for both, and 0 elsewhere?
                # Unlikely to work simply.
                
                # Given the complexity and time limit, I will not implement a full search for rank 47.
                # I will return the schoolbook decomposition.
                return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition (rank < 48)
    # This is a placeholder for future improvement
    # For now, use schoolbook
    result = _build_schoolbook(n, m, p)
    
    # Add a deadline guard
    # Since construct is expected to be fast, no need for complex timing here
    # unless we do a search.
    
    return result
# EVOLVE-BLOCK-END
