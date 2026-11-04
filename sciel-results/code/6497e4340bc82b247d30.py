"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = n * m * p - 1 = 35
    rank = n * m * p - 1
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # We will construct a rank-35 decomposition by starting with the 36 schoolbook
    # terms and combining two of them into one linear combination? No, tensor rank
    # doesn't work that way directly.
    # However, we can use the fact that for small dimensions, sometimes rank is
    # n*m*p - 1 or lower.
    # Let's check if 35 is valid.
    # The tensor is T[i,j,k] = delta_{i, i'} delta_{j, j'} delta_{k, k'}
    # Multiplication: C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{ijk} corresponds to coefficient of A_{ij} B_{jk} in C_{ik}.
    # T_{ijk} = 1 if the indices match the multiplication structure, else 0.
    # Specifically, the tensor is M_{(i,j), k}^{(i,k)}? No.
    # The tensor is T_{i, j, k} where i is row of A, j is col of A (row of B), k is col of B.
    # Output is C_{i, k}.
    # So T_{i, j, k} is the tensor for multiplication.
    # T_{i, j, k} = 1 if we are computing C_{i, k} from A_{i, j} B_{j, k}.
    # Wait, the tensor entries are just 1s for the "identity" mapping?
    # No, the multiplication tensor T has T_{i, j, k} = 1 if the indices form a valid product term?
    # Actually, the standard definition:
    # T_{i, j, k} is the coefficient of x_i y_j z_k in the polynomial?
    # For matrix multiplication, T = sum_{i,j,k} e_i (A) e_j (B) e_k (C) * delta?
    # The tensor is:
    # T_{i, j, k} = 1 if (i, j, k) corresponds to a term in the product?
    # No, the multiplication map is linear in A and B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is defined such that <T, A \otimes B> = C?
    # T_{i, j, k} = 1 for all i, j, k? No.
    # T_{i, j, k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{i, j, k} = 1 if we look at component (i, k) of C and component (j, k) of B and (i, j) of A?
    # Yes. So the tensor has 1s at all (i, j, k) positions?
    # Let's check:
    # U_{ij}^r, V_{jk}^r, W_{ki}^r.
    # Product: sum_r U_{ij}^r V_{jk}^r W_{ki}^r = delta_{ik}? No.
    # The condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, the output index is (i, k). The formula says W[k*n+i, r].
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This sum must equal 1 for all i, j, k?
    # No, the multiplication is C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So the tensor T_{i, j, k} must satisfy:
    # T_{i, j, k} = 1?
    # Let's check the schoolbook:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum over col (which is i, j, k):
    # Only when col corresponds to (i, j, k) do all three match.
    # So the sum is 1 for the specific (i, j, k) tuple?
    # Yes.
    # So the tensor T is the identity tensor?
    # T_{i, j, k} = 1 for all i, j, k?
    # If T_{i, j, k} = 1, then the rank is at least max(n*m, m*p, p*n) = 12.
    # But schoolbook has rank 36.
    # Why? Because the schoolbook decomposition uses distinct terms for each (i, j, k).
    # Can we do better?
    # Yes, if the tensor has structure.
    # But the multiplication tensor T_{i, j, k} = 1 for all i, j, k?
    # No!
    # The multiplication tensor is NOT the all-ones tensor.
    # The multiplication tensor T is defined by:
    # T_{i, j, k} = 1 if the indices match?
    # Let's look at the definition in the prompt:
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T_{i, j, k} = 1 if (i, j, k) is a valid triple?
    # No, T_{i, j, k} is 1 for all i, j, k?
    # If T is all-ones, then rank(T) = max(n, m, p) = 4?
    # No, rank of all-ones tensor of size n x m x p is max(n, m, p).
    # But matrix multiplication is not the all-ones tensor.
    # The matrix multiplication tensor T is:
    # T_{i, j, k} = 1 if we are looking at C_{i, k} from A_{i, j} and B_{j, k}.
    # So T_{i, j, k} = 1 for ALL i, j, k?
    # Yes, because for every i, j, k, the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So the tensor IS the all-ones tensor of size n x m x p?
    # If so, the rank is max(n, m, p) = 4.
    # But that's not right. Matrix multiplication rank is much higher.
    # Why?
    # Because the decomposition must work for ALL matrices A and B simultaneously.
    # The condition is:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # This must be represented as:
    # C_{i, k} = sum_r W_{k, i}^r ( sum_j U_{i, j}^r A_{i, j} ? No. )
    # The standard decomposition is:
    # C = sum_r (U^r A) (V^r B)^T ? No.
    # The standard tensor decomposition is:
    # T = sum_r u_r \otimes v_r \otimes w_r
    # where T_{i, j, k} = sum_r u_{i, j}^r v_{j, k}^r w_{k, i}^r ?
    # No, the indices are:
    # U has shape (n*m, rank). Index i*m+j.
    # V has shape (m*p, rank). Index j*p+k.
    # W has shape (p*n, rank). Index k*n+i.
    # The condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    # For matrix multiplication, T[i, j, k] = 1 for all i, j, k?
    # Let's test with 2x2x2.
    # If T is all-ones, rank is 2.
    # But 2x2x2 matrix multiplication has rank 7 (Strassen).
    # So T is NOT all-ones.
    # What is T?
    # The tensor T represents the bilinear map (A, B) -> C.
    # The components of T are T_{i, j, k} where i is row of A, j is col of A/row of B, k is col of B.
    # The output C has indices i (row) and k (col).
    # The term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So T_{i, j, k} = 1 if this term exists.
    # It exists for all i, j, k.
    # So why is the rank not 4?
    # Because the decomposition is not just T_{i, j, k} = sum_r ...
    # The decomposition is:
    # C_{i, k} = sum_r W_{k, i}^r * (U^r A)_i * (V^r B)_k ?
    # No, the standard form is:
    # C = sum_r (u_r A) \otimes (B v_r) ?
    # Let's stick to the index notation.
    # We need sum_r U_{ij}^r V_{jk}^r W_{ki}^r = 1 for all i, j, k?
    # If this is the condition, then the tensor is all-ones.
    # But Strassen's algorithm works.
    # In Strassen's algorithm, the coefficients are not all 1s.
    # The tensor T is NOT all-ones.
    # The tensor T is the structure tensor of the multiplication.
    # T_{i, j, k} = 1 if the indices match?
    # Actually, the tensor is:
    # T = sum_{i, j, k} e_i \otimes e_j \otimes e_k
    # This is the all-ones tensor.
    # Is the rank of the all-ones tensor of size n x m x p equal to the matrix multiplication rank?
    # NO.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The matrix multiplication tensor is:
    # T_{i, j, k} = 1 if i, j, k are valid?
    # Let's check the definition of the multiplication tensor.
    # The multiplication tensor T satisfies:
    # <T, A \otimes B> = C
    # where C = A B.
    # <T, A \otimes B> = sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k}
    # We want this to equal C_{i, k}? No, C is a matrix, so the result must be a matrix.
    # The inner product is a scalar? No, A and B are matrices, so A \otimes B is a vector in the tensor space.
    # The result C is a matrix, which is also a vector in its own space.
    # So the map is: (A, B) -> C.
    # This is a bilinear map.
    # The tensor T defines this map.
    # T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{?, ?}.
    # The output index is (i, k).
    # So T_{i, j, k}^{(i, k)} = 1.
    # This means T is a 4th order tensor? No, the map is bilinear, so it is a 3rd order tensor if we fix the output index?
    # No, the tensor is 3rd order: T_{input, input, output}.
    # Input 1: A_{i, j} (index ij).
    # Input 2: B_{j, k} (index jk).
    # Output: C_{i, k} (index ik).
    # So the tensor is T_{ij, jk, ik}?
    # No, the tensor is T_{(i, j), (j, k), (i, k)}?
    # The dimensions are (n*m) x (m*p) x (p*n).
    # So T is a tensor of size (n*m) x (m*p) x (p*n).
    # T_{(i, j), (j, k), (i, k)} = 1.
    # Note that the index j appears in both inputs.
    # This is the key! The tensor is NOT all-ones.
    # T_{(i, j), (j', k), (i', k')} = 1 if i=i', j=j', k=k'.
    # Wait, the output index is (i, k).
    # So T_{(i, j), (j, k), (i, k)} = 1.
    # For any other combination, it is 0.
    # So the tensor has 1s only when the indices are consistent.
    # Specifically, for a fixed (i, j, k), the entry T_{i*m+j, j*p+k, k*n+i} is 1.
    # All other entries are 0.
    # This is a sparse tensor.
    # It has n*m*p ones.
    # The rank of this tensor is what we want to minimize.
    # For 2x2x2, the tensor has 8 ones.
    # The rank is 7 (Strassen).
    # So we can use Strassen-like decompositions for larger dimensions.
    # For 3x3x3, the rank is 23.
    # For 3x3x4, we can try to find a decomposition.
    # Since I don't have the coefficients, I will use a heuristic:
    # The rank is likely n*m*p - something.
    # For 3x3x4, n*m*p = 36.
    # Let's try rank 35.
    # If I can find a rank 35 decomposition, it would be an improvement.
    # How?
    # We can combine two terms if they are linearly dependent in some sense?
    # Or use a known decomposition.
    # I will output a rank 36 decomposition but with a different structure?
    # No, the score is rank.
    # I will output the schoolbook rank 36.
    # Wait, the prompt says "Find a rank-minimal exact tensor decomposition".
    # And "score is the rank".
    # If I output rank 36, score is 36.
    # If I output rank 35, score is 35.
    # I need to find a valid rank 35 decomposition.
    # I don't have one.
    # I will output rank 36.

    # Re-generate schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
