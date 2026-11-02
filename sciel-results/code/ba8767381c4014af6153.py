"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    def try_rank31_via_linear_dependency():
        """
        Treat the 2x2x9 tensor as a 4x18 linear map.
        The tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication.
        We view this as a linear map from R^{18} (the V-index space: j*p+k) 
        to R^{4} (the U-index space: i*m+j) for each W-index (k*n+i).
        
        Actually, let's think of it differently. The multiplication tensor has
        entries T_{(i,j),(j,k),(k,i)} = 1 for valid indices.
        
        We can view the tensor as a matrix of shape (n*m) x (m*p) for each
        W-index, but that's not quite right either.
        
        Better approach: The tensor is in R^{(n*m) x (m*p) x (p*n)}.
        A rank-1 decomposition term is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{p*n}.
        
        For the linear dependency approach: we have 32 rank-1 terms. We want to
        find if there's a linear combination of these 32 terms that equals zero
        (or equals the tensor), allowing us to drop one term.
        
        Actually, the key insight: if we can express one of the 32 rank-1 terms
        as a linear combination of the other 31, we can drop it.
        
        Let's set up the system. Each rank-1 term c gives us a vector in the
        space of tensors. The tensor space has dimension (n*m)*(m*p)*(p*n) = 4*18*18 = 1296.
        But our tensor is sparse - only 36 entries are non-zero (one for each (i,j,k) triple
        where the indices are valid for matrix multiplication).
        
        Wait, for 2x2 times 2x9: i in {0,1}, j in {0,1}, k in {0,...,8}.
        The tensor T has T[(i,j),(j,k),(k,i)] = 1.
        So there are 2*2*9 = 36 non-zero entries.
        
        We can flatten the tensor to a vector of length 36 (only the non-zero positions).
        Each rank-1 term c contributes a vector of length 36:
        vec_c[(i,j,k)] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        We have 32 such vectors, and their sum should equal the target vector (all 1s in 36 positions).
        
        To find a rank-31 decomposition, we need to find coefficients a_0, ..., a_31
        such that sum_c a_c * vec_c = target, with at least one a_c = 0.
        
        This is a linear system: 36 equations, 32 unknowns. Generally overdetermined.
        But if we allow one coefficient to be zero, we have 31 unknowns and 36 equations.
        
        Alternatively: we want to find if any one of the 32 vectors is in the span
        of the other 31. If so, we can drop that one.
        
        Let's check: is vec_31 (the last trivial term for k=8) in the span of vec_0..vec_30?
        Or more generally, for each c in 0..31, is vec_c in span(vec_0..vec_{c-1}, vec_{c+1}..vec_31)?
        """
        rank32 = build_rank32()
        U32 = np.array(rank32['U'], dtype=float)
        V32 = np.array(rank32['V'], dtype=float)
        W32 = np.array(rank32['W'], dtype=float)
        rank = 32
        
        # Build the 36-dimensional target vector and the 32 term vectors
        # Index mapping for the 36 non-zero tensor entries
        def tensor_index(i, j, k):
            """Map (i,j,k) to 0..35"""
            return i * m * p + j * p + k
        
        target = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[tensor_index(i, j, k)] = 1.0
        
        # Build the 32 term vectors (each of length 36)
        terms = np.zeros((rank, n * m * p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms[c, tensor_index(i, j, k)] = U32[i * m + j, c] * V32[j * p + k, c] * W32[k * n + i, c]
        
        # Check: sum of all terms should equal target
        assert np.allclose(terms.sum(axis=0), target, atol=1e-9), "Rank-32 terms don't sum to target"
        
        # For each c, check if terms[c] is in the span of the other 31 terms
        for c_drop in range(rank):
            if time.time() > deadline - 5:
                break
            
            # Build the matrix of the other 31 terms
            mask = [i for i in range(rank) if i != c_drop]
            A = terms[mask, :]  # shape (31, 36)
            b = terms[c_drop, :]  # shape (36,)
            
            # Solve A^T x = b (least squares), check if residual is small
            # Actually, we want: does there exist x such that A @ x = b?
            # A is 31x36, so A^T is 36x31. We solve A^T x = b.
            # This is overdetermined (36 eqs, 31 unknowns).
            
            # Use least squares
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A.T, b, rcond=None)
                residual = np.linalg.norm(A.T @ x - b)
                
                if residual < 1e-8:
                    # Found a dependency! We can drop term c_drop.
                    # Now build the rank-31 decomposition
                    U31 = U32[:, mask]
                    V31 = V32[:, mask]
                    W31 = W32[:, mask]
                    
                    # But we need to adjust: the sum of the remaining 31 terms
                    # should equal target + terms[c_drop] (since we're dropping it,
                    # but it was contributing to the sum)
                    # Actually no: sum_{c != c_drop} terms[c] = target - terms[c_drop]
                    # But we need sum = target. So we need to adjust.
                    
                    # Wait, I made an error. If terms[c_drop] is in the span of the others,
                    # that means terms[c_drop] = A^T x for some x.
                    # So sum_{c != c_drop} terms[c] * x_c + terms[c_drop] = target
                    # But we want sum_{c != c_drop} terms[c] * y_c = target
                    # So y_c = x_c + (coefficient of terms[c_drop])
                    # Hmm, this is getting complicated.
                    
                    # Actually, the correct approach: we have 32 terms that sum to target.
                    # If term c_drop = sum_{c != c_drop} alpha_c * term_c,
                    # then target = sum_{c != c_drop} term_c + sum_{c != c_drop} alpha_c * term_c
                    # = sum_{c != c_drop} (1 + alpha_c) * term_c
                    # So we can scale each remaining term by (1 + alpha_c).
                    
                    # But this changes the rank-1 structure! Each term is still rank-1,
                    # but scaled. That's fine - scaling a rank-1 tensor gives another rank-1 tensor.
                    
                    # However, the scaling might introduce non-integer values.
                    # Let's check if the alpha values are nice.
                    
                    # Get the coefficients
                    alpha = x  # This is the solution to A^T x = terms[c_drop]
                    
                    # New coefficients for the remaining terms
                    new_coeffs = np.ones(len(mask)) + alpha
                    
                    # Build the new U, V, W with scaled terms
                    U_new = U32[:, mask].copy()
                    V_new = V32[:, mask].copy()
                    W_new = W32[:, mask].copy()
                    
                    # Scale each column by its coefficient
                    for idx, coeff in enumerate(new_coeffs):
                        U_new[:, idx] *= coeff
                        V_new[:, idx] *= coeff
                        W_new[:, idx] *= coeff
                    
                    # Verify
                    U_list = U_new.tolist()
                    V_list = V_new.tolist()
                    W_list = W_new.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
            except np.linalg.LinAlgError:
                continue
        
        return None

    # First, try the linear dependency approach for rank 31
    result31 = try_rank31_via_linear_dependency()
    if result31 is not None and verify(result31["U"], result31["V"], result31["W"]):
        return result31
    
    # Fallback to rank 32
    result = build_rank32()
    if verify(result["U"], result["V"], result["W"]):
        return result

    return schoolbook()
# EVOLVE-BLOCK-END
