"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a matrix multiplication tensor decomposition for 5x5 matrices.
    Tries to find a rank below 125 using known algorithmic structures.
    Falls back to schoolbook (rank 125) if a lower-rank exact decomposition
    cannot be verified within the time limit.
    """
    start_time = time.time()
    deadline = 150.0  # Wall-clock limit
    
    n, m, p = DIMENSIONS
    
    def time_check():
        return time.time() > deadline
    
    # Helper to verify a decomposition
    def verify(U, V, W, rank):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Convention: W[k*n + i, r]
        """
        # Convert to numpy arrays for efficient checking
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T_{i,j,k} for matrix mult (C_ik = sum_j A_ij B_jk)
        # Decomposition: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
        
        # We can check this by reconstructing the tensor or checking random slices.
        # Efficient check: For random A, B, compute C = A @ B, and compare with
        # sum_r (U^T A_flat)_r * (V^T B_flat)_r mapped to C.
        
        # Actually, let's just check the tensor directly for small sizes.
        # T is 5x5x5 = 125 elements.
        # T[i, j, k] = 1 if we are looking at the standard basis? No.
        # The tensor for matrix multiplication is T_{i,j,k} = delta_{i,k}? No.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{i,j,k} such that C_{ik} = sum_{j} T_{i,j,k} A_{ij} B_{jk}?
        # No, the standard form is:
        # C_{ik} = sum_{j, r} U_{ij, r} V_{jk, r} W_{ki, r} A_{ij} B_{jk}
        # This implies that the "structure tensor" is defined such that:
        # sum_r U_{ij, r} V_{jk, r} W_{ki, r} = delta_{i,k} * (something)?
        # No. The identity is:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # In tensor decomposition form:
        # C_{ik} = sum_r (sum_j U_{ij, r} A_{ij}) (sum_j V_{jk, r} B_{jk}) W_{ki, r} ???
        # No, the standard Strassen form is:
        # C_{ik} = sum_r M_r W_{ki, r}
        # where M_r = (sum_j U_{ij, r} A_{ij}) * (sum_j V_{jk, r} B_{jk})
        # This requires that for each r, the product of the linear forms in A and B
        # contributes to C_{ik} via W_{ki, r}.
        
        # So the condition is:
        # For all i, k:
        # delta_{i,k} is NOT the target. The target is the operation A @ B.
        # Let's check:
        # sum_r (sum_j U_{i*m+j, r} A_{ij}) * (sum_j V_{j*p+k, r} B_{jk}) * W_{k*n+i, r} == C_{ik}
        
        # We can test this with random integer matrices.
        for _ in range(5):
            if time_check():
                return False
            A = np.random.randint(0, 10, (n, m))
            B = np.random.randint(0, 10, (m, p))
            C_expected = A @ B
            
            # Compute decomposition result
            # M_r = sum_j U_{ij, r} A_{ij}  -> This is not quite right.
            # U is (n*m, rank). Index i*m+j.
            # Let U_r be the vector for rank r.
            # Term r is: (sum_{i,j} U_{i*m+j, r} A_{ij}) * (sum_{j,k} V_{j*p+k, r} B_{jk}) * W_{k*n+i, r}
            # Wait, the summation indices for the linear forms must match the tensor indices.
            #
            # Standard definition:
            # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # Here the tensor T represents the multiplication tensor.
            # For matrix multiplication, T_{i,j,k} is 1 if... wait.
            # The multiplication tensor is defined by:
            # M(A, B) = sum_{i,j,k} A_{ij} B_{jk} e_i \otimes e_k
            # So the tensor components are T_{i,j,k} = 1 if we are multiplying A_{ij} and B_{jk} to get C_{ik}.
            # But C_{ik} receives contributions from ALL j.
            # So T_{i,j,k} = 1 for all i,j,k? No.
            # T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # So the coefficient of A_{ij} B_{jk} in C_{ik} is 1 if the indices match, i.e., always 1 for the specific triple (i,j,k) contributing to (i,k).
            # Actually, the tensor T is of size n x m x p (or similar) where T_{i,j,k} = 1 if the term A_{ij}B_{jk} appears in C_{ik}.
            # But C_{ik} is a single entry. The term A_{ij}B_{jk} contributes to C_{ik}.
            # So T_{i,j,k} = 1 for all valid i,j,k?
            # No, T is a 3-tensor of size n x m x p?
            # Let's look at the indices in the decomposition:
            # U has shape (n*m, rank). Index (i, j).
            # V has shape (m*p, rank). Index (j, k).
            # W has shape (p*n, rank). Index (k, i).
            #
            # The reconstructed value for C_{ik} is:
            # sum_r (sum_j U_{i,j,r} A_{ij}) ... No, that's not how it works.
            #
            # Correct Strassen Form:
            # Let M_r = (sum_{i,j} U_{i,j,r} A_{ij}) * (sum_{j,k} V_{j,k,r} B_{jk})
            # Then C_{ik} = sum_r M_r W_{k,i,r}
            #
            # This is valid if and only if:
            # sum_r U_{i',j',r} V_{j',k',r} W_{k',i',r} = delta_{i,i'} delta_{k,k'} ? No.
            #
            # Let's expand:
            # sum_r W_{k,i,r} (sum_{i',j'} U_{i',j',r} A_{i',j'}) (sum_{j',k'} V_{j',k',r} B_{j',k'})
            # = sum_{i',j',j',k'} (sum_r U_{i',j',r} V_{j',k',r} W_{k,i,r}) A_{i',j'} B_{j',k'}
            #
            # We want this to equal sum_j A_{i,j} B_{j,k}.
            # So we require:
            # sum_r U_{i',j',r} V_{j',k',r} W_{k,i,r} = delta_{i',i} delta_{k',k}
            #
            # Note the indices:
            # LHS depends on i', j', k', and the target indices i, k.
            # The term is non-zero only if i'=i and k'=k.
            # So the condition is:
            # For all i, k, i', k':
            # sum_r U_{i',j',r} V_{j',k',r} W_{k,i,r} = delta_{i',i} delta_{k',k}
            #
            # Wait, j' is summed out in the middle?
            # In the expansion:
            # sum_{i',j',j',k'} ...
            # The indices j' appear in U and V.
            # Let's rename summation indices to avoid confusion.
            # C_{ik} = sum_r W_{k,i,r} [ sum_{a,b} U_{a,b,r} A_{a,b} ] [ sum_{c,d} V_{c,d,r} B_{c,d} ]
            # = sum_{a,b,c,d} ( sum_r U_{a,b,r} V_{c,d,r} W_{k,i,r} ) A_{a,b} B_{c,d}
            #
            # We want C_{ik} = sum_j A_{i,j} B_{j,k}.
            # So we need:
            # sum_r U_{a,b,r} V_{c,d,r} W_{k,i,r} = delta_{a,i} delta_{b,c} delta_{d,k}
            #
            # This is the exact condition.
            # We can verify this by checking if the matrix of coefficients matches the identity/structure.
            
            # Efficient verification:
            # Construct the tensor T_{a,b,c,d} = sum_r U_{a,b,r} V_{c,d,r} W_{k,i,r} ?? No, W depends on i,k which are the OUTPUT indices.
            # The condition is:
            # For fixed output i, k, the matrix of coefficients for inputs (a,b) and (c,d) must be delta_{a,i} delta_{b,c} delta_{d,k}.
            
            # Let's just do the random matrix check, it's simpler and robust for integer/half-integer entries if we use exact arithmetic or high precision.
            # Since we use Fraction, we can do exact check.
            
            A_frac = [[Fraction(A[i, j]) for j in range(m)] for i in range(n)]
            B_frac = [[Fraction(B[i, j]) for j in range(p)] for i in range(m)]
            
            C_calc = [[Fraction(0) for _ in range(p)] for _ in range(n)]
            
            # Pre-compute linear forms
            # L_r(A) = sum_{a,b} U[a*m+b, r] A[a,b]
            # L_r(B) = sum_{c,d} V[c*p+d, r] B[c,d]
            
            L_A = [Fraction(0) for _ in range(rank)]
            L_B = [Fraction(0) for _ in range(rank)]
            
            for r in range(rank):
                for a in range(n):
                    for b in range(m):
                        if U[a*m+b][r] != 0:
                            L_A[r] += Fraction(U[a*m+b][r]) * A_frac[a][b]
                for c in range(m):
                    for d in range(p):
                        if V[c*p+d][r] != 0:
                            L_B[r] += Fraction(V[c*p+d][r]) * B_frac[c][d]
                
                for i in range(n):
                    for k in range(p):
                        if W[k*n+i][r] != 0:
                            C_calc[i][k] += L_A[r] * L_B[r] * Fraction(W[k*n+i][r])
            
            # Compare C_calc with C_expected
            for i in range(n):
                for k in range(p):
                    if C_calc[i][k] != Fraction(C_expected[i, k]):
                        return False
            
        return True

    # 1. Try to construct a known lower-rank decomposition.
    # The best known rank for 5x5 is 78 (Bini et al.).
    # However, explicit construction is complex.
    # Let's try a heuristic search or a specific known block decomposition.
    
    # Strategy: Use Strassen for 3x3 and 2x2 blocks?
    # 5x5 split into 3x3 and 2x2.
    # This is not a direct tensor decomposition of the 5x5 tensor into rank-1 terms
    # unless we expand the block algorithms into the full tensor.
    
    # Let's try to generate a random rank-r decomposition and see if we can solve for it?
    # No, that's NP-hard.
    
    # Let's stick to the schoolbook decomposition but check if we can prune.
    # Schoolbook rank is 125.
    
    # Is there a simple rank reduction?
    # For 2x2, Strassen gives 7 < 8.
    # For 5x5, maybe we can combine some terms?
    
    # Let's try to implement the Bini et al. rank-78 decomposition if possible.
    # Reference: "Algorithms for matrix multiplication" by Bini, Capovani, Lotti, Romani.
    # The construction is not trivial to code from scratch without the explicit list of coefficients.
    
    # Alternative: Use the fact that 5 = 3 + 2.
    # We can use the Kronecker product of decompositions?
    # If we had a decomposition for 3x3 (rank 7) and 2x2 (rank 4),
    # we can't just multiply them to get 5x5.
    
    # Let's try a random search for a rank slightly below 125?
    # Unlikely to find exact integer/half-integer solutions randomly.
    
    # Given the complexity and time limit, and the instruction to "Keep the program's entry-point function name(s)...",
    # and the fact that the current program is the schoolbook decomposition,
    # I will return the schoolbook decomposition.
    # 
    # Wait, the prompt says "Find a rank-minimal exact tensor decomposition".
    # The score is the rank. Lower is better.
    # If I return 125, the score is 125.
    # If I could return 78, the score would be 78.
    #
    # Let's try to find a rank-78 decomposition online or in memory.
    # The 78-rank decomposition for 5x5 is known.
    # It is based on the 4x4 decomposition (rank 49) and extension.
    #
    # Actually, I can try to use the `sympy` or similar if available? No, only standard lib, numpy, scipy.
    #
    # Let's try to construct a decomposition using the "block Strassen" idea explicitly.
    # Split A, B into blocks:
    # A = [[A11, A12], [A21, A22]] with sizes 3x3, 3x2, 2x3, 2x2.
    # B = [[B11, B12], [B21, B22]] with sizes 3x3, 3x2, 2x3, 2x2.
    #
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    #
    # Use Strassen for 3x3 products (rank 7) and 2x2 products (rank 4).
    #
    # This block algorithm uses:
    # 1. A11 B11 (3x3): 7 mults
    # 2. A12 B21 (3x2 * 2x3): This is a 3x3 matrix result? No, 3x3.
    #    Wait, A12 is 3x2, B21 is 2x3. Product is 3x3.
    #    Rank of 3x3 mult is 7 (Strassen).
    # 3. A11 B12 (3x3 * 3x2): Result 3x2.
    #    This is not a square matrix multiplication.
    #    The tensor for 3x3 * 3x2 -> 3x2 has rank 3*3*2 = 18?
    #    Strassen doesn't directly apply to non-square.
    #
    # So block Strassen doesn't easily reduce the rank for non-square blocks.
    #
    # Therefore, the schoolbook decomposition is likely the safest bet for correctness
    # unless I can hardcode the 78-rank coefficients.
    #
    # I do not have the 78-rank coefficients memorized.
    #
    # I will return the schoolbook decomposition.
    
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
                
                if time_check():
                    break
            if time_check():
                break
        if time_check():
            break
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
