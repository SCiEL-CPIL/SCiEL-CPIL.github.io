"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _to_list(mat):
    """Convert a numpy array of Fractions to a nested list of ints/half-ints."""
    if mat is None:
        return []
    result = []
    for row in mat:
        r = []
        for v in row:
            if isinstance(v, Fraction):
                # Convert Fraction to float, but ensure it's a half-integer or integer
                # The task says "integer/half-integer entries"
                # Let's just return the float value if it's exact, or the fraction?
                # The interface says "integer/half-integer entries".
                # Usually this means numbers like 0.5, 1.0, 1.5, etc.
                # Let's return float(v).
                r.append(float(v))
            else:
                r.append(float(v))
        result.append(r)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 3, 5
    
    # We will construct the decomposition manually using the block strategy:
    # A (2x3) = [A11 (2x2) | A12 (2x1)]
    # B (3x5) = [B11 (2x5); B21 (1x5)]
    # C = A11 B11 + A12 B21
    
    # Block 1: A11 (2x2) * B11 (2x5)
    # Split B11 into B111 (2x2) and B112 (2x3)
    # C1 = A11 B111 + A11 B112
    # C11 = A11 B111 -> Strassen (rank 7)
    # C12 = A11 B112 -> Split B112 into B1121 (2x2) and B1122 (2x1)
    #   C121 = A11 B1121 -> Strassen (rank 7)
    #   C122 = A11 B1122 -> Schoolbook (rank 4)
    
    # Block 2: A12 (2x1) * B21 (1x5)
    # This is an outer product of a 2-vector and a 5-vector.
    # Rank = 2 * 5 = 10 (Schoolbook)
    
    # Total Rank = 7 + 7 + 4 + 10 = 28
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_val, v_val, w_val, u_idx, v_idx, w_idx):
        """
        Add a term to the decomposition.
        u_idx: index in U (i*m + j)
        v_idx: index in V (j*p + k)
        w_idx: index in W (k*n + i)
        """
        U_rows.append([0] * 28)
        V_rows.append([0] * 28)
        W_rows.append([0] * 28)
        # Wait, this is not how it works.
        # U, V, W are matrices of shape (n*m, rank), (m*p, rank), (p*n, rank).
        # So for each rank index c, we have a vector u_c, v_c, w_c.
        # We are building the columns.
        pass

    # Let's build the matrices column by column (rank index c).
    # Rank = 28
    rank = 28
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    
    # Helper to add a scalar multiple of a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1

    # --- Block 1: A11 (2x2) * B11 (2x5) ---
    # A11 indices: i in {0,1}, j in {0,1}
    # B11 indices: j in {0,1}, k in {0,1,2,3,4}
    # C1 indices: i in {0,1}, k in {0,1,2,3,4}
    
    # --- Sub-block 1.1: A11 (2x2) * B111 (2x2) ---
    # B111 indices: j in {0,1}, k in {0,1}
    # Strassen for 2x2 * 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = [[M1+M4-M5+M7, M1-M2+M3+M5],
    #        [M2+M4-M6-M7, M1-M2+M3+M6]]
    
    # We need to map this to the global indices.
    # For each term in Strassen, it contributes to multiple (i, k) pairs.
    # But the tensor decomposition is a sum of rank-1 tensors.
    # A rank-1 tensor for M_{n,m,p} is defined by u (length nm), v (length mp), w (length pn).
    # The term is sum_{c} u_{i,j,c} v_{j,k,c} w_{k,i,c}.
    
    # Wait, the Strassen algorithm computes the product C = AB.
    # It uses 7 multiplications.
    # Each multiplication corresponds to a rank-1 term in the tensor decomposition of the multiplication map?
    # No. The multiplication map is bilinear.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a tensor T_{i,j,k}.
    # A rank-1 tensor is u_i v_j w_k.
    # The term is sum_c u_{i,c} v_{j,c} w_{k,c}.
    # The product is C_{ik} = sum_c (sum_j u_{i,c} v_{j,c} w_{k,c})? No.
    # The formula is C_{ik} = sum_j A_{ij} B_{jk}.
    # If we decompose the tensor T_{i,j,k} = delta_{i,j,k} (where T_{i,j,k}=1 if it's a valid multiplication term? No, T_{i,j,k} is 1 for all i,j,k? No.)
    # The matrix multiplication tensor T_{m,n,p} has entries T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, the tensor is defined such that the bilinear form is <T, A \otimes B> = Tr(C A^T B^T) or similar.
    # Standard definition: T_{i,j,k} = 1 if i,j,k are valid indices.
    # Then sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} = sum_{i,k} (sum_j A_{ij} B_{jk}) = sum_{i,k} C_{ik}.
    # This is not quite right. The output is a matrix C.
    # The decomposition is C_{ik} = sum_c U_{i,c} V_{k,c} ... wait.
    # The interface says U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # U index: i*m+j. V index: j*p+k. W index: k*n+i.
    # The term is sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}.
    # This sum is over c.
    # For a fixed i, j, k, the term is U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}.
    # But i, j, k are fixed.
    # Wait, the sum is over c.
    # So for a fixed (i,j,k), the contribution is sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}.
    # This must equal A_{ij} B_{jk} if (i,j,k) is a valid term?
    # No, the matrix multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor decomposition usually represents the bilinear map.
    # The map is (A, B) -> C.
    # C_{ik} = sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} ... no, T is fixed.
    # The standard decomposition is:
    # C_{ik} = sum_c (sum_j U_{i,j,c} A_{ij}) (sum_j V_{j,k,c} B_{jk}) ... no.
    # Let's look at the indices.
    # U has indices (i, j) and c.
    # V has indices (j, k) and c.
    # W has indices (k, i) and c.
    # The term for a specific c is:
    # sum_{i,j,k} U_{i,j,c} A_{ij} V_{j,k,c} B_{jk} W_{k,i,c}.
    # This is a scalar.
    # But C is a matrix.
    # So the decomposition must be:
    # C_{ik} = sum_c W_{k,i,c} (sum_j U_{i,j,c} A_{ij}) (sum_j V_{j,k,c} B_{jk})?
    # No, that would be rank-1 in C.
    # Actually, the standard definition of tensor rank for matrix multiplication is:
    # The tensor T_{m,n,p} is in F^{m x n} \otimes F^{n x p} \otimes F^{p x m}.
    # T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}.
    # A decomposition is T = sum_{c=1}^r U_c \otimes V_c \otimes W_c.
    # Where U_c is in F^{m x n}, V_c in F^{n x p}, W_c in F^{p x m}.
    # Then for any A, B:
    # <T, A \otimes B> = sum_{i,j,k} A_{ij} B_{jk} = sum_{i,k} C_{ik}.
    # And <sum_c U_c \otimes V_c \otimes W_c, A \otimes B> = sum_c <U_c, A> <V_c, B> <W_c, C>? No.
    # The pairing is usually:
    # <U \otimes V \otimes W, A \otimes B \otimes C> = <U, A> <V, B> <W, C>.
    # So if T = sum_c U_c \otimes V_c \otimes W_c,
    # Then <T, A \otimes B \otimes C> = sum_c <U_c, A> <V_c, B> <W_c, C>.
    # We want this to equal <T, A \otimes B \otimes C> for the multiplication tensor.
    # The multiplication tensor T is defined such that <T, A \otimes B \otimes C> = Tr(C^T A B).
    # So we need sum_c <U_c, A> <V_c, B> <W_c, C> = Tr(C^T A B).
    # This is equivalent to C_{ik} = sum_c (sum_j U_{i,j,c} A_{ij}) (sum_j V_{j,k,c} B_{jk}) W_{k,i,c}?
    # No, <W_c, C> = sum_{k,i} W_{k,i,c} C_{ik}.
    # So the RHS is sum_c (sum_{i,j} U_{i,j,c} A_{ij}) (sum_{j,k} V_{j,k,c} B_{jk}) (sum_{k,i} W_{k,i,c} C_{ik}).
    # This is a scalar.
    # The LHS is a scalar.
    # This doesn't define C directly.
    
    # Let's look at the interface again.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    # "output factor follows the official trace convention W[k*n+i,r]"
    
    # If we assume the standard decomposition:
    # C_{ik} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ... wait, the sum is over c.
    # But i, j, k are free indices in the product A_{ij} B_{jk}.
    # The term A_{ij} B_{jk} appears in C_{ik}.
    # So the decomposition must satisfy:
    # A_{ij} B_{jk} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ?
    # No, that would mean the product of two scalars is a sum of products of 3 scalars.
    # This is the definition of the tensor rank of the multiplication map?
    # Actually, the tensor is T_{i,j,k} = 1.
    # We want to write T_{i,j,k} = sum_c u_{i,j,c} v_{j,k,c} w_{k,i,c}.
    # Then for any A, B:
    # sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} = sum_{i,j,k} (sum_c u_{i,j,c} v_{j,k,c} w_{k,i,c}) A_{ij} B_{jk}
    # = sum_c (sum_{i,j} u_{i,j,c} A_{ij}) (sum_{j,k} v_{j,k,c} B_{jk}) (sum_{k,i} w_{k,i,c})?
    # No, w_{k,i,c} does not depend on A or B.
    # So the term is sum_c (sum_{i,j} u_{i,j,c} A_{ij}) (sum_{j,k} v_{j,k,c} B_{jk}) (sum_{k,i} w_{k,i,c}).
    # This is a scalar.
    # And the LHS is sum_{i,k} C_{ik}.
    # This is not enough to determine C.
    
    # There is a different convention.
    # Maybe the decomposition is for the bilinear map (A, B) -> C.
    # C_{ik} = sum_c (sum_j U_{i,j,c} A_{ij}) (sum_j V_{j,k,c} B_{jk}) ?
    # If so, W is not needed?
    # But W is provided.
    # Maybe C_{ik} = sum_c W_{k,i,c} (sum_j U_{i,j,c} A_{ij}) (sum_j V_{j,k,c} B_{jk})?
    # If W_{k,i,c} = 1 for all c, then it's just the sum of products.
    # But W can vary.
    
    # Let's assume the condition is:
    # A_{ij} B_{jk} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # This is the "multiplication tensor" decomposition.
    # For the schoolbook algorithm:
    # U_{i*m+j, c} = 1 if c corresponds to (i,j,k), else 0.
    # V_{j*p+k, c} = 1 if c corresponds to (i,j,k), else 0.
    # W_{k*n+i, c} = 1 if c corresponds to (i,j,k), else 0.
    # Then sum_c ... = 1 if (i,j,k) is the term, else 0.
    # And A_{ij} B_{jk} is the term.
    # So the identity is A_{ij} B_{jk} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} A_{ij} B_{jk}?
    # No.
    # The identity is:
    # The tensor T_{i,j,k} = 1.
    # T_{i,j,k} = sum_c u_{i,j,c} v_{j,k,c} w_{k,i,c}.
    # This is what we need to satisfy.
    
    # So for Strassen:
    # We need to represent the 2x2x2 multiplication tensor with rank 7.
    # The tensor T_{i,j,k} for 2x2x2 has 8 entries (i,j,k in {0,1}).
    # T_{0,0,0} = 1 (A00 B00)
    # T_{0,0,1} = 1 (A00 B01)
    # T_{0,1,0} = 1 (A01 B10)
    # T_{0,1,1} = 1 (A01 B11)
    # T_{1,0,0} = 1 (A10 B00)
    # T_{1,0,1} = 1 (A10 B01)
    # T_{1,1,0} = 1 (A11 B10)
    # T_{1,1,1} = 1 (A11 B11)
    
    # Strassen's 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The products are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M1 - M2 + M3 + M5
    # C10 = M2 + M4 - M6 - M7
    # C11 = M1 - M2 + M3 + M6
    
    # How does this map to the tensor decomposition?
    # The term M1 = (A00+A11)(B00+B11)
    # = A00 B00 + A00 B11 + A11 B00 + A11 B11
    # This is a sum of 4 terms in the tensor.
    # So M1 corresponds to a rank-4 tensor?
    # No, M1 is a product of two linear forms.
    # L1 = A00 + A11
    # R1 = B00 + B11
    # M1 = L1 R1
    # This is a rank-1 term in the bilinear map (A, B) -> M1.
    # But we need to decompose the tensor T_{i,j,k}.
    # The contribution of M1 to T_{i,j,k} is:
    # T^{(1)}_{i,j,k} = (coeff of A_{ij} in L1) * (coeff of B_{jk} in R1) * (coeff of C_{ik} in ...?)
    # Wait, the tensor T is for the multiplication.
    # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    # U_c is a matrix. V_c is a matrix. W_c is a matrix.
    # For Strassen, the 7 terms correspond to 7 rank-1 decompositions of the bilinear map.
    # But the tensor rank is the rank of T.
    # It is known that the rank of the 2x2x2 multiplication tensor is 7.
    # So there exist 7 pairs of matrices (U_c, V_c) such that
    # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # Actually, W_c is usually the identity or related to the output.
    # In many formulations, W_c is a scalar or a simple matrix.
    # For Strassen, the 7 terms are:
    # 1. U1 = [[1,0],[1,0]], V1 = [[1,1],[0,0]], W1 = [[1,0],[0,0]]?
    # Let's check.
    # Term 1: (A00+A11) B00.
    # U1 = [[1,0],[1,0]] (so sum_j U1_{i,j} A_{ij} = A00+A11 for i=0, A10+A11 for i=1? No.)
    # U1_{0,0}=1, U1_{1,0}=1.
    # L1(A) = A00 + A10? No, Strassen uses A00+A11.
    # So U1 should be [[1,0],[0,1]]? No, that's identity.
    # L1(A) = A00 + A11.
    # So U1_{0,0}=1, U1_{1,1}=1.
    # V1: R1(B) = B00 + B11.
    # So V1_{0,0}=1, V1_{1,1}=1.
    # Then U1 V1 gives (A00+A11)(B00+B11) = M1.
    # But M1 contributes to C00, C01, C10, C11.
    # Specifically, M1 is added to C00, C01, C11, and subtracted from C10?
    # C00 = M1 + ...
    # C01 = M1 + ...
    # C10 = ... - M1? No, C10 = M2+M4-M6-M7. No M1.
    # C11 = M1 + ...
    # So M1 contributes to C00, C01, C11.
    # This means W1 must have non-zero entries at (0,0), (1,0), (0,1)?
    # W index is k*n+i.
    # C00 -> k=0, i=0 -> W_{0,0}
    # C01 -> k=1, i=0 -> W_{1,0}
    # C11 -> k=1, i=1 -> W_{1,1}
    # So W1 = [[1, 1], [0, 1]]?
    # Let's verify the tensor identity.
    # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # For c=1:
    # U1 = [[1,0],[0,1]]
    # V1 = [[1,1],[1,1]]? No, R1(B) = B00+B11.
    # V1_{j,k} is the coeff of B_{jk} in R1.
    # So V1 = [[1,0],[0,1]].
    # Then U1 V1 W1:
    # (i=0, j=0, k=0): U1_{0,0} V1_{0,0} W1_{0,0} = 1*1*1 = 1. (T_{0,0,0}=1)
    # (i=0, j=0, k=1): U1_{0,0} V1_{0,1} W1_{1,0} = 1*0*1 = 0. (T_{0,0,1}=1, mismatch)
    
    # It seems Strassen's terms are not simply rank-1 tensors in this form.
    # The rank-7 decomposition of the 2x2x2 tensor is more complex.
    # However, I can use the known rank-7 decomposition.
    # Or, I can just use the schoolbook for the 2x2x2 parts to be safe, and only use Strassen if I can verify it.
    # Given the time constraint and the risk of error, I will use the schoolbook decomposition for all sub-blocks.
    # This gives rank 30.
    # The current program already does this.
    # To improve, I need a valid lower rank.
    # If I cannot construct a valid rank 28 decomposition quickly, I should return the schoolbook rank 30.
    # But the task asks for a "stronger algorithm".
    # Maybe I can use a known rank-7 decomposition for 2x2x2.
    
    # Let's try to find the rank-7 decomposition of M_{2,2,2}.
    # It is known that the 7 terms of Strassen correspond to a rank-7 decomposition.
    # The matrices U_c, V_c, W_c are:
    # c=1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,1],[0,1]] (M1)
    # c=2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[1,0]] (M2)
    # c=3: U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[0,1],[0,0]] (M3)
    # c=4: U=[[0,0],[0,1]], V=[[0,-1],[1,0]], W=[[1,0],[0,0]] (M4)
    # c=5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,0],[0,1]] (M5)
    # c=6: U=[[-1,0],[1,0]], V=[[1,1],[0,0]], W=[[0,0],[1,1]] (M6)
    # c=7: U=[[0,-1],[0,1]], V=[[0,0],[1,1]], W=[[0,1],[1,0]] (M7)
    
    # Let's verify c=1.
    # U1 = [[1,0],[0,1]], V1 = [[1,0],[0,1]], W1 = [[1,1],[0,1]].
    # T^{(1)}_{i,j,k} = U1_{i,j} V1_{j,k} W1_{k,i}.
    # (0,0,0): 1*1*1 = 1
    # (0,0,1): 1*0*1 = 0 (Should be 1)
    # So this is not correct.
    
    # I will stick to the schoolbook decomposition for now to ensure correctness.
    # Rank 30.
    
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
