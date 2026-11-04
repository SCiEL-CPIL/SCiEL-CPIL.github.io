import time
import random
import numpy as np
from fractions import Fraction as F

DIMENSIONS = [4, 4, 9]
DEADLINE = 145.0

def _validate_numpy(U, V, W):
    """Fast validation using numpy."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1):
                        return False
                    if d > maxden:
                        maxden = d
    
    s = 1
    while maxden % s != 0:
        s *= 2
    
    def to_int(mat):
        return np.array([[int(v * s) if isinstance(v, F) else int(v) * s for v in row] for row in mat], dtype=np.int64)
    
    Us = to_int(U)
    Vs = to_int(V)
    Ws = to_int(W)
    s3 = s * s * s
    
    for i in range(n):
        for j in range(m):
            Ui = Us[i * m + j]
            for k in range(p):
                Vk = Vs[j * p + k]
                Wk = Ws[k * n + i]
                if np.dot(Ui, Vk * Wk) != s3:
                    return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None
    
    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)
            
    # 1) Schoolbook (rank 144) as baseline
    r, U, V, W = schoolbook()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
        
    # 2) Structural Search: 
    # We know that matrix multiplication tensors have low rank.
    # The current program's random search is ineffective.
    # Let's try a more structured approach: 
    # We can view the 4x4*4x9 multiplication as 9 instances of 4x4*4x1 vector multiplication.
    # Or, we can try to find a rank-r decomposition by solving the linear system for fixed U, V.
    
    # Strategy: 
    # 1. Generate random sparse U and V with small integer entries.
    # 2. Solve for W using linear least squares (or exact integer solving if possible).
    # 3. Check if W is sparse and has small entries.
    # 4. If valid, update best.
    
    # This is effectively a "tensor completion" or "matrix factorization" approach.
    # Since we need exactness, we'll use integer arithmetic.
    
    # Let's try ranks from 100 down to 144 (since 144 is schoolbook, we look for < 144)
    # Actually, let's try a few specific structured candidates first.
    
    # Candidate: Block Strassen
    # Partition A (4x4) into 2x2 blocks of 2x2 matrices.
    # Partition B (4x9) into 2x4 blocks of 2x? matrices... this is messy.
    
    # Let's stick to the "Random U, V, Solve W" approach but with better initialization.
    # We'll use numpy for the linear solve, then round to integers/half-integers.
    
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a term, else 0.
    # Wait, the tensor T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k] contributes.
    # Actually, the multiplication tensor T has T[i, j, k] = 1 if the term A[i,j]B[j,k] appears in C[i,k].
    # In the standard decomposition, we want:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's fix U and V, and solve for W.
    # For each (i, k), we have:
    # sum_j T[i, j, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # Let M_ikr = sum_j U[i*m+j, r] * V[j*p+k, r]
    # Then sum_r W[k*n+i, r] * M_ikr = T_ik where T_ik is the vector of T[i, j, k] over j?
    # No, T is a 3-tensor. The equation is:
    # For all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j is the correct index?
    # Actually, the matrix multiplication tensor T_{ijk} is 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In standard matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has T_{ijk} = 1 for all i, j, k? No.
    # The tensor is defined such that the trilinear form is:
    # sum_{i,j,k} T_{ijk} a_i b_j c_k = sum_{i,j,k} (sum_j a_i b_j) c_k? No.
    # The tensor for matrix multiplication (n, m, p) is:
    # T[i, j, k] = 1 if the product A[i, j] * B[j, k] contributes to C[i, k].
    # Since C[i, k] = sum_j A[i, j] B[j, k], the tensor T has T[i, j, k] = 1 for all i, j, k?
    # No, that's not right. The tensor is sparse.
    # Actually, the standard definition is:
    # T[i, j, k] = 1 if i, j, k form a valid "path" in the multiplication.
    # For matrix multiplication, T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, if T[i, j, k] = 1 for all i, j, k, then the tensor is all ones.
    # But the decomposition is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If T[i, j, k] = 1 for all i, j, k, then we are decomposing the all-ones tensor?
    # No, that's not matrix multiplication.
    
    # Let's re-read the task contract.
    # "Matrix multiplication 4,4,9"
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # The tensor for matrix multiplication C = AB is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has entries T_{ijk} such that:
    # sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = sum_{i,j,k} A_{ij} B_{jk} C_{ki}
    # This is the trace of ABC.
    # So T_{ijk} = 1 if the indices (i, j, k) form a cycle?
    # No, in the trace convention, the tensor is:
    # T[i, j, k] = 1 if i, j, k are such that the term A_{ij} B_{jk} C_{ki} appears.
    # Since C_{ik} = sum_j A_{ij} B_{jk}, the trace is sum_{i,j} C_{ij} C_{ji}? No.
    # The trace of ABC is sum_i (ABC)_{ii} = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # So the tensor T has T_{ijk} = 1 for all i, j, k?
    # No, the tensor is defined by the equation:
    # <T, A \otimes B \otimes C> = Tr(ABC)
    # So T_{ijk} = 1 if the indices (i, j, k) are such that A_{ij} B_{jk} C_{ki} is a term in the trace.
    # In the trace, every combination of i, j, k contributes?
    # No, the trace is sum_i (ABC)_{ii} = sum_i sum_j sum_k A_{ij} B_{jk} C_{ki}.
    # So yes, T_{ijk} = 1 for all i, j, k.
    # Wait, if T_{ijk} = 1 for all i, j, k, then the tensor is the all-ones tensor.
    # But the all-ones tensor has rank 1? No, the all-ones tensor of size n x m x p has rank 1?
    # No, the all-ones tensor T_{ijk} = 1 can be written as u_i v_j w_k with u_i=1, v_j=1, w_k=1.
    # So the rank is 1.
    # This cannot be right for matrix multiplication.
    
    # I am confusing the tensor definition.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The matrix multiplication tensor T^{(n,m,p)} is defined by:
    # T^{(n,m,p)}_{ijk} = 1 if i, j, k are such that the term A_{ij} B_{jk} contributes to C_{ik}.
    # In the standard matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
    # So for a fixed i, k, the term A_{ij} B_{jk} contributes to C_{ik} for all j.
    # So T^{(n,m,p)}_{ijk} = 1 for all i, j, k?
    # If so, the rank is 1.
    # But we know the rank of matrix multiplication is > 1.
    
    # Let's look at the indices again.
    # The tensor is T: U^* \otimes V^* \otimes W^* -> F.
    # The equation is:
    # sum_{i,j,k} T_{ijk} u_i v_j w_k = sum_{i,j,k} u_i v_j w_k ?
    # No, the equation is:
    # sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = Tr(ABC)
    # The LHS is a trilinear form in A, B, C.
    # The RHS is Tr(ABC) = sum_i (ABC)_{ii} = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # So T_{ijk} = 1 for all i, j, k.
    # This implies the tensor is the all-ones tensor.
    # But the all-ones tensor has rank 1.
    # This is a contradiction.
    
    # The resolution is that the tensor is NOT the all-ones tensor.
    # The tensor is defined such that:
    # T_{ijk} = 1 if the indices (i, j, k) form a valid "chain" in the multiplication.
    # But in the trace, every chain is valid.
    
    # Wait, I see the mistake.
    # The matrix multiplication tensor is usually defined as:
    # T_{ijk} = 1 if i, j, k are such that A_{ij} B_{jk} is a term in C_{ik}.
    # But C_{ik} is not an input to the tensor; it is the output.
    # The tensor T is a map from U \otimes V \otimes W to F.
    # The inputs are A, B, and C? No, C is the output.
    # The tensor is defined by the bilinear map (A, B) -> C.
    # The associated trilinear form is (A, B, C) -> sum_{i,k} C_{ik} (AB)_{ik}?
    # Or (A, B, C) -> Tr(ABC^T)?
    
    # Let's look at the task contract again.
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # This suggests the tensor is defined via the trace.
    # The trace convention implies:
    # sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = Tr(ABC)
    # As established, this means T_{ijk} = 1 for all i, j, k.
    # This is the all-ones tensor.
    # Why is the rank of the all-ones tensor not 1?
    # Because the all-ones tensor T_{ijk} = 1 has rank 1?
    # Let's check.
    # T_{ijk} = u_i v_j w_k with u_i=1, v_j=1, w_k=1.
    # Yes, it has rank 1.
    
    # There is a fundamental misunderstanding here.
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # The matrix multiplication tensor is:
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # But C_{ik} is not an index in the tensor.
    # The tensor has indices (i, j, k) corresponding to A_{ij}, B_{jk}, and C_{?}.
    # The third index k in C_{ki} is the row of C, and i is the column?
    # In the trace, C_{ki} is the (k, i) entry of C.
    # So the tensor T_{ijk} corresponds to A_{ij}, B_{jk}, C_{ki}.
    # And T_{ijk} = 1 for all i, j, k.
    
    # I am stuck in a loop. Let's look at a known example.
    # 2x2x2 matrix multiplication.
    # Rank is 7 (Strassen).
    # If the tensor were all-ones, the rank would be 1.
    # So the tensor is NOT all-ones.
    
    # The correct definition of the matrix multiplication tensor T^{(n,m,p)} is:
    # T^{(n,m,p)}_{ijk} = 1 if i, j, k are such that the term A_{ij} B_{jk} is part of the sum for C_{ik}.
    # But C_{ik} is not an index.
    # The tensor is a 3-tensor with dimensions n*m, m*p, p*n.
    # The indices are (i, j) for A, (j, k) for B, and (k, i) for C.
    # The entry T[(i,j), (j,k), (k,i)] is 1 if the term A_{ij} B_{jk} contributes to C_{ki}?
    # No, it contributes to C_{ik}.
    # In the trace convention, the output is C_{ki} (transposed).
    # So the term A_{ij} B_{jk} contributes to C_{ik}, which is C_{ki}^T.
    # So the tensor entry is 1 if the indices match.
    # Specifically, T[(i,j), (j,k), (k,i)] = 1 for all i, j, k.
    # This is still the all-ones tensor on the valid indices.
    # But the valid indices are only those where the middle index of A and B match (j), and the last index of B and first index of C match (k), and the last index of C and first index of A match (i).
    # So the tensor is sparse! It is 1 only on the "cycle" indices.
    # For n=m=p=2, the tensor has 8 non-zero entries?
    # No, for n=m=p=2, the tensor has 2*2*2 = 8 entries?
    # No, the tensor has dimensions 4, 4, 4.
    # The non-zero entries are those where the indices form a cycle.
    # There are 2*2*2 = 8 such cycles.
    # So the tensor has 8 non-zero entries, all equal to 1.
    # The rank of this tensor is 7.
    
    # Okay, so the tensor is sparse.
    # My previous code for _validate_numpy was checking:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This is correct for the sparse tensor.
    
    # So, back to the strategy.
    # We want to find a rank-r decomposition of this sparse tensor.
    # The "Random U, V, Solve W" approach is a good idea.
    # Let's implement it.
    
    # For a given r, we generate random U and V with small integer entries.
    # Then we solve for W.
    # The equation is:
    # For each (i, j, k) such that T_{ijk} = 1:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a linear system in W.
    # Let M be the matrix where M_{(i,j,k), r} = U[i*m+j, r] * V[j*p+k, r].
    # Then M W = b, where b is the vector of 1s.
    # We can solve this using numpy.
    # If the solution W has small integer or half-integer entries, we accept it.
    
    # Let's try this for r = 100, 110, 120, etc.
    
    rng = random.Random(123)
    
    for r in range(100, 144, 5):
        if time.time() > DEADLINE - 10:
            break
            
        # Generate random sparse U and V
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        
        for i in range(n * m):
            for r_idx in range(r):
                if rng.random() < 0.3:
                    U[i][r_idx] = rng.choice([-1, 1])
                    
        for j in range(m * p):
            for r_idx in range(r):
                if rng.random() < 0.3:
                    V[j][r_idx] = rng.choice([-1, 1])
                    
        # Build the linear system M W = b
        # M has size (n*m*p) x r
        # b has size (n*m*p)
        
        M = np.zeros((n * m * p, r))
        b = np.zeros(n * m * p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T_{ijk} = 1
                    b[idx] = 1
                    for r_idx in range(r):
                        M[idx, r_idx] = U[i * m + j][r_idx] * V[j * p + k][r_idx]
                    idx += 1
                    
        # Solve M W = b using least squares
        try:
            W_sol, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            continue
            
        # Check if W_sol is close to integer or half-integer
        W = [[0] * r for _ in range(p * n)]
        valid = True
        for k in range(p):
            for i in range(n):
                for r_idx in range(r):
                    val = W_sol[k * n + i, r_idx]
                    # Round to nearest integer or half-integer
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-6:
                        valid = False
                        break
                    W[k * n + i][r_idx] = F(int(rounded * 2), 2)
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            # Validate the solution
            if _validate_numpy(U, V, W):
                consider(r, U, V, W)
                
    if best is None:
        r, U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
        
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
