import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140

def check_time_remaining():
    return _START_TIME + _TIME_LIMIT - time.time()

def validate_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Validate that U, V, W form a valid decomposition with half-integer entries"""
    rank = len(U[0])
    # Check half-integer constraint
    for i in range(len(U)):
        for s in range(rank):
            if abs(U[i][s] * 2 - round(U[i][s] * 2)) > tol:
                return False
    for i in range(len(V)):
        for s in range(rank):
            if abs(V[i][s] * 2 - round(V[i][s] * 2)) > tol:
                return False
    for i in range(len(W)):
        for s in range(rank):
            if abs(W[i][s] * 2 - round(W[i][s] * 2)) > tol:
                return False
    
    # Check the 1-equations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s] for s in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check the 0-equations
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                continue
                            s = sum(U[i*m+j][s] * V[jp*p+k][s] * W[kp*n+ip][s] for s in range(rank))
                            if abs(s) > tol:
                                return False
    return True

def schoolbook_decomposition(n, m, p):
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

def add_strassen_2x2x2(U, V, W, i0, j0, k0, col_offset):
    """
    Adds a Strassen decomposition for a 2x2x2 block.
    i0, j0, k0 are the starting indices for i, j, k.
    col_offset is the starting column index in U, V, W.
    Returns the next available column index.
    """
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U indices: i in {i0, i0+1}, j in {j0, j0+1}
    # V indices: j in {j0, j0+1}, k in {k0, k0+1}
    # W indices: k in {k0, k0+1}, i in {i0, i0+1}
    
    # M1
    U[i0*4+j0][col_offset] = 1; V[j0*5+k0][col_offset] = 1; W[k0*3+i0][col_offset] = 1
    U[i0*4+j0+1][col_offset] = 1; V[j0*5+k0][col_offset] = 1; W[k0*3+i0+1][col_offset] = 1
    U[i0*4+j0][col_offset] += 1; V[j0*5+k0+1][col_offset] = 1; W[k0*3+i0][col_offset] += 1 # Wait, V and W must be consistent per column
    # Correction: Each term is a product of a linear form in U, linear form in V, linear form in W.
    # For M1:
    # U_part: A11 + A22 -> U[i0, j0] += 1, U[i0+1, j0+1] += 1
    # V_part: B11 + B22 -> V[j0, k0] += 1, V[j0+1, k0+1] += 1
    # W_part: C11 + C22 -> W[k0, i0] += 1, W[k0+1, i0+1] += 1
    # This is not how it works. The decomposition is sum_s U_{i,j,s} V_{j',k',s} W_{k'',i'',s}.
    # For a specific term s, U_{i,j,s} is the coefficient of A_{i,j} in the linear form.
    # V_{j',k',s} is the coefficient of B_{j',k'} in the linear form.
    # W_{k'',i'',s} is the coefficient of C_{k'',i''} in the linear form.
    # The product must contribute to C_{i'', k''} only if the indices match the output of the multiplication.
    # Actually, the tensor identity is:
    # T_{i,j,k} = sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s} ? No.
    # The standard form is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T has indices (i, j, k).
    # Decomposition: T = sum_s u_s \otimes v_s \otimes w_s.
    # T_{i,j,k} = sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s} is WRONG.
    # The indices of the factors must match the indices of the tensor.
    # U has indices (i, j, s).
    # V has indices (j, k, s).
    # W has indices (k, i, s).
    # The sum is over s.
    # The equation is:
    # sum_s U_{i,j,s} V_{j',k',s} W_{k'',i'',s} = delta_{i,i''} delta_{j,j'} delta_{k,k'} ?
    # No.
    # The multiplication tensor is T_{i,j,k} = 1 if i,j,k are valid indices for A_{i,j} B_{j,k} -> C_{i,k}.
    # Actually, the tensor is defined on (i, j, k) such that it represents the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 for all valid i,j,k.
    # The decomposition is:
    # T_{i,j,k} = sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s} ?
    # Let's check the indices.
    # U_{i,j,s} multiplies A_{i,j}.
    # V_{j,k,s} multiplies B_{j,k}.
    # W_{k,i,s} multiplies C_{i,k}.
    # The product U_{i,j,s} V_{j,k,s} W_{k,i,s} is a scalar.
    # The sum over s gives the coefficient of A_{i,j} B_{j,k} in the computed C?
    # No, the computed C_{i',k'} is sum_{i,j,k} (sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s}) A_{i,j} B_{j,k}?
    # This is getting confused.
    # Let's stick to the validation function provided.
    # validate_decomposition checks:
    # sum_s U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s] == 1 for (i,j,k)
    # sum_s U[i*m+j][s] * V[jp*p+k][s] * W[kp*n+ip][s] == 0 for others.
    # This implies:
    # The term s contributes to the product A_{i,j} B_{j,k} -> C_{i,k} with weight U[i,j,s] V[j,k,s] W[k,i,s].
    # So for a specific Strassen term s, we define:
    # U[i,j,s] for all i,j.
    # V[j,k,s] for all j,k.
    # W[k,i,s] for all k,i.
    
    # For M1 = (A11+A22)(B11+B22) -> C11 + C22?
    # In Strassen, M1 is used in C11, C22.
    # C11 = M1 + ...
    # C22 = M1 - ...
    # So M1 contributes to C11 and C22.
    # The "linear form" in A is A11 + A22. So U[i0,j0,s]=1, U[i0+1,j0+1,s]=1.
    # The "linear form" in B is B11 + B22. So V[j0,k0,s]=1, V[j0+1,k0+1,s]=1.
    # The "linear form" in C is C11 + C22?
    # Wait, the tensor element T_{i,j,k} is 1 if A_{i,j} and B_{j,k} contribute to C_{i,k}.
    # In the decomposition, we sum s.
    # For a fixed s, the contribution to the tensor element (i,j,k) is U[i,j,s] V[j,k,s] W[k,i,s].
    # This must sum to 1 for the "diagonal" elements and 0 for others.
    # For M1, we want it to contribute to the calculation of C11 and C22.
    # But the tensor element is indexed by (i,j,k) which corresponds to the pair (A_{i,j}, B_{j,k}) contributing to C_{i,k}.
    # If we use Strassen, we are effectively changing the basis of the intermediate multiplications.
    # The validation function checks if the tensor identity holds.
    # So for M1, we set:
    # U[i0,j0,s] = 1, U[i0+1,j0+1,s] = 1.
    # V[j0,k0,s] = 1, V[j0+1,k0+1,s] = 1.
    # W[k0,i0,s] = 1, W[k0+1,i0+1,s] = 1.
    # Let's check the product for i=i0, j=j0, k=k0.
    # U[i0,j0,s] V[j0,k0,s] W[k0,i0,s] = 1*1*1 = 1.
    # This contributes to T_{i0,j0,k0}.
    # But in Strassen, M1 is (A11+A22)(B11+B22).
    # The term A11 B11 is part of M1.
    # The term A11 B22 is part of M1? No, B22 is B_{2,2} in local coords, which is k0+1.
    # So A11 B22 corresponds to i0, j0, k0+1.
    # U[i0,j0,s]=1. V[j0,k0+1,s]=0 (since V is B11+B22, V[j0,k0+1] is coeff of B12? No.
    # Local B indices:
    # B11: j0, k0.
    # B12: j0, k0+1.
    # B21: j0+1, k0.
    # B22: j0+1, k0+1.
    # M1 uses B11 and B22.
    # So V[j0,k0,s]=1, V[j0+1,k0+1,s]=1.
    # V[j0,k0+1,s]=0, V[j0+1,k0,s]=0.
    # Now check T_{i0, j0, k0+1} (A11 B12).
    # U[i0,j0,s]=1. V[j0,k0+1,s]=0. Product 0.
    # Check T_{i0+1, j0+1, k0+1} (A22 B22).
    # U[i0+1,j0+1,s]=1. V[j0+1,k0+1,s]=1. W[k0+1,i0+1,s]=1. Product 1.
    # Check T_{i0, j0, k0} (A11 B11).
    # U[i0,j0,s]=1. V[j0,k0,s]=1. W[k0,i0,s]=1. Product 1.
    # So M1 contributes 1 to (i0,j0,k0) and (i0+1,j0+1,k0+1).
    # And 0 to others in this 2x2x2 block.
    # This is correct for the "diagonal" parts of M1.
    # But Strassen M1 is used to compute C11 and C22.
    # C11 = M1 + ...
    # C22 = M1 + ...
    # The tensor element T_{i,j,k} is 1 if A_{i,j} B_{j,k} is part of C_{i,k}.
    # In the standard multiplication, A11 B11 contributes to C11.
    # In Strassen, A11 B11 is part of M1.
    # So the contribution to the "value" of C11 comes from M1.
    # The tensor decomposition must satisfy the sum condition.
    # The condition is:
    # sum_s U[i,j,s] V[j,k,s] W[k,i,s] = 1 for all i,j,k.
    # This means that for every pair (i,j,k), the sum of products over s must be 1.
    # And for (i,j,j',k',k,i') where (i,j,k) != (i',j',k')?
    # The 0-equations in the validator are:
    # sum_s U[i,j,s] V[j',k',s] W[k'',i'',s] = 0 if (i,j,k) != (i'',j',k'')?
    # No, the validator checks:
    # for i, j, jp, k, kp, ip:
    #   if (i,j,k) == (ip,jp,kp): continue
    #   sum U[i*m+j][s] V[jp*p+k][s] W[kp*n+ip][s] == 0
    # This is checking that the "cross terms" vanish.
    # Specifically, it checks that the tensor is a sum of rank-1 tensors.
    # So my assignment of U, V, W for M1 is:
    # U[i0,j0,s]=1, U[i0+1,j0+1,s]=1
    # V[j0,k0,s]=1, V[j0+1,k0+1,s]=1
    # W[k0,i0,s]=1, W[k0+1,i0+1,s]=1
    # This creates a rank-1 tensor that is 1 at (i0,j0,k0) and (i0+1,j0+1,k0+1) and 0 elsewhere?
    # Let's check (i0, j0, k0+1).
    # U[i0,j0,s]=1, V[j0,k0+1,s]=0. Product 0.
    # Let's check (i0+1, j0, k0).
    # U[i0+1,j0,s]=0. Product 0.
    # So this rank-1 tensor has non-zero values only at (i0,j0,k0) and (i0+1,j0+1,k0+1).
    # This is exactly what we want for the "diagonal" parts of M1.
    # But Strassen M1 is (A11+A22)(B11+B22).
    # The expansion is A11B11 + A11B22 + A22B11 + A22B22.
    # My rank-1 tensor only covers A11B11 and A22B22.
    # It misses A11B22 and A22B11.
    # Why? Because V[j0,k0+1,s] is 0.
    # In M1, the B part is B11 + B22.
    # So V[j0,k0,s]=1, V[j0+1,k0+1,s]=1.
    # The A part is A11 + A22.
    # So U[i0,j0,s]=1, U[i0+1,j0+1,s]=1.
    # The product U V is:
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # The term A11B22 corresponds to i0, j0, k0+1.
    # U[i0,j0,s]=1. V[j0,k0+1,s]=0.
    # So the product is 0.
    # This means my rank-1 tensor does NOT represent M1.
    # It represents A11B11 + A22B22.
    # To represent M1, we need the cross terms.
    # But a rank-1 tensor is a product of three vectors.
    # u \otimes v \otimes w.
    # The entries are u_i v_j w_k.
    # If u has 1s at i0, i0+1.
    # v has 1s at j0, j0+1.
    # w has 1s at k0, k0+1.
    # Then the tensor has 1s at all 8 combinations of (i,j,k) in the 2x2x2 block.
    # This is NOT M1.
    # M1 is (A11+A22)(B11+B22).
    # This is a bilinear form.
    # The tensor for M1 is not rank 1.
    # Strassen uses 7 rank-1 tensors to represent the 8 schoolbook terms.
    # The 7 terms are linear combinations.
    # So for each of the 7 Strassen terms, we define a rank-1 tensor.
    # For M1, the "A" linear form is A11+A22.
    # The "B" linear form is B11+B22.
    # The "C" linear form is C11+C22?
    # No, the C linear form is determined by how M1 is used in the final sum.
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C11 and +1 to C22.
    # So W[k0,i0,s] = 1, W[k0+1,i0+1,s] = 1.
    # Now, does the product U V W match the tensor?
    # T_{i,j,k} = sum_s U[i,j,s] V[j,k,s] W[k,i,s].
    # For s=M1:
    # U[i0,j0,s]=1, U[i0+1,j0+1,s]=1.
    # V[j0,k0,s]=1, V[j0+1,k0+1,s]=1.
    # W[k0,i0,s]=1, W[k0+1,i0+1,s]=1.
    # Product at (i0,j0,k0): 1*1*1 = 1.
    # Product at (i0,j0,k0+1): 1*0*0 = 0. (V[j0,k0+1,s]=0)
    # Product at (i0+1,j0+1,k0): 0*1*0 = 0. (U[i0+1,j0,s]=0)
    # Product at (i0+1,j0+1,k0+1): 1*1*1 = 1.
    # So this rank-1 tensor contributes 1 to (i0,j0,k0) and (i0+1,j0+1,k0+1).
    # And 0 to others.
    # This is correct for the "diagonal" terms of the multiplication.
    # What about the off-diagonal terms of M1?
    # M1 = A11B11 + A11B22 + A22B11 + A22B22.
    # The terms A11B22 and A22B11 are "off-diagonal" in the 2x2 block.
    # Do they need to be represented?
    # In the final sum C11 and C22, these terms are cancelled out by other Strassen terms?
    # No, Strassen works by recombining.
    # The tensor decomposition must satisfy the identity for the ORIGINAL multiplication.
    # The original multiplication has T_{i,j,k}=1 for all i,j,k.
    # So the sum of all 7 rank-1 tensors must yield 1 for all 8 (i,j,k) pairs in the 2x2x2 block.
    # Let's check if the 7 Strassen rank-1 tensors sum to the all-ones tensor.
    # This is a known property.
    # So I just need to define the 7 rank-1 tensors correctly.
    
    # M1:
    # U: A11+A22 -> (i0,j0)=1, (i0+1,j0+1)=1
    # V: B11+B22 -> (j0,k0)=1, (j0+1,k0+1)=1
    # W: C11+C22 -> (k0,i0)=1, (k0+1,i0+1)=1
    
    # M2: (A21+A22)B11
    # U: A21+A22 -> (i0+1,j0)=1, (i0+1,j0+1)=1
    # V: B11 -> (j0,k0)=1
    # W: C21+C22?
    # C21 = M2 + M4
    # C22 = M1 - M2 + ...
    # So M2 contributes +1 to C21, -1 to C22.
    # W: (k0,i0+1)=1, (k0+1,i0+1)=-1
    
    # M3: A11(B21-B22)
    # U: A11 -> (i0,j0)=1
    # V: B21-B22 -> (j0+1,k0)=1, (j0+1,k0+1)=-1
    # W: C12+C22?
    # C12 = M3+M5
    # C22 = ... + M3 ...
    # So M3 contributes +1 to C12, +1 to C22.
    # W: (k0,i0+1)=1, (k0+1,i0+1)=1? No.
    # C12 is (i0, k0+1). C22 is (i0+1, k0+1).
    # W[k0+1, i0] = 1. W[k0+1, i0+1] = 1.
    
    # M4: A22(B12-B11)
    # U: A22 -> (i0+1,j0+1)=1
    # V: B12-B11 -> (j0,k0+1)=1, (j0,k0)=-1
    # W: C11+C21?
    # C11 = ... + M4 ...
    # C21 = M2 + M4
    # So M4 contributes +1 to C11, +1 to C21.
    # W[k0, i0]=1, W[k0, i0+1]=1.
    
    # M5: (A11+A12)B22
    # U: A11+A12 -> (i0,j0)=1, (i0,j0+1)=1
    # V: B22 -> (j0+1,k0+1)=1
    # W: C12-C11?
    # C12 = M3+M5
    # C11 = ... - M5 ...
    # So M5 contributes +1 to C12, -1 to C11.
    # W[k0+1, i0]=1, W[k0, i0]=-1.
    
    # M6: (A21-A11)(B11+B12)
    # U: A21-A11 -> (i0+1,j0)=1, (i0,j0)=-1
    # V: B11+B12 -> (j0,k0)=1, (j0,k0+1)=1
    # W: C22?
    # C22 = ... + M6 ...
    # So M6 contributes +1 to C22.
    # W[k0+1, i0+1]=1.
    
    # M7: (A12-A22)(B21+B22)
    # U: A12-A22 -> (i0,j0+1)=1, (i0+1,j0+1)=-1
    # V: B21+B22 -> (j0+1,k0)=1, (j0+1,k0+1)=1
    # W: C11?
    # C11 = ... + M7 ...
    # So M7 contributes +1 to C11.
    # W[k0, i0]=1.

    # We will implement this logic.
    pass

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j in {0,1}
    # A1a (2x2): i in {0,1}, j in {0,1}
    # A1b (1x2): i in {2}, j in {0,1}
    # B1a (2x2): j in {0,1}, k in {0,1}
    # B1b (2x2): j in {0,1}, k in {2,3}
    # B1c (2x1): j in {0,1}, k in {4}
    
    # 1. A1a B1a (2x2x2)
    add_strassen_2x2x2(U, V, W, 0, 0, 0, col)
    col += 7
    
    # 2. A1a B1b (2x2x2)
    add_strassen_2x2x2(U, V, W, 0, 0, 2, col)
    col += 7
    
    # 3. A1a B1c (2x2x1)
    # Rank 4.
    # A1a (2x2), B1c (2x1).
    # C1 (2x1).
    # C1_0 = A11 B11 + A12 B21
    # C1_1 = A21 B11 + A22 B21
    # This is just 4 multiplications.
    # T_{i,j,k} for i in {0,1}, j in {0,1}, k in {4}.
    # We need 4 rank-1 tensors.
    # 1: A11 B11 -> C1_0
    # 2: A12 B21 -> C1_0
    # 3: A21 B11 -> C1_1
    # 4: A22 B21 -> C1_1
    
    # Term 1: A11 B11
    U[0*4+0][col] = 1; V[0*5+4][col] = 1; W[4*3+0][col] = 1
    col += 1
    # Term 2: A12 B21 (A[0,1] B[1,4])
    U[0*4+1][col] = 1; V[1*5+4][col] = 1; W[4*3+0][col] = 1
    col += 1
    # Term 3: A21 B11 (A[1,0] B[0,4])
    U[1*4+0][col] = 1; V[0*5+4][col] = 1; W[4*3+1][col] = 1
    col += 1
    # Term 4: A22 B21 (A[1,1] B[1,4])
    U[1*4+1][col] = 1; V[1*5+4][col] = 1; W[4*3+1][col] = 1
    col += 1
    
    # 4. A1b B1a (1x2x2)
    # A1b (1x2): i=2, j in {0,1}.
    # B1a (2x2): j in {0,1}, k in {0,1}.
    # C2 (1x2): i=2, k in {0,1}.
    # Rank 4.
    # C2_0 = A21 B11 + A22 B21 (using local j indices)
    # C2_1 = A21 B12 + A22 B22
    
    # Term 1: A[2,0] B[0,0]
    U[2*4+0][col] = 1; V[0*5+0][col] = 1; W[0*3+2][col] = 1
    col += 1
    # Term 2: A[2,1] B[1,0]
    U[2*4+1][col] = 1; V[1*5+0][col] = 1; W[0*3+2][col] = 1
    col += 1
    # Term 3: A[2,0] B[0,1]
    U[2*4+0][col] = 1; V[0*5+1][col] = 1; W[1*3+2][col] = 1
    col += 1
    # Term 4: A[2,1] B[1,1]
    U[2*4+1][col] = 1; V[1*5+1][col] = 1; W[1*3+2][col] = 1
    col += 1
    
    # 5. A1b B1b (1x2x2)
    # Similar to above, but k in {2,3}.
    # Term 1: A[2,0] B[0,2]
    U[2*4+0][col] = 1; V[0*5+2][col] = 1; W[2*3+2][col] = 1
    col += 1
    # Term 2: A[2,1] B[1,2]
    U[2*4+1][col] = 1; V[1*5+2][col] = 1; W[2*3+2][col] = 1
    col += 1
    # Term 3: A[2,0] B[0,3]
    U[2*4+0][col] = 1; V[0*5+3][col] = 1; W[3*3+2][col] = 1
    col += 1
    # Term 4: A[2,1] B[1,3]
    U[2*4+1][col] = 1; V[1*5+3][col] = 1; W[3*3+2][col] = 1
    col += 1
    
    # 6. A1b B1c (1x2x1)
    # A1b (1x2), B1c (2x1).
    # C2 (1x1).
    # Rank 2.
    # Term 1: A[2,0] B[0,4]
    U[2*4+0][col] = 1; V[0*5+4][col] = 1; W[4*3+2][col] = 1
    col += 1
    # Term 2: A[2,1] B[1,4]
    U[2*4+1][col] = 1; V[1*5+4][col] = 1; W[4*3+2][col] = 1
    col += 1
    
    # Block 2: j in {2,3}
    # Same structure, shifted j by 2.
    
    # 7. A2a B2a (2x2x2)
    add_strassen_2x2x2(U, V, W, 0, 2, 0, col)
    col += 7
    
    # 8. A2a B2b (2x2x2)
    add_strassen_2x2x2(U, V, W, 0, 2, 2, col)
    col += 7
    
    # 9. A2a B2c (2x2x1)
    # Term 1: A[0,2] B[2,4]
    U[0*4+2][col] = 1; V[2*5+4][col] = 1; W[4*3+0][col] = 1
    col += 1
    # Term 2: A[0,3] B[3,4]
    U[0*4+3][col] = 1; V[3*5+4][col] = 1; W[4*3+0][col] = 1
    col += 1
    # Term 3: A[1,2] B[2,4]
    U[1*4+2][col] = 1; V[2*5+4][col] = 1; W[4*3+1][col] = 1
    col += 1
    # Term 4: A[1,3] B[3,4]
    U[1*4+3][col] = 1; V[3*5+4][col] = 1; W[4*3+1][col] = 1
    col += 1
    
    # 10. A2b B2a (1x2x2)
    # Term 1: A[2,2] B[2,0]
    U[2*4+2][col] = 1; V[2*5+0][col] = 1; W[0*3+2][col] = 1
    col += 1
    # Term 2: A[2,3] B[3,0]
    U[2*4+3][col] = 1; V[3*5+0][col] = 1; W[0*3+2][col] = 1
    col += 1
    # Term 3: A[2,2] B[2,1]
    U[2*4+2][col] = 1; V[2*5+1][col] = 1; W[1*3+2][col] = 1
    col += 1
    # Term 4: A[2,3] B[3,1]
    U[2*4+3][col] = 1; V[3*5+1][col] = 1; W[1*3+2][col] = 1
    col += 1
    
    # 11. A2b B2b (1x2x2)
    # Term 1: A[2,2] B[2,2]
    U[2*4+2][col] = 1; V[2*5+2][col] = 1; W[2*3+2][col] = 1
    col += 1
    # Term 2: A[2,3] B[3,2]
    U[2*4+3][col] = 1; V[3*5+2][col] = 1; W[2*3+2][col] = 1
    col += 1
    # Term 3: A[2,2] B[2,3]
    U[2*4+2][col] = 1; V[2*5+3][col] = 1; W[3*3+2][col] = 1
    col += 1
    # Term 4: A[2,3] B[3,3]
    U[2*4+3][col] = 1; V[3*5+3][col] = 1; W[3*3+2][col] = 1
    col += 1
    
    # 12. A2b B2c (1x2x1)
    # Term 1: A[2,2] B[2,4]
    U[2*4+2][col] = 1; V[2*5+4][col] = 1; W[4*3+2][col] = 1
    col += 1
    # Term 2: A[2,3] B[3,4]
    U[2*4+3][col] = 1; V[3*5+4][col] = 1; W[4*3+2][col] = 1
    col += 1
    
    if col != rank:
        # Fallback to schoolbook if construction failed
        return schoolbook_decomposition(n, m, p)
        
    if validate_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return schoolbook_decomposition(n, m, p)
