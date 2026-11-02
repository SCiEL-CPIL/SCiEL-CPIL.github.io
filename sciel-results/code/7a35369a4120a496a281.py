import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (15, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # --- Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # Strassen for 2x2x2
    # A indices: 0,1 (i), 0,1 (j) -> U[i*3+j]
    # B indices: 0,1 (j), 0,1 (k) -> V[j*5+k]
    # C indices: 0,1 (k), 0,1 (i) -> W[k*2+i]
    
    # p1 = (a+d)(e+h)
    # a=U[0,0]=U[0], d=U[1,1]=U[4]
    # e=V[0,0]=V[0], h=V[1,1]=V[6]
    # We need to form linear combinations.
    # The standard way is to define the U, V, W entries directly for each bilinear form.
    # For Strassen, the forms are:
    # 1. (A00 + A11) * (B00 + B11)
    # 2. (A10 + A11) * B00
    # 3. A00 * (B01 - B11)
    # 4. A11 * (B10 - B00)
    # 5. (A00 + A01) * B11
    # 6. (A10 - A00) * (B00 + B01)
    # 7. (A01 - A11) * (B10 + B11)
    
    # We will add 7 terms for this block.
    
    # Term 1: (A00 + A11) * (B00 + B11)
    # U: A00, A11. V: B00, B11. W: C00, C01, C10, C11 with coeffs.
    # This is complex to map directly to single U/V/W entries if we want to reuse.
    # But here we are building the full U, V, W matrices.
    # Each "term" in the tensor decomposition corresponds to one column in U, V, W.
    # The entries in U, V, W for that column define the bilinear form.
    
    # So for Strassen Term 1:
    # U col: A00=1, A11=1. Others 0.
    # V col: B00=1, B11=1. Others 0.
    # W col: C00=1, C01=1, C10=1, C11=1. (Wait, Strassen recombination is linear combination of p's).
    # No, the decomposition is $C_{ij} = \sum_k U_{ik} V_{jk} W_{kj}$.
    # So for each k (rank index), we have a U vector, V vector, W vector.
    # The product $\sum_k U_{i,k} V_{j,k} W_{k,?}$?
    # The convention is $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$?
    # Task says: $W[k*n+i, r]$.
    # So $C_{i,k} = \sum_r U_{i,m?} ...$
    # Standard: $C_{i,k} = \sum_{j,r} A_{i,j} B_{j,k}$.
    # Decomposition: $A_{i,j} B_{j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # No, usually $C_{i,k} = \sum_r (\sum_j U_{i,j,r} A_{i,j}) (\sum_l V_{j,l,r} B_{j,l}) W_{k,i,r}$?
    # The interface is:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    # Equation: $C_{i,k} = \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]$?
    # No, the sum is over r. But i, j, k are fixed for a specific C element?
    # Actually, the tensor equation is:
    # $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # No, $T_{i,j,k}$ is the coefficient of $A_{i,j} B_{j,k}$ in $C_{i,k}$.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
    # In matrix mult, $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
    # So $T_{i,j,k} = 1$ for all valid i,j,k.
    # Decomposition: $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
    # Wait, W index is $k*n+i$.
    # So $W_{k,i,r}$.
    # Yes.
    
    # So for each rank r, we define vectors u_r, v_r, w_r.
    # $u_r$ is length n*m. $v_r$ is length m*p. $w_r$ is length p*n.
    
    # Let's implement the 28 terms.
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Strassen 7 terms.
    
    # 1. (A00+A11)(B00+B11)
    # U: [0,0]=1, [1,1]=1 (idx 0, 4)
    # V: [0,0]=1, [1,1]=1 (idx 0, 6)
    # W: [0,0]=1, [1,0]=1, [0,1]=1, [1,1]=1 (idx 0, 2, 1, 3) -> Wait, W[k*n+i].
    # C00 (k=0,i=0) -> W[0]
    # C01 (k=1,i=0) -> W[1]
    # C10 (k=0,i=1) -> W[2]
    # C11 (k=1,i=1) -> W[3]
    # Strassen C00 = p1 + p4 - p5 + p7
    # So p1 contributes +1 to C00, C01, C10, C11?
    # No, p1 is just one term. The recombination happens in W?
    # No, W is part of the decomposition.
    # $C_{i,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # If we use Strassen, we define 7 terms.
    # Term 1: $u_1$ has A00, A11. $v_1$ has B00, B11.
    # $w_1$ must be such that $\sum_j u_{i,j} v_{j,k} w_{k,i}$ gives the correct contribution.
    # Actually, it's easier to just set W to 1 for the target C elements and 0 elsewhere,
    # but that doesn't work for Strassen because the combination is linear.
    
    # Correct approach:
    # For each Strassen term r, the W vector has entries for the C elements that this term contributes to.
    # But Strassen terms are combined to form C.
    # So $C_{i,k} = \sum_{r \in \text{Strassen}} c_{i,k,r} P_r$.
    # And $P_r = (\sum_j U_{i,j,r} A_{i,j}) (\sum_l V_{j,l,r} B_{j,l})$.
    # So $W_{k,i,r}$ is the coefficient $c_{i,k,r}$.
    
    # Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for Term 1 (P1):
    # U: A00=1, A11=1. (U[0]=1, U[4]=1)
    # V: B00=1, B11=1. (V[0]=1, V[6]=1)
    # W: C00=1, C01=0, C10=0, C11=1. (W[0]=1, W[3]=1)
    
    add_term(0, 0, 0, 1) # Placeholder, will do manually
    
    # I will write a helper to set the column.
    
    def set_col(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    # Indices:
    # U[i*3+j]: i in 0..1, j in 0..2
    # V[j*5+k]: j in 0..2, k in 0..4
    # W[k*2+i]: k in 0..4, i in 0..1
    
    # --- Strassen Block 1 (A[0:2,0:2], B[0:2,0:2], C[0:2,0:2]) ---
    # P1
    set_col({0:1, 4:1}, {0:1, 6:1}, {0:1, 3:1}) # C00, C11
    # P2
    set_col({3:1, 4:1}, {0:1}, {2:1}) # C10
    # P3
    set_col({0:1}, {1:1, 7:-1}, {1:1}) # C01 (B01=V[1], B11=V[6]? No, B11 is V[1*5+1]=V[6]. B01 is V[0*5+1]=V[1])
    # Correction: B00=V[0], B01=V[1], B10=V[5], B11=V[6].
    # P3 = A00(B01-B11)
    set_col({0:1}, {1:1, 6:-1}, {1:1}) # C01
    # P4
    # P4 = A11(B10-B00)
    set_col({4:1}, {5:1, 0:-1}, {0:1, 2:1}) # C00, C10
    # P5
    # P5 = (A00+A01)B11
    set_col({0:1, 1:1}, {6:1}, {0:-1, 1:1}) # C00(-1), C01(1)
    # P6
    # P6 = (A10-A00)(B00+B01)
    set_col({3:1, 0:-1}, {0:1, 1:1}, {3:1}) # C11
    # P7
    # P7 = (A01-A11)(B10+B11)
    set_col({1:1, 4:-1}, {5:1, 6:1}, {0:1, 3:-1}) # C00(1), C11(-1)
    
    # --- Strassen Block 2 (A[0:2,0:2], B[0:2,2:4], C[0:2,2:4]) ---
    # B cols 2,3. B02=V[2], B03=V[3], B12=V[7], B13=V[8].
    # C cols 2,3. C02=W[4], C03=W[5], C12=W[6], C13=W[7].
    
    # P1'
    set_col({0:1, 4:1}, {2:1, 8:1}, {4:1, 7:1})
    # P2'
    set_col({3:1, 4:1}, {2:1}, {6:1})
    # P3'
    set_col({0:1}, {3:1, 8:-1}, {5:1})
    # P4'
    set_col({4:1}, {7:1, 2:-1}, {4:1, 6:1})
    # P5'
    set_col({0:1, 1:1}, {8:1}, {4:-1, 5:1})
    # P6'
    set_col({3:1, 0:-1}, {2:1, 3:1}, {7:1})
    # P7'
    set_col({1:1, 4:-1}, {7:1, 8:1}, {4:1, 7:-1})
    
    # --- Schoolbook Block 3 (A[0:2,0:2], B[0:2,4:5], C[0:2,4:5]) ---
    # B col 4. B04=V[4], B14=V[9].
    # C col 4. C04=W[8], C14=W[9].
    # 4 terms.
    # A00 B04
    set_col({0:1}, {4:1}, {8:1})
    # A01 B04
    set_col({1:1}, {4:1}, {8:1})
    # A10 B14
    set_col({3:1}, {9:1}, {9:1})
    # A11 B14
    set_col({4:1}, {9:1}, {9:1})
    
    # --- Schoolbook Block 4 (A[0:2,2:3], B[2:3,0:5], C[0:2,0:5]) ---
    # A col 2. A02=U[2], A12=U[5].
    # B row 2. B20=V[10], B21=V[11], B22=V[12], B23=V[13], B24=V[14].
    # C all.
    # 10 terms.
    # A02 B20 -> C00
    set_col({2:1}, {10:1}, {0:1})
    # A02 B21 -> C01
    set_col({2:1}, {11:1}, {1:1})
    # A02 B22 -> C02
    set_col({2:1}, {12:1}, {4:1})
    # A02 B23 -> C03
    set_col({2:1}, {13:1}, {5:1})
    # A02 B24 -> C04
    set_col({2:1}, {14:1}, {8:1})
    # A12 B20 -> C10
    set_col({5:1}, {10:1}, {2:1})
    # A12 B21 -> C11
    set_col({5:1}, {11:1}, {3:1})
    # A12 B22 -> C12
    set_col({5:1}, {12:1}, {6:1})
    # A12 B23 -> C13
    set_col({5:1}, {13:1}, {7:1})
    # A12 B24 -> C14
    set_col({5:1}, {14:1}, {9:1})
    
    return {"U": U, "V": V, "W": W}
