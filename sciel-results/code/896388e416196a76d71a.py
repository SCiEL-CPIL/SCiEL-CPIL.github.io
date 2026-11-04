import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: 2x2 x 2x6 using 3 Strassen blocks
    # A part: rows 0,1 cols 0,1
    # B part: rows 0,1 cols 0..5
    # Output: rows 0,1 cols 0..5
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output combinations for C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate over the 3 column blocks of B (size 2 each)
    for q in range(3):
        # k indices for this block: 2q, 2q+1
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # Define the 7 multiplications
        # Each multiplication is a sum of terms U_val * V_val
        # We need to add to U, V, W for each term
        
        # M1
        U[0*3+0, col] += 1; V[0*6+k0, col] += 1
        U[1*3+1, col] += 1; V[1*6+k1, col] += 1
        # Wait, M1 = (A00+A11)(B00+B11)
        # The term is (A00+A11) * (B00+B11).
        # This is a product of two sums.
        # In tensor decomposition, a product of sums is not a single rank-1 term.
        # Strassen works because we compute the sums first?
        # No, the tensor decomposition represents the bilinear form.
        # The rank-1 term is $u \otimes v \otimes w$.
        # The multiplication $M_1$ is not a single scalar multiplication in the final sum, it is an intermediate value.
        # However, in the context of the tensor decomposition of the whole matrix multiplication,
        # we are looking for $C_{ik} = \sum_{c} U_{i,m?} V_{j,p?} W_{k,n?}$.
        # The Strassen algorithm reduces the number of *scalar multiplications* if we view the blocks as scalars.
        # But here the "scalars" are the entries.
        # Does the rank add up?
        # The rank of the $2 \times 2 \times 2$ tensor is 7.
        # This means we can represent the multiplication of two $2 \times 2$ matrices with 7 rank-1 tensors.
        # So for each $2 \times 2 \times 2$ block, we add 7 columns to U, V, W.
        
        # Let's define the 7 rank-1 tensors for $2 \times 2 \times 2$.
        # Indices: i,j in {0,1}, k in {0,1}
        # Tensors $t_1, \dots, t_7$.
        
        # t1: A00+A11, B00+B11 -> C00, C01, C10, C11 contributions
        # Actually, it's easier to just hardcode the 7 tensors for 2x2x2.
        
        # Standard Strassen Tensors for 2x2x2:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        # i = m1 + m4 - m5 + m7
        # j = m3 + m5
        # k = m2 + m4
        # l = m1 - m2 + m3 + m6
        
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(g-h)
        # m4 = d(f-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # Each ml is a product of two linear forms.
        # In the tensor decomposition, each ml corresponds to ONE rank-1 term?
        # No, ml is a scalar. The term in the sum for C is $1 \cdot ml$.
        # But ml itself is a product of sums.
        # Wait, the tensor decomposition is for the BILINEAR map $C = A B$.
        # The rank is the minimum number of terms in $\sum U_c \otimes V_c \otimes W_c$.
        # If we use Strassen, we are effectively using the fact that the tensor $M_{2,2,2}$ has rank 7.
        # So yes, we add 7 columns for each $2 \times 2 \times 2$ block.
        
        # Let's define the coefficients for the 7 columns.
        # For block q:
        # A indices: (0,0), (0,1), (1,0), (1,1) -> U rows 0,1,3,4
        # B indices: (0,k0), (0,k1), (1,k0), (1,k1) -> V rows k0, k1, 6+k0, 6+k1
        # C indices: (0,k0), (0,k1), (1,k0), (1,k1) -> W rows 2*k0, 2*k0+1, 2*k1, 2*k1+1
        
        # m1
        U[0, col] += 1; V[k0, col] += 1
        U[4, col] += 1; V[6+k1, col] += 1
        # Wait, m1 = (A00+A11)(B00+B11).
        # This is NOT a rank 1 term in the original variables.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is a sum of 4 rank-1 terms.
        # So m1 alone does not correspond to 1 column in the final U,V,W.
        # 
        # RE-EVALUATION:
        # The rank of the tensor $M_{2,2,2}$ is 7.
        # This means there exist 7 vectors $u_c, v_c, w_c$ such that
        # $C_{ik} = \sum_{c=1}^7 u_{c,i} v_{c,j} w_{c,k}$.
        # The standard Strassen decomposition provides these 7 terms.
        # Let's list them explicitly for $2 \times 2 \times 2$.
        
        # The 7 rank-1 tensors are:
        # 1. (1,1,0,0) for A, (1,0,0,1) for B, (1,0,0,1) for C? No.
        # Let's use the known decomposition.
        
        # T1: A00+A11, B00+B11. 
        #    U: [1,0,0,1] (for A00,A01,A10,A11) -> wait, A is 2x2.
        #    Let's flatten A: A00, A01, A10, A11.
        #    U1 = [1, 0, 0, 1]
        #    V1 = [1, 0, 0, 1] (for B00, B01, B10, B11)
        #    W1 = [1, 0, 0, 1] (for C00, C01, C10, C11)
        #    Does this work?
        #    Sum = (A00+A11)(B00+B11)(C00+C11).
        #    This contributes to C00, C01, C10, C11.
        #    C00 gets 1, C01 gets 1, C10 gets 1, C11 gets 1.
        #    This is part of m1.
        
        # Actually, it is easier to just implement the Strassen decomposition directly by adding the 7 terms.
        
        # Term 1:
        # A: A00+A11 -> U[0]+=1, U[4]+=1
        # B: B00+B11 -> V[k0]+=1, V[6+k1]+=1
        # C: C00+C11? No, the output of m1 is added to C00 and C11 with +1, and to others?
        # m1 contributes +1 to C00 and +1 to C11.
        # So W[2*k0]+=1, W[2*k1+1]+=1.
        
        # Let's verify the Strassen formulas again.
        # i (C00) = m1 + m4 - m5 + m7
        # j (C01) = m3 + m5
        # k (C10) = m2 + m4
        # l (C11) = m1 - m2 + m3 + m6
        
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(g-h)
        # m4 = d(f-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # We add 7 columns.
        
        # Col 0 (m1):
        # U: a+d -> U[0]=1, U[4]=1
        # V: e+h -> V[k0]=1, V[6+k1]=1
        # W: i+l -> W[2*k0]=1, W[2*k1+1]=1
        
        # Col 1 (m2):
        # U: c+d -> U[3]=1, U[4]=1
        # V: e -> V[k0]=1
        # W: k -> W[2*k1]=1
        
        # Col 2 (m3):
        # U: a -> U[0]=1
        # V: g-h -> V[6+k0]=1, V[6+k1]=-1
        # W: j -> W[2*k0+1]=1
        
        # Col 3 (m4):
        # U: d -> U[4]=1
        # V: f-e -> V[k1]=1, V[k0]=-1
        # W: i+k -> W[2*k0]=1, W[2*k1]=1
        
        # Col 4 (m5):
        # U: a+b -> U[0]=1, U[1]=1
        # V: h -> V[6+k1]=1
        # W: j - i -> W[2*k0+1]=1, W[2*k0]=-1
        
        # Col 5 (m6):
        # U: c-a -> U[3]=1, U[0]=-1
        # V: e+f -> V[k0]=1, V[k1]=1
        # W: l -> W[2*k1+1]=1
        
        # Col 6 (m7):
        # U: b-d -> U[1]=1, U[4]=-1
        # V: g+h -> V[6+k0]=1, V[6+k1]=1
        # W: i - j -> W[2*k0]=1, W[2*k0+1]=-1
        
        # Let's apply these to the actual indices.
        
        # m1
        U[0, col] += 1; U[4, col] += 1
        V[k0, col] += 1; V[6+k1, col] += 1
        W[2*k0, col] += 1; W[2*k1+1, col] += 1
        col += 1
        
        # m2
        U[3, col] += 1; U[4, col] += 1
        V[k0, col] += 1
        W[2*k1, col] += 1
        col += 1
        
        # m3
        U[0, col] += 1
        V[6+k0, col] += 1; V[6+k1, col] -= 1
        W[2*k0+1, col] += 1
        col += 1
        
        # m4
        U[4, col] += 1
        V[k1, col] += 1; V[k0, col] -= 1
        W[2*k0, col] += 1; W[2*k1, col] += 1
        col += 1
        
        # m5
        U[0, col] += 1; U[1, col] += 1
        V[6+k1, col] += 1
        W[2*k0+1, col] += 1; W[2*k0, col] -= 1
        col += 1
        
        # m6
        U[3, col] += 1; U[0, col] -= 1
        V[k0, col] += 1; V[k1, col] += 1
        W[2*k1+1, col] += 1
        col += 1
        
        # m7
        U[1, col] += 1; U[4, col] -= 1
        V[6+k0, col] += 1; V[6+k1, col] += 1
        W[2*k0, col] += 1; W[2*k0+1, col] -= 1
        col += 1

    # Part 2: 2x1 x 1x6
    # A part: rows 0,1 col 2
    # B part: row 2 cols 0..5
    # Output: rows 0,1 cols 0..5
    
    for i in range(2):
        for k in range(6):
            U[i*3+2, col] += 1
            V[2*6+k, col] += 1
            W[k*2+i, col] += 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
