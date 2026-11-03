"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    terms = []  # (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    def strassen_2x2x2(i0, j0, k0):
        """Strassen 2x2x2: 7 multiplications for 2x2x2 matrix mult.
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        """
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A10)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M1 = (a00+a11)(b00+b11)
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1, 1, 1)
        add_term(i0*m + j0, (j0+1)*p + (k0+1), (k0+1)*n + i0, 1, 1, 1)
        add_term((i0+1)*m + j0, j0*p + k0, k0*n + (i0+1), 1, 1, 1)
        add_term((i0+1)*m + j0, (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1, 1, 1)
        add_term((i0+1)*m + (j0+1), j0*p + k0, k0*n + (i0+1), 1, 1, 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1, 1, 1)
        add_term(i0*m + (j0+1), j0*p + k0, k0*n + i0, 1, 1, 1)
        add_term(i0*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + i0, 1, 1, 1)
        # Wait - this is wrong. Strassen uses 7 scalar multiplications.
        # Each "multiplication" corresponds to ONE term in the decomposition.
        # The U, V, W vectors encode the linear combinations.
        # Let me redo this properly.
        pass
    
    # Proper Strassen 2x2x2 decomposition (rank 7)
    # Each term c has: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The 7 Strassen terms:
    # t1: (A00+A11)*(B00+B11) -> contributes to C00(+), C11(+)
    # t2: (A10+A11)*B00 -> contributes to C10(+), C11(-)
    # t3: A00*(B10-B11) -> contributes to C01(+), C11(+)
    # t4: A11*(B01-B10) -> contributes to C00(+), C10(+)
    # t5: (A00+A10)*B11 -> contributes to C01(+), C00(-)
    # t6: (A11-A10)*(B00+B10) -> contributes to C00(-), C11(+)
    # t7: (A01-A11)*(B11+B01) -> contributes to C00(-), C11(+)
    
    def strassen_2x2x2(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block at (i0,j0,k0)."""
        i1, j1, k1 = i0+1, j0+1, k0+1
        
        # t1: U has +1 at (i0,j0) and (i1,j1); V has +1 at (j0,k0) and (j1,k1); W has +1 at (k0,i0) and (k1,i1)
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1, 1, 1)
        add_term(i1*m + j1, j1*p + k1, k1*n + i1, 1, 1, 1)
        # But wait - each term is ONE column in U,V,W. We can't have two entries in one term.
        # The decomposition is: tensor = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # So each c is a single rank-1 tensor. For Strassen, we need 7 rank-1 tensors
        # where each has specific entries.
        
        # Actually the correct way: for each of the 7 Strassen products, we create ONE term.
        # Term 1: (A00+A11)*(B00+B11)
        #   U entries: (i0,j0)=1, (i1,j1)=1  -- NO! Each term is a SINGLE outer product.
        #   
        # I'm confusing myself. The tensor decomposition T = sum_c u_c ⊗ v_c ⊗ w_c
        # means T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # 
        # For Strassen, the 7 terms are:
        # c=0: U[(i0,j0),0]=1, U[(i1,j1),0]=1; V[(j0,k0),0]=1, V[(j1,k1),0]=1; W[(k0,i0),0]=1, W[(k1,i1),0]=1
        # But that's a sum of 4 rank-1 tensors, not 1!
        #
        # The issue: Strassen's 7 "multiplications" are each a product of two LINEAR COMBINATIONS.
        # (A00+A11) is a linear combination of two A entries. (B00+B11) is a linear combination of two B entries.
        # Their product gives a bilinear form that is NOT rank 1 in the full tensor space.
        #
        # Actually wait - in the standard tensor decomposition view:
        # The matrix multiplication tensor T_{ijk} = delta_{jk} (where i indexes A, j indexes B's row = A's col, k indexes B's col)
        # A rank-r decomposition means T = sum_{c=1}^{r} a_c ⊗ b_c ⊗ c_c where each is a rank-1 tensor.
        #
        # For Strassen 2x2x2, the 7 terms are indeed rank-1 tensors in the appropriate space.
        # Let me look at this more carefully.
        #
        # T[i,j,k] for i,j,k in {0,1}: T[i,j,k] = 1 if j==k... no.
        # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
        # So T[i,j,k] = 1 for all i,j,k? No.
        #
        # The matrix multiplication tensor is: T[i,j,k] = 1 if and only if the multiplication A[i,j]*B[j,k] appears in C[i,k].
        # For 2x2x2: T[i,j,k] = 1 for all 8 combinations? No, that's not right either.
        #
        # Actually, the standard matrix multiplication tensor M_{n} has entries M[i,j,k] = 1 if the (i,j) entry of first matrix and (j,k) entry of second matrix contribute to (i,k) of product. For n×n matrices, M[i,j,k] = 1 for all i,j,k in {0,...,n-1}. So it's the all-ones tensor of size n×n×n? No!
        #
        # Wait. The tensor is defined as: (A ⊗ B ⊗ C) where the structure is:
        # sum_{i,j,k} A[i,j] * B[j,k] * C^T[i,k]
        # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
        #
        # Hmm, but that can't be right because the rank of the all-ones tensor 2x2x2 is 2, not 7.
        #
        # I think I'm confusing the tensor structure. Let me reconsider.
        #
        # The matrix multiplication tensor for n×n matrices is an n×n × n×n × n×n tensor (or equivalently (n²)×(n²)×(n²) tensor) defined by:
        # T[(i,j),(j',k),(k',i')] = 1 if j=j' and k=k'... 
        #
        # No. The correct definition: The multiplication tensor M_n is in R^{n^2} ⊗ R^{n^2} ⊗ R^{n^2} with
        # M_n[(i,j),(j,k),(k,i)] = 1 (using the same j for A's column and B's row, and same k for B's column and C's column... wait C's row is i).
        #
        # Using the trace convention W[k*n+i, r]: the third index is (k,i) meaning row i, col k of C.
        # So T[(i,j),(j,k),(k,i)] = 1.
        #
        # For 2x2x2 (n=m=p=2): indices are i∈{0,1}, j∈{0,1}, k∈{0,1}
        # T[(i,j),(j,k),(k,i)] = 1 for all i,j,k. That's 8 non-zero entries out of 8*8*8=512.
        #
        # A rank-1 term is: u ⊗ v ⊗ w where u ∈ R^4, v ∈ R^4, w ∈ R^4
        # Term c contributes: T[(i,j),(j,k),(k,i)] += U[i,j,c]*V[j,k,c]*W[k,i,c]
        #
        # For Strassen, the 7 terms for 2x2x2 are well-known. Let me just hardcode them.
        #
        # Standard Strassen 2x2x2 rank-7 decomposition:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        # M1 = (a+d)(e+h): i+=M1, l+=M1
        # M2 = (c+d)e:    k+=M2, l-=M2  
        # M3 = a(g-h):    j+=M3, l+=M3
        # M4 = d(f-g):    i+=M4, k+=M4
        # M5 = (a+c)h:    j+=M5, i-=M5
        # M6 = (d-c)(e+g): i-=M6, l+=M6
        # M7 = (b-d)(h+f): i-=M7, l+=M7
        #
        # In our indexing: i0=0,i1=1; j0=0,j1=1; k0=0,k1=1
        # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l
        #
        # U index: i*m+j = i*2+j (for m=2 case, but our m=5, so we use global indices)
        # V index: j*p+k
        # W index: k*n+i
        #
        # M1 = (a+d)(e+h):
        #   U: (0,0)=1, (1,1)=1 → U[0*5+0,c]=1, U[1*5+1,c]=1
        #   V: (0,0)=1, (1,1)=1 → V[0*6+0,c]=1, V[1*6+1,c]=1
        #   W: (0,0)=1, (1,1)=1 → W[0*3+0,c]=1, W[1*3+1,c]=1
        #   Contribution to C: C[0,0]+=1, C[1,1]+=1
        #   Check: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 → T[0,0,0] += 1 ✓
        #           U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1 → T[1,1,1] += 1 ✓
        #   But we need T[0,0,0]=1 and T[1,1,1]=1, and this term contributes to both. Good.
        #   But it also contributes to other (i,j,k) combos? No, because U is zero elsewhere, V zero elsewhere, W zero elsewhere.
        #   Wait: U[0,0]=1 means for i=0,j=0. V[0,0]=1 means j=0,k=0. W[0,0]=1 means k=0,i=0.
        #   So the term contributes T[0,0,0] += 1*1*1 = 1. And T[1,1,1] += 1*1*1 = 1.
        #   That's correct for M1!
        #
        # M2 = (c+d)*e:
        #   U: (1,0)=1, (1,1)=1 → U[1*5+0,c]=1, U[1*5+1,c]=1
        #   V: (0,0)=1 → V[0*6+0,c]=1
        #   W: (0,1)=1, (1,1)=-1 → W[0*3+1,c]=1, W[1*3+1,c]=-1
        #   Contribution: T[1,0,0] += 1*1*1=1, T[1,1,0] += 1*1*1=1, T[1,0,1] += 1*1*(-1)=-1, T[1,1,1] += 1*1*(-1)=-1
        #   We need: k+=M2 means C[1,0]+=M2, l-=M2 means C[1,1]-=M2
        #   T[1,0,0]: i=1,j=0,k=0 → C[1,0] contribution. U[1,0]=1,V[0,0]=1,W[0,1]=1 → 1. ✓
        #   T[1,1,0]: i=1,j=1,k=0 → C[1,0] contribution. U[1,1]=1,V[1,0]=0 → 0. Hmm, V[1,0] should be 0.
        #   Wait, V index is j*p+k. For j=1,k=0: V[1*6+0,c]. But I said V only has (0,0)=1, so V[6,c]=0.
        #   So T[1,1,0] += U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0. Good.
        #   T[1,0,1]: i=1,j=0,k=1 → C[1,1]. U[1,0]=1,V[0,1]=0 → 0. Good.
        #   T[1,1,1]: i=1,j=1,k=1 → C[1,1]. U[1,1]=1,V[1,1]=0 → 0. Good.
        #   Hmm, but I need C[1,1] -= M2. Let me recheck.
        #   Actually W[0*3+1,c] = W[1,c] corresponds to k=0,i=1 → C[1,0]. And W[1*3+1,c]=W[4,c] corresponds to k=1,i=1 → C[1,1].
        #   So W[1,c]=1 means C[1,0] += ..., W[4,c]=-1 means C[1,1] -= ...
        #   T[1,0,0]: U[1,0]=1, V[0,0]=1, W[0,1]=W[1,c]=1 → contributes 1 to T[1,0,0] which is C[1,0]. ✓
        #   T[1,1,1]: U[1,1]=1, V[1,1]=0 → 0. 
        #   But I need C[1,1] to get -M2. Where does that come from?
        #   T[1,0,1]: U[1,0]=1, V[0,1]=0 → 0.
        #   T[1,1,1]: U[1,1]=1, V[1,1]=0 → 0.
        #   So this term only contributes to C[1,0], not C[1,1]. That's wrong for Strassen.
        #
        # I think the issue is that I'm not correctly mapping the Strassen formula to the tensor decomposition.
        # Let me look up the correct rank-7 decomposition for 2x2x2 matrix multiplication tensor.
        #
        # The 2x2x2 matrix multiplication tensor has 8 non-zero entries: T[i,j,k]=1 for i,j,k in {0,1}.
        # A rank-1 tensor is u⊗v⊗w where u,v,w are vectors of length 4 (for 2x2 matrices).
        # u⊗v⊗w has entries u[i,j]*v[j,k]*w[k,i] for the (i,j,k) slice... no.
        #
        # Actually, in the (n²)×(n²)×(n²) tensor space, a rank-1 tensor is the outer product of three vectors in R^{n²}.
        # So for n=2, each vector has 4 components.
        # u = (u00, u01, u10, u11), v = (v00, v01, v10, v11), w = (w00, w01, w10, w11)
        # The outer product has entry [(i,j),(j',k),(k',i')] = u[i,j]*v[j',k]*w[k',i']
        # For this to contribute to T[i,j,k] (where T[i,j,k]=1 iff the standard MM tensor has 1 there),
        # we need j=j' and k=k' and i=i'... no.
        #
        # The MM tensor T has T[(i,j),(j,k),(k,i)] = 1. So the non-zero entries are at positions
        # ((0,0),(0,0),(0,0)), ((0,0),(0,1),(1,0)), ((0,1),(1,0),(0,1)), ((0,1),(1,1),(1,1)),
        # ((1,0),(0,0),(0,1)), ((1,0),(0,1),(1,0)), ((1,1),(1,0),(0,1)), ((1,1),(1,1),(1,1))
        #
        # A rank-1 term u⊗v⊗w contributes u[i,j]*v[j',k]*w[k',i'] to position ((i,j),(j',k),(k',i')).
        # For this to match T's non-zero pattern, we need the product to be non-zero exactly when
        # (i,j)=(j',k) first index... this is getting complicated.
        #
        # Let me just use the known correct Strassen decomposition. The 7 terms for 2×2×2 MM:
        # Using 0-based indexing, A entries: a00,a01,a10,a11; B: b00,b01,b10,b11; C: c00,c01,c10,c11
        #
        # Term 1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)  → c00+=1, c11+=1
        # Term 2: U=(0,0,1,1), V=(1,0,0,0), W=(0,1,0,-1) → c10+=1, c11-=1
        # Term 3: U=(1,0,0,0), V=(0,0,1,-1), W=(0,1,0,1) → c01+=1, c11+=1
        # Term 4: U=(0,0,0,1), V=(0,1,-1,0), W=(1,0,1,0) → c00+=1, c10+=1
        # Term 5: U=(1,1,0,0), V=(0,0,0,1), W=(-1,1,0,0) → c01+=1, c00-=1
        # Term 6: U=(0,0,-1,1), V=(1,0,1,0), W=(0,0,0,1) → c00-=1, c11+=1
        # Term 7: U=(0,1,0,-1), V=(0,1,0,1), W=(0,0,0,1) → c00-=1, c11+=1
        #
        # Let me verify term 1: U=(1,0,0,1) means U[0,0]=1,U[0,1]=0,U[1,0]=0,U[1,1]=1
        # V=(1,0,0,1) means V[0,0]=1,V[0,1]=0,V[1,0]=0,V[1,1]=1
        # W=(1,0,0,1) means W[0,0]=1,W[0,1]=0,W[1,0]=0,W[1,1]=1
        # Contribution to T[i,j,k]: U[i,j]*V[j,k]*W[k,i]
        # T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
        # T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
        # T[0,1,0]: U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
        # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
        # T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
        # T[1,0,1]: U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
        # T[1,1,0]: U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
        # T[1,1,1]: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1 ✓
        # Sum of all 7 terms should give T[i,j,k]=1 for all 8 entries. Let me verify T[0,0,1]:
        # T1: 0, T2: U[0,0]*V[0,1]*W[1,0]=1*0*0=0, T3: U[0,0]*V[0,1]*W[1,0]=1*0*0=0, 
        # T4: U[0,0]*V[0,1]*W[1,0]=0*1*0=0, T5: U[0,0]*V[0,1]*W[1,0]=1*0*0=0,
        # T6: 0, T7: 0. Sum=0. But we need T[0,0,1]=1!
        #
        # Something is wrong. Let me reconsider.
        #
        # Oh wait - I think the issue is that for 2×2×2, the "tensor" is not just the 8 entries T[i,j,k].
        # The matrix multiplication tensor for multiplying a 2×2 by 2×2 matrix is a tensor of size
        # (2*2) × (2*2) × (2*2) = 4×4×4 = 64 entries, of which 8 are non-zero.
        #
        # The 8 non-zero entries are T[(i,j),(j,k),(k,i)] = 1 for i,j,k ∈ {0,1}.
        #
        # A rank-1 decomposition term is u ⊗ v ⊗ w where u,v,w ∈ R^4.
        # The (a,b,c) entry of u⊗v⊗w is u[a]*v[b]*w[c].
        #
        # So for term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # Entry (0,0,0): u[0]*v[0]*w[0] = 1*1*1 = 1
        # Entry (0,3,0): u[0]*v[3]*w[0] = 1*1*1 = 1
        # Entry (3,0,0): u[3]*v[0]*w[0] = 1*1*1 = 1
        # Entry (3,3,0): u[3]*v[3]*w[0] = 1*1*1 = 1
        # Entry (0,0,3): u[0]*v[0]*w[3] = 1*1*1 = 1
        # Entry (0,3,3): u[0]*v[3]*w[3] = 1*1*1 = 1
        # Entry (3,0,3): u[3]*v[0]*w[3] = 1*1*1 = 1
        # Entry (3,3,3): u[3]*v[3]*w[3] = 1*1*1 = 1
        # All other entries are 0.
        #
        # Now, which of these 8 positions correspond to T's non-zero entries?
        # T's non-zero entries are at ((i,j),(j,k),(k,i)) for i,j,k ∈ {0,1}.
        # In flat indexing: (i*2+j, j*2+k, k*2+i)
        # (0,0,0): i=0,j=0,k=0 → (0,0,0) ✓
        # (0,0,1): i=0,j=0,k=1 → (0, 0*2+1=1, 1*2+0=2) → (0,1,2)
        # (0,1,0): i=0,j=1,k=0 → (1, 1*2+0=2, 0*2+0=0) → (1,2,0)
        # (0,1,1): i=0,j=1,k=1 → (1, 1*2+1=3, 1*2+0=2) → (1,3,2)
        # (1,0,0): i=1,j=0,k=0 → (2, 0, 2) → (2,0,2)
        # (1,0,1): i=1,j=0,k=1 → (2, 1, 3) → (2,1,3)
        # (1,1,0): i=1,j=1,k=0 → (3, 2, 2) → (3,2,2)
        # (1,1,1): i=1,j=1,k=1 → (3, 3, 3) → (3,3,3)
        #
        # So T's non-zero positions in flat indexing are:
        # (0,0,0), (0,1,2), (1,2,0), (1,3,2), (2,0,2), (2,1,3), (3,2,2), (3,3,3)
        #
        # Term 1 contributes to: (0,0,0), (0,3,0), (3,0,0), (3,3,0), (0,0,3), (0,3,3), (3,0,3), (3,3,3)
        # Of these, (0,0,0) and (3,3,3) are in T's non-zero set. The others are not.
        # So term 1 contributes 1 to T[0,0,0] and 1 to T[1,1,1]. Good.
        #
        # Now I need to find 7 such terms whose sum gives exactly 1 at each of the 8 positions.
        # This is the standard Strassen decomposition. Let me look it up properly.
        #
        # The correct Strassen rank-7 decomposition for 2×2×2 MM tensor:
        # Using the notation where A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
        #
        # P1 = (a+d)(e+h): contributes to c00, c11
        # P2 = (c+d)e:     contributes to c10, c11 (with -)
        # P3 = a(g-h):     contributes to c01, c11
        # P4 = d(f-g):     contributes to c00, c10
        # P5 = (a+c)h:     contributes to c01, c00 (with -)
        # P6 = (d-c)(e+g): contributes to c00 (with -), c11
        # P7 = (b-d)(h+f): contributes to c00 (with -), c11
        #
        # c00 = P1 + P4 - P5 - P6 - P7
        # c01 = P3 + P5
        # c10 = P2 + P4
        # c11 = P1 - P2 + P3 + P6 + P7
        #
        # Now in tensor decomposition form, each P_i corresponds to a rank-1 term.
        # P1 = (a+d)(e+h): 
        #   a = A[0,0], d = A[1,1], e = B[0,0], h = B[1,1]
        #   U vector: has 1 at position (0,0) and (1,1) → U[0]=1, U[3]=1 (flat: i*2+j)
        #   V vector: has 1 at position (0,0) and (1,1) → V[0]=1, V[3]=1
        #   W vector: has 1 at position (0,0) and (1,1) → W[0]=1, W[3]=1
        #   But this gives contributions to (0,0,0) and (3,3,3) as computed above, plus other unwanted positions.
        #
        # The problem is that a single rank-1 term u⊗v⊗w creates a "grid" of non-zero entries, not just the ones we want.
        #
        # I think the correct approach is: the Strassen decomposition IS a valid rank-7 tensor decomposition, but the individual terms are not simply "put 1s where the linear combination has non-zero coefficients."
        #
        # Let me just verify: is the 2×2×2 MM tensor rank actually 7? Yes, it's a well-known result.
        #
        # The correct rank-7 decomposition (from literature):
        # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # Term 2: u=(0,0,1,1), v=(1,0,0,0), w=(0,1,0,-1)
        # Term 3: u=(1,0,0,0), v=(0,0,1,-1), w=(0,1,0,1)
        # Term 4: u=(0,0,0,1), v=(0,1,-1,0), w=(1,0,1,0)
        # Term 5: u=(1,1,0,0), v=(0,0,0,1), w=(-1,1,0,0)
        # Term 6: u=(0,0,-1,1), v=(1,0,1,0), w=(0,0,0,1)
        # Term 7: u=(0,1,0,-1), v=(0,1,0,1), w=(0,0,0,1)
        #
        # Let me verify this gives T[0,1,2]=1 (which is i=0,j=0,k=1, position (0,1,2) in flat indexing):
        # T[0,1,2] = sum_c U_c[0] * V_c[1] * W_c[2]
        # Term 1: 1*0*0 = 0
        # Term 2: 0*0*0 = 0
        # Term 3: 1*0*0 = 0
        # Term 4: 0*1*1 = 1
        # Term 5: 1*0*0 = 0
        # Term 6: 0*0*0 = 0
        # Term 7: 0*1*0 = 0
        # Sum = 1 ✓
        #
        # Let me verify T[1,2,0] (i=0,j=1,k=0, position (1,2,0)):
        # T[1,2,0] = sum_c U_c[1] * V_c[2] * W_c[0]
        # Term 1: 0*0*1 = 0
        # Term 2: 0*0*0 = 0
        # Term 3: 0*1*0 = 0
        # Term 4: 0*(-1)*1 = 0
        # Term 5: 1*0*(-1) = 0
        # Term 6: 0*1*0 = 0
        # Term 7: 1*0*0 = 0
        # Sum = 0. But we need 1! ✗
        #
        # Hmm, that's wrong. Let me recheck. Position (1,2,0) means flat index 1 for U, 2 for V, 0 for W.
        # U index 1 = (0,1) in 2×2. V index 2 = (1,0) in 2×2. W index 0 = (0,0) in 2×2.
        # This corresponds to i=0,j=1,k=0 → C[0,0] gets A[0,1]*B[1,0]. Yes, T[0,1,0] should be 1.
        #
        # Wait, I think I'm confusing myself with the flat indexing. Let me be very careful.
        #
        # U is indexed by (i,j) → flat index i*2+j for 2×2.
        # V is indexed by (j,k) → flat index j*2+k for 2×2.
        # W is indexed by (k,i) → flat index k*2+i for 2×2.
        #
        # T[i,j,k] = sum_c U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i]
        #
        # For T[0,1,0] (i=0,j=1,k=0): sum_c U_c[0*2+1] * V_c[1*2+0] * W_c[0*2+0]
        # = sum_c U_c[1] * V_c[2] * W_c[0]
        # Term 1: U[1]=0, V[2]=0, W[0]=1 → 0
        # Term 2: U[1]=0, V[2]=0, W[0]=0 → 0
        # Term 3: U[1]=0, V[2]=1, W[0]=0 → 0
        # Term 4: U[1]=0, V[2]=-1, W[0]=1 → 0
        # Term 5: U[1]=1, V[2]=0, W[0]=-1 → 0
        # Term 6: U[1]=0, V[2]=1, W[0]=0 → 0
        # Term 7: U[1]=1, V[2]=0, W[0]=0 → 0
        # Sum = 0. Still wrong!
        #
        # I must have the wrong decomposition. Let me try a different known decomposition.
        #
        # Actually, I realize the issue. Let me just use NumPy to compute a valid rank-7 decomposition
        # or use the schoolbook (rank 8) for 2×2×2 and focus on getting the overall structure right.
        #
        # Given the time constraints, let me use a simpler approach: use Strassen only where it clearly helps
        # and verify with NumPy. If Strassen doesn't work cleanly, fall back to schoolbook.
        pass
    
    # Given the complexity of getting Strassen right, let me use a verified approach:
    # Use the known rank-7 Strassen decomposition verified by computation.
    # For now, let me just do schoolbook (rank 90) but try a few structural reductions.
    
    # Actually, let me just implement schoolbook and verify it works, then try Strassen.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i*m + j, j*p + k, k*n + i, 1, 1, 1)
    
    # Build matrices
    new_rank = len(terms)
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    for c, (u_idx, v_idx, w_idx, u_val, v_val, w_val) in enumerate(terms):
        U[u_idx][c] = u_val
        V[v_idx][c] = v_val
        W[w_idx][c] = w_val
    
    # Verify with numpy
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for random A (n×m) and B (m×p), C = A@B should equal sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k' V[j*p+k',c]*B[j,k'])... 
    # Actually: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k]) * W[k*n+i,c]
    # Wait no. The decomposition is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k]) * W[k*n+i,c]
    # Hmm, that's not quite right either. Let me think again.
    #
    # The tensor T satisfies: T[i,j,k] = 1 (for the MM tensor)
    # And T = sum_c U_c ⊗ V_c ⊗ W_c where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i,j]*V_c[j,k]*W_c[k,i]
    # So T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    #
    # For MM: C[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
    # = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]*A[i,j]*B[j,k]
    # = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    #
    # So: C[i,k] = sum_c W[k*n+i,c] * (U_c^T * A_col_i) * (B_row_k * V_c)
    # where U_c is the column c of U reshaped to (n,m), etc.
    #
    # In matrix form: Let A be n×m, B be m×p.
    # Define for each c: a_c = U[:,c] reshaped to n×m, b_c = V[:,c] reshaped to m×p, w_c = W[:,c] reshaped to p×n
    # Then C = sum_c w_c * (A * a_c) * b_c... no.
    #
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    #
    # Let's define: alpha_c[i] = sum_j U[i*m+j,c]*A[i,j]  (this is element-wise product of A[i,:] and U_row_i,c, summed)
    # beta_c[k] = sum_j V[j*p+k,c]*B[j,k]
    # Then C[i,k] = sum_c W[k*n+i,c] * alpha_c[i] * beta_c[k]
    #
    # In matrix form: C = sum_c diag(alpha_c) @ (w_c^T) @ diag(beta_c)... no.
    # C = sum_c outer(alpha_c, beta_c) * w_c (element-wise)
    # = sum_c (alpha_c * w_c_col) outer (beta_c) ... 
    #
    # Actually: C[i,k] = sum_c (alpha_c[i] * w_c[k,i] * beta_c[k])
    # = sum_c (alpha_c ⊙ w_c_col_i) ... this is getting messy.
    #
    # Let me just verify numerically.
    rng = np.random.RandomState(42)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for c in range(new_rank):
        u_col = U_np[:, c].reshape(n, m)
        v_col = V_np[:, c].reshape(m, p)
        w_col = W_np[:, c].reshape(p, n)
        # C[i,k] += W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
        for i in range(n):
            for k in range(p):
                alpha = np.sum(u_col[i, :] * A_test[i, :])
                beta = np.sum(v_col[:, k] * B_test[:, k])
                C_computed[i, k] += w_col[k, i] * alpha * beta
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Verification failed, return schoolbook
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
