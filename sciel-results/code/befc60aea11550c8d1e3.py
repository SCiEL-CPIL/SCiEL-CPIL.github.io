import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Attempt rank 24 construction
    # Split p=6 into 6 single columns. For each k in 0..5, compute C(:,k) = A * B(:,k).
    # C(:,k) is a 2x1 vector: C(0,k) = sum_j A(0,j)*B(j,k), C(1,k) = sum_j A(1,j)*B(j,k)
    # For each (j,k), we need to capture A(0,j)*B(j,k) in C(0,k) and A(1,j)*B(j,k) in C(1,k).
    # Use 2 terms per (j,k): one for i=0, one for i=1.
    # Term (j,k,0): U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1 -> contributes A(0,j)*B(j,k) to C(0,k)
    # Term (j,k,1): U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1 -> contributes A(1,j)*B(j,k) to C(1,k)
    # Total: 4 * 6 * 2 = 48. That's schoolbook, not 24.
    
    # Alternative: Can we do 1 term per (j,k)?
    # If U[0*m+j]=1, U[1*m+j]=1, V[j*p+k]=1, W[k*n+0]=a, W[k*n+1]=b
    # Then C(0,k) += a*A(0,j)*B(j,k) + a*A(1,j)*B(j,k) = a*(A(0,j)+A(1,j))*B(j,k)
    # C(1,k) += b*A(0,j)*B(j,k) + b*A(1,j)*B(j,k) = b*(A(0,j)+A(1,j))*B(j,k)
    # This forces C(0,k)/a = C(1,k)/b, i.e., C(0,k)*b = C(1,k)*a.
    # For general A,B, this is not true. So 1 term per (j,k) doesn't work.
    
    # What if we use a different splitting? Split m=4 into 4 single columns.
    # For each j in 0..3, k in 0..5: C(:,k) += A(:,j)*B(j,k)
    # A(:,j) is 2x1, B(j,k) is scalar.
    # C(0,k) += A(0,j)*B(j,k), C(1,k) += A(1,j)*B(j,k)
    # This is rank 1 in the (i,k) space for each (j,k)? No, it's a vector (A(0,j), A(1,j))*B(j,k).
    # So for each (j,k), we need a term where U is (1,1) at column j, V is 1 at (j,k), and W is (1,1) at column k.
    # But that gives C(0,k) += (1)*1*(1) = 1 and C(1,k) += (1)*1*(1) = 1, not A(0,j)*B(j,k) and A(1,j)*B(j,k).
    
    # The issue is that U and W are fixed vectors, they can't depend on the actual values of A and B.
    # In tensor decomposition, U, V, W are constant matrices. The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r].
    # For matrix multiplication, T[i,j,k] = 1 if the indices are consistent, else 0.
    # Specifically, T[i,j,k] = 1 for all valid (i,j,k) in the multiplication, and the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # And = 0 otherwise. But in matrix multiplication, ALL (i,j,k) triples are "valid" in the sense that
    # they all contribute to some C entry. Wait, no.
    
    # Let me re-read the tensor definition. The matrix multiplication tensor has entries:
    # T[i, j, k] where i indexes rows of A (0..n-1), j indexes cols of A / rows of B (0..m-1), 
    # k indexes cols of B (0..p-1).
    # The tensor satisfies: C[i,k] = sum_j A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k] * 1
    # So the tensor is T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    # Actually, the matrix multiplication tensor M_{n,m,p} is defined such that:
    # <M, A ⊗ B ⊗ C^T> = trace(C^T * A * B) = sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
    # So M[i,j,k] = 1 for all i,j,k. The tensor is the all-ones tensor!
    
    # Wait, that can't be right. Let me reconsider.
    # The bilinear form is: sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] * C[k,i]
    # We want this to equal sum_{i,k} C[i,k] * (A*B)[i,k] = sum_{i,k} C[i,k] * sum_j A[i,j]*B[j,k]
    # = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
    # So M[i,j,k] * C[k,i] should equal C[i,k] when summed.
    # This means M[i,j,k] = 1 for all i,j,k, and the C index is k*n+i.
    
    # So the tensor is indeed all-ones! T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # Then the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # And for "invalid" triples... but there are no invalid triples. All (i,j,k) are valid.
    
    # Wait, but the verification code checks that for "zero" entries, the sum is 0.
    # Let me re-read the verification:
    # for a in range(nm): for b in range(mp): for c in range(pn):
    #   i = a // m; j = a % m
    #   j_b = b // p; k = b % p
    #   i_w = c % n; k_w = c // n
    #   if i == i_w and j == j_b and k == k_w: continue  # This is the "one" entry
    #   else: check sum = 0
    
    # So the tensor has 1s where i==i_w and j==j_b and k==k_w, i.e., where the indices are consistent.
    # But for matrix multiplication, ALL (i,j,k) triples contribute. There's no "inconsistent" triple.
    
    # I think I'm confusing myself. Let me look at the actual tensor structure.
    # The matrix multiplication tensor for 2x4x6 has 2*4*6 = 48 entries, all equal to 1.
    # The decomposition must satisfy: for each (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i',
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if j==j' and i==i' and k==k', else 0.
    
    # Wait, that doesn't match matrix multiplication. In matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k].
    # The bilinear form is sum_{i,j,k} A[i,j]*B[j,k]*C[k,i] (with trace convention C[k*n+i]).
    # = sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
    # We want this to equal sum_{i,k} C[i,k]*(A*B)[i,k] = sum_{i,k} C[i,k]*sum_j A[i,j]*B[j,k]
    # = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
    
    # So we need C[k,i] = C[i,k] for the equality to hold? No, that's not right.
    # The issue is the index convention. With W[k*n+i], the C matrix is stored as W[k*n+i] = C[i,k].
    # So the bilinear form is sum_{i,j,k} A[i,j]*B[j,k]*W[k*n+i] = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k].
    # And (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # So sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] = sum_{i,k} C[i,k]*sum_j A[i,j]*B[j,k] = sum_{i,k} C[i,k]*(A*B)[i,k].
    
    # So the tensor T[i,j,k] = 1 for all i,j,k. There are no "zero" entries in the tensor itself.
    # But the verification code checks for zeros when indices are "inconsistent".
    
    # I think the verification is checking that the decomposition only produces non-zero values
    # when the indices are consistent with the matrix multiplication structure.
    # But since the tensor is all-ones, all entries should be 1.
    
    # Let me re-examine. The verification says:
    # if i == i_w and j == j_b and k == k_w: this is the "one" case
    # else: this should be zero
    
    # But for the all-ones tensor, ALL cases should be 1, not just the consistent ones.
    # Unless the tensor is NOT all-ones.
    
    # I think I've been wrong. Let me reconsider the tensor definition.
    # The matrix multiplication tensor is NOT the all-ones tensor. It's a specific tensor
    # that encodes the multiplication algorithm.
    
    # Actually, I believe the tensor T has T[i,j,k] = 1 for all i,j,k, and the "structure"
    # comes from how the indices are mapped. The verification is checking that the decomposition
    # correctly produces 1 for all valid (i,j,k) and 0 for... wait, there are no invalid (i,j,k).
    
    # I'm going in circles. Let me just focus on the practical aspect: the current rank-42 solution
    # works, and I need to try to improve it. The rank-24 attempt in the code is likely failing
    # because the construction is incorrect.
    
    # Given the time constraints, let me keep the rank-42 solution and try a different approach
    # to reduce the rank: use a more efficient block decomposition or find linear dependencies.
    
    # Actually, let me try a completely different approach: use the fact that we can decompose
    # 2x4x6 as a combination of smaller multiplications with shared terms.
    
    # For now, let me keep the existing rank-42 code and add a verification-based rank reduction.
    
    rank = 42
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Strassen 2x2x2 decomposition pattern
    # Inputs:
    #   A indices: a00, a01, a10, a11 (corresponding to local (0,0), (0,1), (1,0), (1,1))
    #   B indices: b00, b01, b10, b11
    #   W indices: w00, w01, w10, w11 (corresponding to local C (0,0), (0,1), (1,0), (1,1))
    #   Note: Global W index for local (i, k) is k*n + i.
    #   So w00 is C(0,0), w01 is C(0,1), w10 is C(1,0), w11 is C(1,1).
    #   Wait, the standard Strassen computes C = A*B.
    #   C00 = M1 - M4 + M6 - M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M5 + M7
    #
    # Let's map the 7 terms to columns offset..offset+6.
    # Term 0 (M1): (A00+A11)(B00+B11)
    #   U: a00 += 1, a11 += 1
    #   V: b00 += 1, b11 += 1
    #   W: w00 += 1, w11 += 1
    #
    # Term 1 (M2): (A10+A11)B00
    #   U: a10 += 1, a11 += 1
    #   V: b00 += 1
    #   W: w10 += 1, w11 -= 1
    #
    # Term 2 (M3): A00(B01-B11)
    #   U: a00 += 1
    #   V: b01 += 1, b11 -= 1
    #   W: w01 += 1, w11 += 1
    #
    # Term 3 (M4): A11(B10-B00)
    #   U: a11 += 1
    #   V: b10 += 1, b00 -= 1
    #   W: w00 += 1, w10 += 1
    #
    # Term 4 (M5): (A00+A01)B11
    #   U: a00 += 1, a01 += 1
    #   V: b11 += 1
    #   W: w00 -= 1, w01 += 1
    #
    # Term 5 (M6): (A10-A00)(B00+B01)
    #   U: a10 += 1, a00 -= 1
    #   V: b00 += 1, b01 += 1
    #   W: w11 += 1
    #
    # Term 6 (M7): (A01-A11)(B10+B11)
    #   U: a01 += 1, a11 -= 1
    #   V: b10 += 1, b11 += 1
    #   W: w00 += 1

    def add_strassen_block(i_start, j_start, k_start, offset):
        # i_start: row block of A (0 or 1)
        # j_start: col block of A / row block of B (0, 1, or 2)
        # k_start: col block of B (0, 1, or 2)
        
        # Local indices
        # A: (i, j) -> global i*m + j
        # B: (j, k) -> global j*p + k
        # W: (k, i) -> global k*n + i
        
        # A indices
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        # B indices
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        # W indices
        # Local C (i, k) corresponds to global (k_start + k, i_start + i) in the sub-block?
        # No, C is n x p.
        # A is n x m, B is m x p.
        # A block: rows i_start..i_start+1, cols j_start..j_start+1.
        # B block: rows j_start..j_start+1, cols k_start..k_start+1.
        # Product C block: rows i_start..i_start+1, cols k_start..k_start+1.
        # Global C index for local (i_loc, k_loc) is (k_start + k_loc)*n + (i_start + i_loc).
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        # Term 0
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0
        
        # Term 1
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0
        
        # Term 2
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0
        
        # Term 3
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0
        
        # Term 4
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0
        
        # Term 5
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        # Term 6
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    # Apply blocks
    # i_start in {0, 1}
    # j_start in {0, 1} (since m=4, two blocks of 2)
    # k_start in {0, 1, 2} (since p=6, three blocks of 2)
    
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        # Check non-zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    if verify(U, V, W):
        # Try to reduce rank by finding linearly dependent columns using SVD
        # Stack U, V, W into a single matrix and find the effective rank
        # Actually, we need to check if any column can be expressed as a linear combination
        # of others in a way that preserves the tensor structure.
        
        # A simpler approach: try to find a rank-r decomposition for r < 42 by
        # checking if the tensor can be decomposed with fewer terms.
        # This is NP-hard in general, but for small cases we can try greedy methods.
        
        # Alternative: Use the fact that the 6 Strassen blocks might have some shared structure.
        # Check if any two blocks can be merged.
        
        # For now, return the rank-42 solution which is verified.
        print(f"Returning rank {rank} decomposition", file=sys.stderr)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: return schoolbook rank 48
        rank_fallback = n * m * p
        U_fb = np.zeros((nm, rank_fallback), dtype=float)
        V_fb = np.zeros((mp, rank_fallback), dtype=float)
        W_fb = np.zeros((pn, rank_fallback), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        
        print(f"Returning rank {rank_fallback} fallback decomposition", file=sys.stderr)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
