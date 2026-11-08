import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # A is n x m = 2x3, B is m x p = 3x9, C is n x p = 2x9
    # Tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # Indices:
    #   A: i*m + j  (i in 0..1, j in 0..2) -> 6 elements
    #   B: j*p + k  (j in 0..2, k in 0..8) -> 27 elements
    #   C: k*n + i  (k in 0..8, i in 0..1) -> 18 elements (official trace convention W[k*n+i, r])
    
    # The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # since C[i,k] = sum_j A[i,j] B[j,k]
    
    # We want to find a rank-minimal decomposition:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's try to find a low-rank decomposition.
    
    # The trivial decomposition has rank = n*m*p = 2*3*9 = 54
    # But we can do better.
    
    # For 2x3x9, let's think about the structure:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # One approach: use the fact that for each k, we're doing a 2x3 times 3x1 multiplication.
    # The rank of 2x3x1 matrix multiplication is 3 (since it's just 3 scalar products per output).
    # But we have 9 different k values, so naively 9*3 = 27? No, that's not right either.
    
    # Actually, let's think more carefully. The tensor T is of size 2x3x9.
    # T[i,j,k] = 1 for all i,j,k.
    
    # We can write:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let's try rank 9: for each k, use a separate term?
    # T[i,j,k] = U[i,j,k] * V[j,k,k] * W[k,i,k]
    # But this gives rank 9 only if we can set U[i,j,k]=1, V[j,k,k]=1, W[k,i,k]=1 for the right entries.
    # But V[j,k,r] needs to be 1 when j matches and k matches, 0 otherwise.
    # Similarly for U and W.
    
    # Actually, let's try a simpler approach. Since the tensor is all-ones, we can use:
    # T[i,j,k] = 1 * 1 * 1 for all i,j,k
    # This is rank 1! But wait, that's not right because the decomposition requires:
    # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_k' V[j,k',r] * B[j,k'])
    # Hmm, I need to be more careful about the structure.
    
    # The correct tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = 1 (the multiplication tensor)
    # A decomposition of rank r is: T[i,j,k] = sum_{r=1}^r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # For the all-ones tensor of size 2x3x9, the rank is actually 1!
    # Because T[i,j,k] = 1 = 1*1*1 for all i,j,k.
    # So U[i,j,r] = 1, V[j,k,r] = 1, W[k,i,r] = 1 for r=1.
    
    # Wait, but that would mean C[i,k] = W[k,i,1] * (sum_j U[i,j,1] A[i,j]) * (sum_j' V[j',k,1] B[j',k])
    # = 1 * (sum_j A[i,j]) * (sum_j' B[j',k])
    # That's NOT the same as sum_j A[i,j] B[j,k]!
    
    # I was confusing the tensor structure. Let me reconsider.
    
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 if the (i,k) entry of C
    # depends on A[i,j] and B[j,k]. For standard matrix multiplication, M[i,j,k] = 1 for all valid i,j,k.
    
    # A rank-r decomposition is a set of r triples (u_r, v_r, w_r) where:
    # u_r is a vector in R^{n*m}, v_r is a vector in R^{m*p}, w_r is a vector in R^{n*p}
    # such that for all i,j,k:
    #   1 = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # And the reconstruction is:
    #   C[i,k] = sum_r w_r[k*n+i] * (sum_j u_r[i*m+j] * A[i,j]) * (sum_j v_r[j*p+k] * B[j,k])
    
    # Wait, that's not right either. Let me look at the standard formulation.
    
    # In the standard tensor decomposition for matrix multiplication:
    # We have the tensor T of size (n*m) x (m*p) x (n*p) where
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    # A rank-r decomposition is: T = sum_r U_r outer V_r outer W_r
    # where U_r is in R^{n*m}, V_r is in R^{m*p}, W_r is in R^{n*p}.
    
    # The reconstruction formula is:
    # C = sum_r (U_r * A) * (V_r * B) * W_r
    # where U_r * A means summing over the appropriate indices.
    
    # More precisely, if we flatten A to a vector a of length n*m, B to b of length m*p,
    # and C to c of length n*p, then:
    # c = sum_r W_r * (U_r^T a) * (V_r^T b)
    # Wait, that's not dimensionally correct.
    
    # Let me use the formulation from the task description:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: for all i,j,k (where i is row of A/C, j is col of A/row of B, k is col of B/C):
    #   1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And the matrix multiplication is:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Hmm, that still doesn't look right. Let me think again.
    
    # Actually, I think the correct formula is:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # No wait, that would give C[i,k] as a product of two sums, which is not linear in A and B separately.
    
    # I think the correct interpretation is:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # But that's a bilinear form in A and B, which is correct for matrix multiplication.
    
    # Let me verify with the all-ones tensor. If U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1,
    # then C[i,k] = 1 * (sum_j A[i,j]) * (sum_j B[j,k]), which is NOT the correct matrix product.
    
    # So the all-ones tensor does NOT correspond to matrix multiplication. I was confused.
    
    # The correct tensor for matrix multiplication has T[i*m+j, j*p+k, k*n+i] = 1, but the
    # decomposition must satisfy the bilinear reconstruction formula.
    
    # Let me just use the current program's approach, which uses Strassen-like blocks.
    # The current program achieves rank 50. Let me see if I can do better.
    
    # For 2x3x9, one approach is to split into 2x2x9 and 2x1x9 parts.
    # 2x2x9 can be done with 4 blocks of 2x2x2 (using Strassen, rank 7 each) plus 1 block of 2x2x1 (rank 4).
    # That gives 4*7 + 4 = 32 for the 2x2x9 part.
    # 2x1x9 is just 9 scalar multiplications, rank 9... but the current program uses 18.
    
    # Wait, for 2x1x9, C[i,k] = A[i,2] * B[2,k] for i=0,1 and k=0..8.
    # That's 2*9 = 18 scalar multiplications, so rank 18 is correct for this part.
    
    # Total: 32 + 18 = 50. That matches the current program.
    
    # Can we do better? Maybe by not splitting this way?
    
    # Let me try a different approach: use the fact that 3 = 2+1 and 9 = 9.
    # Or maybe use a more global Strassen-like approach.
    
    # Actually, let me just return the current program's decomposition, which is known to work.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
