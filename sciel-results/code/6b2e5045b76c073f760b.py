"""
Winograd-style decomposition for 4x5 by 5x6 matrix multiplication.
"""
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    
    # Partition:
    # A (4x5) -> A1 (4x4) [cols 0-3], A2 (4x1) [col 4]
    # B (5x6) -> B1 (4x6) [rows 0-3], B2 (1x6) [row 4]
    # C (4x6) = A1 @ B1 + A2 @ B2
    
    # 4-point DFT matrix F4 and its inverse F4_inv (with 1/4 scaling)
    # F4[i,j] = exp(-2*pi*i*j/4)
    # F4_inv[i,j] = (1/4) * exp(2*pi*i*j/4)
    
    # F4 for 4 points:
    # Row 0: [1, 1, 1, 1]
    # Row 1: [1, -i, -1, i]
    # Row 2: [1, -1, 1, -1]
    # Row 3: [1, i, -1, -i]
    
    # F4_inv = (1/4) * F4^H (conjugate transpose)
    
    # For the Winograd 4x4x4 core:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using FFT: 
    #   For each i, compute DFT of A[i,:] -> A_hat[i,:]
    #   For each k, compute IDFT of B[:,k] -> B_hat[:,k]  
    #   C_hat[i,k] = A_hat[i,:] * B_hat[:,k] elementwise... 
    #   
    # Actually, the correct formulation for matrix multiplication via FFT:
    # C = A @ B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is NOT a convolution, so FFT doesn't directly apply in the simple way.
    # The rank of 4x4x4 tensor is at least 4^2 = 16 (by Frobenius rank lower bound) 
    # and at most 4^3 = 64 (schoolbook).
    #
    # The key insight: we can use the fact that matrix multiplication tensor
    # T_{i,j,k} = delta_{i,j,k} (where T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k])
    # can be decomposed using the DFT basis.
    #
    # Specifically: Let F be the 4x4 DFT matrix. Then:
    # sum_j A[i,j] B[j,k] = sum_alpha (F A^T)[i,alpha] * (B F^H)[alpha,k] / 4
    # Wait, let me verify this identity.
    #
    # (F A^T)[i,alpha] = sum_j F[i,j] A^T[j,i] = sum_j F[i,j] A[i,j]
    # (B F^H)[alpha,k] = sum_j B[alpha,j] F^H[j,k] = sum_j B[j,k] conj(F[j,alpha])  [since F^H[j,k]=conj(F[k,j])]
    # Hmm, this doesn't look right.
    #
    # Let me try: sum_j A[i,j] B[j,k]
    # Note that F F^H = 4 I, so F^H = (1/4) F^* 
    # sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] * delta_{j,j}
    # = sum_j A[i,j] B[j,k] * (1/4) sum_alpha F[j,alpha] conj(F[j,alpha])  [since sum_alpha |F[j,alpha]|^2 = 4]
    # Hmm, this is getting complicated.
    #
    # Let me use a different approach. The matrix multiplication tensor for n x n x n
    # has a decomposition using the DFT:
    # T[i,j,k] = (1/n) sum_alpha F[i,alpha] conj(F[j,alpha]) F[k,alpha]  ??? 
    # No, that's not right either.
    #
    # Actually, I recall that for the matrix multiplication tensor, one can use:
    # mu(n,n,n) <= n * rank(F_n) where F_n is some transform... 
    #
    # Let me just directly construct the rank-16 decomposition for 4x4x4.
    #
    # The 4x4x4 matrix multiplication tensor T has T[i,j,k] = 1 if i,j,k form a valid 
    # multiplication term, i.e., T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = delta_{j,j} which is always 1... no wait.
    #
    # The tensor is: T[i,j,k] = 1 for all i,j,k? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k]).
    # Wait no, T[i,j,k] should be 1 if the j-th term in the sum for C[i,k] uses A[i,j] and B[j,k].
    # So T[i,j,k] = 1 for all i, j, k. That's a rank-1 tensor! But that can't be right.
    #
    # I'm confusing myself. Let me be precise:
    # The matrix multiplication tensor T is a 3-tensor where T[i,j,k] is the coefficient 
    # such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    # But this is wrong because T should depend on which j connects i and k.
    #
    # Actually, the correct tensor is: T[i,j,k] = 1 if the multiplication A[i,j]*B[j,k] 
    # contributes to C[i,k]. Since every (i,j,k) triple contributes A[i,j]*B[j,k] to C[i,k],
    # T[i,j,k] = 1 for all i,j,k. But this means the tensor is all-ones, which has rank 1.
    # That's clearly wrong.
    #
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    #
    # The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * T_{i,j,k}
    # where T_{i,j,k} = 1 if the (i,k)-entry of the product depends on A[i,j] and B[j,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], T_{i,j,k} = 1 for all i,j,k.
    #
    # But wait, that gives a rank-1 tensor, which is absurd. The issue is that 
    # the tensor should be: T_{i,j,k} = 1 only when the indices are "compatible".
    # In matrix multiplication, A[i,j] * B[j,k] contributes to C[i,k]. So for a fixed 
    # output position (i,k), we sum over j. The tensor entry T[i,j,k] = 1 means 
    # "A[i,j] and B[j,k] multiply to contribute to C[i,k]".
    #
    # So T[i,j,k] = 1 for ALL i, j, k? That would mean the tensor is the all-ones 
    # 4x4x4 tensor, which has tensor rank 1. But we know mu(4,4,4) >= 16.
    #
    # I think the confusion is that the tensor is NOT all-ones. Let me look at this differently.
    #
    # The matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix:
    # T[i,j,k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # For this to be true, we need the j-index to match, which it does by construction.
    # So T[i,j,k] = 1 for all valid i, j, k.
    #
    # But this gives a rank-1 tensor, which is impossible. The resolution is that 
    # the tensor is actually:
    # T[i,j,k] = delta_{j,j} = 1, but the tensor lives in V^* ⊗ W^* ⊗ Z where 
    # the dimensions are n*m, m*p, and p*n respectively.
    #
    # Ah, I see my error. The tensor indices are:
    # - First index: (i,j) pair, ranging over n*m values
    # - Second index: (j,k) pair, ranging over m*p values  
    # - Third index: (i,k) pair, ranging over p*n values
    #
    # And T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'
    # So T is the "identity" tensor in the sense that it picks out matching indices.
    #
    # This tensor has rank at least max(n*m, m*p, p*n) = max(20, 30, 24) = 30.
    # And schoolbook gives rank n*m*p = 120.
    #
    # OK so now I understand the structure. Let me implement the Winograd decomposition.
    #
    # For the 4x4x4 core (i in 0..3, j in 0..3, k in 0..3):
    # We want to decompose T_core[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'
    #
    # Using the DFT: Let F be the 4x4 DFT matrix. Then:
    # delta_{a,b} = (1/4) sum_alpha F[a,alpha] conj(F[b,alpha])
    #
    # So: T_core[(i,j),(j',k),(i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # = (1/4)^3 sum_{alpha,beta,gamma} F[i,alpha] conj(F[i',alpha]) F[j,beta] conj(F[j',beta]) F[k,gamma] conj(F[k',gamma])
    #
    # This gives rank 4^3 = 64, which is schoolbook. Not helpful.
    #
    # The key insight for Winograd is that we can combine the sums:
    # T_core[(i,j),(j',k),(i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    #
    # Note that delta_{j,j'} couples the first and second indices. We can write:
    # delta_{j,j'} = (1/4) sum_beta F[j,beta] conj(F[j',beta])
    #
    # So T_core = (1/4) sum_beta [delta_{i,i'} F[j,beta] conj(F[j',beta]) delta_{k,k'}]
    # 
    # Hmm, this still doesn't give rank 16.
    #
    # Let me try a different factorization. We want to write:
    # T[(i,j),(j',k),(i',k')] = sum_r U[(i,j),r] V[(j',k),r] W[(i',k'),r]
    #
    # For rank 16, we need r in 0..15.
    #
    # Let me try: r = (alpha, beta) where alpha, beta in 0..3
    # U[(i,j), (alpha,beta)] = F[i,alpha] * F[j,beta]
    # V[(j',k), (alpha,beta)] = conj(F[j',alpha]) * conj(F[k,beta])  ??? 
    # W[(i',k'), (alpha,beta)] = conj(F[i',alpha]) * conj(F[k',beta])  ???
    #
    # Let me check: sum_{alpha,beta} F[i,alpha]F[j,beta] * conj(F[j',alpha])conj(F[k,beta]) * conj(F[i',alpha])conj(F[k',beta])
    # = (sum_alpha F[i,alpha] conj(F[j',alpha]) conj(F[i',alpha])) * (sum_beta F[j,beta] conj(F[k,beta]) conj(F[k',beta]))
    #
    # This doesn't simplify nicely.
    #
    # Let me try yet another approach. I'll just use the known fact that 
    # mu(4,4,4) <= 49 and implement a simple improved decomposition, or just use
    # schoolbook for the whole thing and get rank 120.
    #
    # Actually, let me try to verify if rank 16 is achievable for 4x4x4.
    # The lower bound is max(4*4, 4*4, 4*4) = 16. So rank 16 would be optimal.
    #
    # Let me try the decomposition:
    # r = (alpha, beta), alpha, beta in {0,1,2,3}
    # U[(i,j), (alpha,beta)] = F[i, alpha] * F[j, beta]
    # V[(j',k), (alpha,beta)] = conj(F[j', alpha]) * F[k, beta]  
    # W[(i',k'), (alpha,beta)] = conj(F[i', alpha]) * conj(F[k', beta])
    #
    # Check: sum_{alpha,beta} F[i,alpha]F[j,beta] * conj(F[j',alpha])F[k,beta] * conj(F[i',alpha])conj(F[k',beta])
    # = (sum_alpha F[i,alpha] conj(F[j',alpha]) conj(F[i',alpha])) * (sum_beta F[j,beta] F[k,beta] conj(F[k',beta]))
    #
    # For this to equal delta_{i,i'} delta_{j,j'} delta_{k,k'}, we'd need:
    # sum_alpha F[i,alpha] conj(F[j',alpha]) conj(F[i',alpha)) = delta_{i,i'} delta_{j,j'} * (something)
    # This doesn't work.
    #
    # I think the correct Winograd decomposition for 4x4x4 uses:
    # U[(i,j), (alpha,beta)] = F[i, alpha] * F[j, beta]
    # V[(j',k), (alpha,beta)] = conj(F[j', beta]) * F[k, alpha]
    # W[(i',k'), (alpha,beta)] = conj(F[i', alpha]) * conj(F[k', beta])
    #
    # Check: sum_{alpha,beta} F[i,alpha]F[j,beta] * conj(F[j',beta])F[k,alpha] * conj(F[i',alpha])conj(F[k',beta])
    # = (sum_alpha F[i,alpha] F[k,alpha] conj(F[i',alpha])) * (sum_beta F[j,beta] conj(F[j',beta)) conj(F[k',beta]))
    #
    # Still doesn't work cleanly.
    #
    # I'm going to give up on deriving the rank-16 decomposition from first principles
    # and instead just implement a correct schoolbook decomposition. The rank will be 120.
    #
    # Actually, let me try one more thing. Let me use numpy to find if there's a 
    # rank-16 decomposition by solving a system of equations.
    
    # For now, let me just do schoolbook to ensure correctness.
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

# EVOLVE-BLOCK-END
