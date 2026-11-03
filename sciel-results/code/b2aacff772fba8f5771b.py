import time
import numpy as np

def construct():
    start = time.time()
    n, m, p = 3, 4, 7
    
    # Split m=4 into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, compute 3x2x7 multiplication
    # Using the 1x2xp rank-(p+1) construction for each row i
    
    # 1x2xp decomposition: c_k = a_0*b_{0k} + a_1*b_{1k}
    # Rank (p+1) = 8 for p=7
    # Terms:
    #   t_0: (a_0, b_{00}) -> c_0  : U=a0, V=b00, W=c0
    #   t_1: (a_1, b_{10}) -> c_0  : U=a1, V=b10, W=c0
    #   t_k for k=1..p-1: (a_0+a_1, b_{1k}) -> c_k  : U=a0+a1, V=b1k, W=ck
    #   Wait, that's not right. Let me reconsider.
    #
    # Standard 1x2xp: c_k = a_0*b_{0k} + a_1*b_{1k}
    # Define s = a_0 + a_1
    # Then a_0 = s - a_1
    # c_k = (s-a_1)*b_{0k} + a_1*b_{1k} = s*b_{0k} + a_1*(b_{1k} - b_{0k})
    # This needs: for each k, two products: s*b_{0k} and a_1*(b_{1k}-b_{0k})
    # But that's 2p terms, worse.
    #
    # Better approach: 
    # c_k = a_0*b_{0k} + a_1*b_{1k}
    # = a_0*(b_{0k}) + a_1*(b_{1k})
    # Terms: (a_0, b_{0k}, c_k) for all k, and (a_1, b_{1k}, c_k) for all k = 2p terms
    # That's just schoolbook for 1x2xp.
    #
    # The rank of 1x2xp is actually p+1 (for p>=2).
    # Construction:
    #   T_0: (a_0, b_{00}, c_0)  -- rank 1
    #   T_1: (a_1, b_{10}, c_0)  -- rank 1
    #   For k = 1 to p-1:
    #     T_k: ((a_0 + a_1), b_{1k}, c_k)  -- but this gives (a_0+a_1)*b_{1k}, not what we want
    #
    # Let me look at this differently. The tensor for 1x2xp has entries T_{j,k} where j in {0,1}, k in {0..p-1}
    # and output index k. The multiplication is sum_j a_j * b_{j,k}.
    # This is a matrix of shape (2 x p) for the "middle" dimension.
    # Rank of this tensor: it's the rank of the 2 x p matrix when viewed appropriately.
    # Actually, the tensor T_{j,k} = e_k (the k-th standard basis vector in output).
    # So T = sum_{j,k} a_j b_{j,k} e_k^out
    # This is sum_k (sum_j a_j b_{j,k}) e_k
    # Each k gives one output component.
    # The tensor is: for each (j,k) input pair, the output is delta_{k}(output_index).
    # So T_{j,k,ell} = delta_{k,ell}
    # This is a 2 x p x p tensor. Its rank is at most 2p (schoolbook).
    # Can we do better? 
    # Note that T_{j,k,ell} = delta_{k,ell} is independent of j!
    # So T = sum_j (e_j) tensor (sum_k b_{j,k} e_k) ... no.
    # T_{j,k,ell} = delta_{k,ell}
    # = (sum_j delta_{j,j'}) * delta_{k,ell} for any fixed j'... no.
    # Actually T_{j,k,ell} = delta_{k,ell} for all j.
    # So T = (sum_{j=0}^{1} e_j) tensor (I_p)
    # where I_p is the identity tensor in the (k, ell) indices.
    # The identity tensor I_p has rank p.
    # So T = u tensor I_p where u = (1,1) in the j-space.
    # This has rank p!
    # 
    # So 1x2xp has rank p, not p+1!
    #
    # Let me verify: T_{j,k,ell} = delta_{k,ell}
    # Decomposition: sum_{r=0}^{p-1} U_{j,r} V_{k,r} W_{ell,r}
    # Set U_{j,r} = 1 for all j, r  (so U is 2 x p, all ones)
    # Set V_{k,r} = delta_{k,r}
    # Set W_{ell,r} = delta_{ell,r}
    # Then sum_r U_{j,r} V_{k,r} W_{ell,r} = sum_r 1 * delta_{k,r} * delta_{ell,r} = delta_{k,ell} ✓
    #
    # Great! So 1x2xp rank = p.
    #
    # Now for 3x2x7: we have 3 rows, each doing 1x2x7.
    # The tensor is T_{i,j,k,ell} = delta_{k,ell} * (something in i,j?)
    # Actually for 3x2x7: C_{i,ell} = sum_j A_{i,j} B_{j,ell}
    # T_{i,j,ell} = delta_{?}... 
    # The tensor entries: T_{i,j,k,ell} where i in {0,1,2}, j in {0,1}, k,ell in {0..6}
    # T_{i,j,k,ell} = delta_{k,ell} (independent of i and j!)
    # Wait, that's the same as above but with an extra index i that doesn't matter.
    # T_{i,j,k,ell} = delta_{k,ell}
    # = (sum_i e_i) tensor (sum_j e_j) tensor I_{k,ell}
    # Rank = 3 * 2 * 7 = 42? No, that's schoolbook.
    #
    # Actually: T_{i,j,k,ell} = delta_{k,ell}
    # We can write: sum_{r=0}^{6} U_{i,j,r} V_{k,r} W_{ell,r}
    # With U_{i,j,r} = 1 for all i,j,r, V_{k,r} = delta_{k,r}, W_{ell,r} = delta_{ell,r}
    # Then sum_r 1 * delta_{k,r} * delta_{ell,r} = delta_{k,ell} ✓
    # Rank = 7!
    #
    # Wait, that can't be right. Let me check dimensions.
    # U has shape (3*2, 7) = (6, 7), all ones.
    # V has shape (2*7, 7) = (14, 7), V_{j*7+k, r} = delta_{k,r}
    # W has shape (7*3, 7) = (21, 7), W_{ell*3+i, r} = delta_{ell,r}
    #
    # Check: sum_r U_{i*2+j, r} V_{j*7+k, r} W_{ell*3+i, r}
    # = sum_r 1 * delta_{k,r} * delta_{ell,r} = delta_{k,ell} ✓
    #
    # But this doesn't involve A and B correctly! The decomposition should give:
    # C_{i,ell} = sum_{j,k} A_{i,j} B_{j,k} * T_{i,j,k,ell}
    # = sum_{j,k} A_{i,j} B_{j,k} delta_{k,ell}
    # = sum_j A_{i,j} B_{j,ell} ✓
    #
    # So the tensor T_{i,j,k,ell} = delta_{k,ell} has rank 7 for 3x2x7!
    #
    # Hmm, but that seems too good. Let me double-check.
    # The tensor T is in (i,j) x (j,k) x (ell,i) space... 
    #
    # Actually, I think I'm confusing myself. Let me be very careful.
    #
    # Matrix multiplication: C = A B, A is (n x m), B is (m x p), C is (n x p)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # Tensor T has indices (i, j, k) with T_{i,j,k} = 1 if we're computing the (i,k) entry using A_{i,j} and B_{j,k}
    # More precisely, the multiplication tensor is:
    # M_{(i,j),(j,k),(k,i)} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
    #
    # Let me use the standard formulation. The tensor for n x m x p matrix mult is:
    # T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # and T_{i,j,k} is the coefficient such that C_{i,k} = sum_{j,j'} A_{i,j} B_{j',k} T_{i,j,j',k}
    #
    # No, simpler: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the "tensor" is T_{i,j,k} = 1 (for all valid i,j,k), and the contraction is
    # C_{i,k} = sum_j (A B-contraction)_j
    #
    # The rank of the matrix multiplication tensor <A,B> -> C is the minimum rank R such that
    # C_{i,k} = sum_{c=0}^{R-1} sum_j U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} A_{i,j} B_{j,k}
    #
    # Equivalently, T_{i,j,k} = sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c}
    #
    # For our case n=3, m=4, p=7:
    # T_{i,j,k} = 1 for all i in [0,3), j in [0,4), k in [0,7)
    #
    # This is a 3x4x7 tensor of all ones.
    #
    # Rank of all-ones tensor of size a x b x c is 1!
    # Because T_{i,j,k} = 1 * 1 * 1, so U_{(i,j),0} = 1, V_{(j,k),0} = 1, W_{(k,i),0} = 1.
    #
    # Wait, that would mean matrix multiplication has rank 1?! That's obviously wrong.
    #
    # I'm making an error. The tensor is NOT all ones. Let me reconsider.
    #
    # The matrix multiplication tensor is defined such that:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    #
    # In tensor decomposition form:
    # C_{i,k} = sum_c (sum_j U_{(i,j),c} A_{i,j}) (sum_{j'} V_{(j',k),c} B_{j',k}) W_{(k,i),c}
    #
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_c W_{(k,i),c} (sum_j U_{(i,j),c} A_{i,j}) (sum_{j'} V_{(j',k),c} B_{j',k}) = sum_j A_{i,j} B_{j,k}
    #
    # This must hold for all A, B. The tensor T_{i,j,k} is defined by:
    # T_{i,j,k} = sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c}
    #
    # And the condition is that the trilinear form satisfies the matrix mult identity.
    #
    # For schoolbook: c = (i,j,k) with R = n*m*p
    # U_{(i,j),(i',j',k')} = delta_{i,i'} delta_{j,j'}
    # V_{(j,k),(i',j',k')} = delta_{j,j'} delta_{k,k'}  
    # W_{(k,i),(i',j',k')} = delta_{k,k'} delta_{i,i'}
    # Then T_{i,j,k} = sum_{i',j',k'} delta_{i,i'}delta_{j,j'} delta_{j,j'}delta_{k,k'} delta_{k,k'}delta_{i,i'}
    # = delta_{i,i}delta_{j,j}delta_{k,k} when i'=i,j'=j,k'=k = 1
    # Hmm, that gives T_{i,j,k} = 1 again.
    #
    # I think the issue is that T_{i,j,k} = 1 for all i,j,k is correct for the "raw" tensor,
    # but the constraint is not just on T, but on how U, V, W interact with the specific
    # index structure.
    #
    # The key constraint is:
    # C_{i,k} = sum_c [sum_j U_{(i,j),c} A_{i,j}] [sum_{j'} V_{(j',k),c} B_{j',k}] W_{(k,i),c}
    #
    # This is a trilinear form in A, B, W-outputs. For it to equal matrix multiplication,
    # we need specific algebraic relations.
    #
    # The rank of the matrix multiplication tensor is NOT simply the rank of the all-ones tensor.
    # It's a more subtle invariant.
    #
    # Given the complexity and the fact that I keep making errors in the tensor analysis,
    # and given the 120-second time limit, the safest approach is to return the schoolbook
    # decomposition which is guaranteed correct.
    
    # Schoolbook decomposition: rank = n * m * p = 84
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][c] = 1
                V[idx_v][c] = 1
                W[idx_w][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}
