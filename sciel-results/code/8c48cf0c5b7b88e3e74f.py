"""Matrix multiplication 2x3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    size_a = n * m  # 6
    size_b = m * p  # 21
    size_c = p * n  # 14
    rank_schoolbook = n * m * p  # 42
    
    # Create the schoolbook decomposition
    U_school = np.zeros((size_a, rank_schoolbook), dtype=int)
    V_school = np.zeros((size_b, rank_schoolbook), dtype=int)
    W_school = np.zeros((size_c, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    result_schoolbook = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin before 150s limit

    # Build target tensor
    target_tensor = np.zeros((size_a, size_b, size_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0

    # Strategy 1: Constructive hybrid (Strassen for 2x2 sub-blocks + schoolbook for remainder)
    # The 2x3 matrix can be split into a 2x2 and 2x1 part
    # The 3x7 matrix can be split into 2x7 and 1x7 parts
    
    # Let A = [A12 | A13] where A12 is 2x2, A13 is 2x1
    # Let B = [B17; B27] where B17 is 2x7, B27 is 1x7
    # C = A12*B17 + A13*B27
    # Use Strassen for A12*B17 (2x2 * 2x7 -> 7 Strassen terms)
    # Use schoolbook for A13*B27 (2x1 * 1x7 -> 2 terms)
    # Total: 9 terms? No, need to be more careful.
    
    # Actually, let's think about this differently.
    # 2x3 * 3x7: split inner dimension 3 into 2+1
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x7) = [B_top (2x7); B_bottom (1x7)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left (2x2) @ B_top (2x7): use Strassen with 7 rank-1 terms
    # For A_right (2x1) @ B_bottom (1x7): 2 rank-1 terms (schoolbook)
    # Total rank: 7 + 2 = 9? That seems too low. Let me verify.
    
    # Wait, Strassen decomposes 2x2*2x2 into 7 terms. For 2x2*2x7, we apply Strassen
    # column by column, but the same 7 combinations work for all 7 columns simultaneously.
    # So A_left (2x2) @ B_top (2x7) needs 7 rank-1 terms in the tensor decomposition.
    # A_right (2x1) @ B_bottom (1x7) needs 2*1*7 = 14? No, 2*1 = 2 terms? 
    # No: A_right is 2x1, B_bottom is 1x7. The product is 2x7.
    # Schoolbook: sum over j=0 (only one inner index), so for each output (i,k), C[i,k] = A_right[i,0]*B_bottom[0,k]
    # That's 2*7 = 14 rank-1 terms? No wait, it's just elementwise product of a 2-vector and 7-vector broadcast.
    # In tensor decomposition: C[i,k] = sum_r U[i,r]*V[r]*W[k,r]... no.
    # For A_right (2x1) @ B_bottom (1x7): the inner dimension is 1.
    # C[i,k] = A_right[i,0] * B_bottom[0,k]
    # This is a rank-1 outer product: (2x1) outer (1x7) = 2x7 matrix.
    # In the tensor: one term with U having the 2 values, V having 1, W having the 7 values.
    # So it's 1 rank-1 term! Not 2.
    
    # Hmm, but in the standard tensor decomposition for matrix multiplication M_{n,m,p},
    # each rank-1 term corresponds to one (i,j,k) triple in the schoolbook.
    # For the sub-problem A_right (2x1) @ B_bottom (1x7):
    # The "inner" index is j=0 (local), so we have n=2, m=1, p=7.
    # Schoolbook rank = 2*1*7 = 14.
    # But we can do better: C[i,k] = A_right[i,0]*B_bottom[0,k]
    # This is: U[i,0]=A_right[i,0], V[0,0]=1, W[k,0]=B_bottom[0,k] -- no, that's not right either.
    
    # Let me reconsider. In the tensor T[a,b,c] for a 2x1x7 multiplication:
    # a ranges over 2*1=2 indices (i,j) with j always 0
    # b ranges over 1*7=7 indices (j,k) with j always 0  
    # c ranges over 7*2=14 indices (k,i)
    # T[a,b,c]=1 iff i_a=i_c, j_a=j_b, k_b=k_c
    # Since j_a=0, j_b=0 always, the condition is i_a=i_c and k_b=k_c.
    # T is nonzero when: a=i (i=0,1), b=k (k=0..6), c=k*2+i
    # So T[i, k, k*2+i] = 1 for i in {0,1}, k in {0..6}
    # This is a 2x7x14 tensor with 14 nonzero entries.
    # Can we decompose with fewer than 14 rank-1 terms?
    # T[i,k,c] = sum_r U[i,r] V[k,r] W[c,r]
    # For each (i,k), c is determined: c = k*2+i
    # T[i,k,k*2+i] = 1, all other entries 0.
    # This looks like it might need rank 14, or maybe we can do better.
    
    # Actually, I think the correct approach is:
    # The full 2x3x7 multiplication has tensor rank at most 42 (schoolbook).
    # Known results: rank of 2x3x7 is between... let me just try to find a good decomposition.
    
    # Let me try a different structural approach:
    # Split the 3 into 2+1, use Strassen-like for the 2x2 part.
    
    # A (2x3): columns 0,1 form 2x2 block; column 2 is separate
    # B (3x7): rows 0,1 form 2x7 block; row 2 is separate
    # C = A[:,0:2] @ B[0:2,:] + A[:,2] @ B[2,:]
    
    # First term: 2x2 @ 2x7 using Strassen (7 terms)
    # Second term: 2x1 @ 1x7 (rank 14 in schoolbook, but actually...)
    
    # For 2x1 @ 1x7: the inner dimension is 1, so it's just an outer product.
    # In tensor form: we need 14 rank-1 terms (one for each (i,k) pair) OR
    # we can use the fact that it's separable.
    
    # Actually, for M_{2,1,7}: the tensor has T[i*1+0, 0*7+k, k*2+i] = 1
    # = T[i, k, 2k+i] = 1
    # Each (i,k) gives a unique (a,b,c) = (i, k, 2k+i)
    # There are 2*7=14 such entries, all in different "lines" of the tensor.
    # The rank of this tensor is 14 (I believe it's a permutation-like tensor).
    
    # So total: 7 + 14 = 21? That's better than 42!
    
    # Wait, I need to be more careful. Let me construct this explicitly.
    
    def construct_hybrid():
        """Construct hybrid Strassen + schoolbook decomposition."""
        # Strassen for 2x2 * 2x7
        # Local indices: A has rows 0,1 and cols 0,1 (global cols 0,1)
        #               B has rows 0,1 and cols 0..6 (global rows 0,1, cols 0..6)
        #               C has rows 0,1 and cols 0..6
        
        # Strassen terms for 2x2*2x2, extended to 2x2*2x7:
        # The 7 Strassen products are:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # For 2x2 * 2x7, b00,b01,...,b06 and b10,b11,...,b16
        # Each Strassen term involves combinations of b's across all 7 columns.
        # M1: (a00+a11) * (b0k+b1k) for each k -> this is a product of a 2-vector combo with a 2-vector combo
        # In tensor rank-1 terms, each "product" of (linear combo of A entries) * (linear combo of B entries)
        # contributes one rank-1 term per output entry... no, that's not right.
        
        # Actually, Strassen gives us 7 bilinear forms. For 2x2*2x2, each bilinear form
        # M_r = (linear combo of A) * (linear combo of B)
        # Then C entries are linear combos of the M_r's.
        
        # In tensor decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For Strassen: we have 7 "rank-1" components in the bilinear sense, but in the
        # trilinear tensor sense, each Strassen term M_r = sum_{i,j} alpha_{i,j}^{(r)} * A[i,j] * B[j,k]
        # becomes a sum over k of rank-1 terms? No...
        
        # Let me think again. The tensor T for matrix mult M_{n,m,p} is:
        # T[i*m+j, j*p+k, k*n+i] = 1
        # A rank-R decomposition: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        
        # For Strassen on 2x2*2x2: the tensor has size 4x4x4=64, with 8 nonzero entries.
        # The 7-term decomposition means T = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r with rank 7.
        
        # For 2x2*2x7: the tensor has size 4x14x14. 
        # We can view B as 2 rows of 7 cols. The multiplication is:
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0..6
        # This is 14 output entries, each a sum of 2 products.
        # Total schoolbook: 2*2*7 = 28 rank-1 terms.
        # Strassen should reduce this.
        
        # The key insight: Strassen reduces 2x2*2x2 from 8 to 7 terms.
        # For 2x2*2x7, we can apply the same 7 Strassen "recipes" but each recipe
        # now involves 7 columns of B. 
        # M_r = (linear combo of 4 A entries) * (linear combo of B entries)
        # But "linear combo of B entries" now has 14 entries (2x7), not 4.
        
        # Hmm, I think the right way is:
        # For each column k of B, we have a 2x1 column. The product A @ B[:,k] is 2x1.
        # Strassen applies to each 2x2*2x1? No, Strassen is for 2x2*2x2.
        
        # I think for 2x2*2x7, we should think of it as 2x2*2x2 repeated 3.5 times,
        # which doesn't work cleanly. 
        
        # Alternative: pad to 2x2*2x8 (8=2*4) or use 2x2*2x7 directly.
        
        # Actually, the cleanest approach: treat the 2x2*2x7 as a block.
        # The tensor T_{2,2,7}[a,b,c] where a in {0,1,2,3} (i*2+j), b in {0..13} (j*7+k), c in {0..13} (k*2+i)
        # Nonzero when: i_a=i_c, j_a=j_b, k_b=k_c
        # T[i*2+j, j*7+k, k*2+i] = 1
        
        # Can we find a rank-7 decomposition? Probably not directly from Strassen.
        # The rank of M_{2,2,7} is known to be... I'm not sure.
        
        # Let me try a completely different approach: use the known structure.
        
        # For M_{2,3,7}, the schoolbook rank is 42.
        # We can try to find a lower rank by exploiting linearity.
        
        # Key observation: if we can find linear dependencies among the 42 standard
        # basis tensors, we can reduce the rank.
        
        # The tensor T is determined by 42 nonzero entries (one for each (i,j,k)).
        # T[i*3+j, j*7+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2}, k in {0..6}
        
        # A rank-R decomposition requires finding R vectors u_r in R^6, v_r in R^21, w_r in R^14
        # such that sum_r u_r[i*3+j] * v_r[j*7+k] * w_r[k*2+i] = 1 if (i,j,k) valid, 0 otherwise.
        
        # Let me try to construct a rank-30 decomposition or similar.
        
        # Actually, let me just try random search with better initialization and more iterations.
        
        return None
    
    # Strategy: Use structured initialization based on block decomposition
    # and then refine with optimization.
    
    def try_rank_structured(R, max_restarts=10):
        """Try to find rank-R decomposition with structured initialization."""
        if time.time() > deadline:
            return None
        
        from scipy.optimize import minimize
        
        best_res = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Structured initialization: start from schoolbook and randomly combine
            # Randomly select R out of 42 schoolbook terms and try to "merge" them
            
            # Method 1: Random Gaussian
            if restart == 0:
                params = np.random.normal(0, 0.5, (size_a + size_b + size_c) * R)
            # Method 2: Sparse random
            elif restart == 1:
                params = np.zeros((size_a + size_b + size_c) * R)
                for _ in range(R * 3):
                    idx = np.random.randint(0, len(params))
                    params[idx] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            # Method 3: Based on schoolbook with random perturbations
            elif restart == 2:
                params = np.zeros((size_a + size_b + size_c) * R)
                # Initialize first R schoolbook terms
                for r in range(min(R, 42)):
                    i = r // (m * p)
                    rem = r % (m * p)
                    j = rem // p
                    k = rem % p
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    params[a * R + r] = 1.0
                    params[(size_a + b) * R + r] = 1.0
                    params[(size_a + size_b + c) * R + r] = 1.0
                # Add small noise
                params += np.random.normal(0, 0.1, len(params))
            else:
                # More random initializations
                scale = 0.3 + 0.1 * restart
                params = np.random.normal(0, scale, (size_a + size_b + size_c) * R)
            
            def residual_func(x):
                U = x[:size_a * R].reshape((size_a, R))
                V = x[size_a * R:(size_a + size_b) * R].reshape((size_b, R))
                W = x[(size_a + size_b) * R:].reshape((size_c, R))
                computed = np.einsum('ar,br,cr->abc', U, V, W)
                diff = computed - target_tensor
                return np.sum(diff * diff)
            
            def grad_func(x):
                U = x[:size_a * R].reshape((size_a, R))
                V = x[size_a * R:(size_a + size_b) * R].reshape((size_b, R))
                W = x[(size_a + size_b) * R:].reshape((size_c, R))
                
                # Gradient of sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
                # = 2 * sum_{a,b,c} (computed[a,b,c] - T[a,b,c]) * d(computed)/d(param)
                
                computed = np.einsum('ar,br,cr->abc', U, V, W)
                residual = computed - target_tensor  # (a,b,c)
                
                # d(computed[a,b,c])/d(U[a0,r0]) = V[b,r0]*W[c,r0] if a==a0
                # grad_U[a0,r0] = 2 * sum_{b,c} residual[a0,b,c] * V[b,r0] * W[c,r0]
                # = 2 * sum_c (sum_b residual[a0,b,c] * V[b,r0]) * W[c,r0]
                
                # More efficiently:
                # tmp[a0,c,r0] = sum_b residual[a0,b,c] * V[b,r0]
                # grad_U[a0,r0] = 2 * sum_c tmp[a0,c,r0] * W[c,r0]
                
                tmp1 = np.einsum('abc,br->acr', residual, V)  # (a,c,r)
                grad_U = 2 * np.einsum('acr,cr->ar', tmp1, W)
                
                # Similarly for V:
                # grad_V[b0,r0] = 2 * sum_{a,c} residual[a,b0,c] * U[a,r0] * W[c,r0]
                tmp2 = np.einsum('abc,ar->bcr', residual, U)  # (b,c,r)
                grad_V = 2 * np.einsum('bcr,cr->br', tmp2, W)
                
                # And for W:
                # grad_W[c0,r0] = 2 * sum_{a,b} residual[a,b,c0] * U[a,r0] * V[b,r0]
                tmp3 = np.einsum('abc,ar->bcr', residual, U)  # (b,c,r)
                grad_W = 2 * np.einsum('bcr,br->cr', tmp3, V)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                res = minimize(
                    residual_func,
                    params,
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 500}
                )
            except Exception as e:
                continue
            
            if res.fun < best_res:
                best_res = res.fun
                best_params = res.x.copy()
            
            if best_res < 1e-6:
                # Try to snap to half-integers
                U_flat = best_params[:size_a * R]
                V_flat = best_params[size_a * R:(size_a + size_b) * R]
                W_flat = best_params[(size_a + size_b) * R:]
                
                # Try snapping to nearest 0.5
                U_snap = np.round(U_flat * 2) / 2.0
                V_snap = np.round(V_flat * 2) / 2.0
                W_snap = np.round(W_flat * 2) / 2.0
                
                U_s = U_snap.reshape((size_a, R))
                V_s = V_snap.reshape((size_b, R))
                W_s = W_snap.reshape((size_c, R))
                
                computed_snap = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                if np.max(np.abs(computed_snap - target_tensor)) < 1e-9:
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                # Try snapping to nearest integer
                U_snap2 = np.round(U_flat)
                V_snap2 = np.round(V_flat)
                W_snap2 = np.round(W_flat)
                U_s2 = U_snap2.reshape((size_a, R))
                V_s2 = V_snap2.reshape((size_b, R))
                W_s2 = W_snap2.reshape((size_c, R))
                computed_snap2 = np.einsum('ar,br,cr->abc', U_s2, V_s2, W_s2)
                if np.max(np.abs(computed_snap2 - target_tensor)) < 1e-9:
                    return {"U": U_s2.tolist(), "V": V_s2.tolist(), "W": W_s2.tolist()}
                
                # Polish with very tight tolerance
                try:
                    res2 = minimize(
                        residual_func,
                        best_params,
                        jac=grad_func,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
                    )
                    if res2.fun < best_res:
                        best_res = res2.fun
                        best_params = res2.x.copy()
                        
                        U_flat = best_params[:size_a * R]
                        V_flat = best_params[size_a * R:(size_a + size_b) * R]
                        W_flat = best_params[(size_a + size_b) * R:]
                        U_snap = np.round(U_flat * 2) / 2.0
                        V_snap = np.round(V_flat * 2) / 2.0
                        W_snap = np.round(W_flat * 2) / 2.0
                        U_s = U_snap.reshape((size_a, R))
                        V_s = V_snap.reshape((size_b, R))
                        W_s = W_snap.reshape((size_c, R))
                        computed_snap = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                        if np.max(np.abs(computed_snap - target_tensor)) < 1e-9:
                            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                except Exception:
                    pass
                
                if best_res < 1e-10:
                    # Accept as-is (floats)
                    U_s = best_params[:size_a * R].reshape((size_a, R))
                    V_s = best_params[size_a * R:(size_a + size_b) * R].reshape((size_b, R))
                    W_s = best_params[(size_a + size_b) * R:].reshape((size_c, R))
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None
    
    # Try ranks from 35 down to 25
    best_result = result_schoolbook
    for R in range(38, 24, -1):
        if time.time() > deadline:
            break
        result = try_rank_structured(R, max_restarts=5)
        if result is not None:
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
