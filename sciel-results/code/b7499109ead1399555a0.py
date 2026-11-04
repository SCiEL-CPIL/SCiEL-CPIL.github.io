import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 24
    
    # Trivial decomposition
    U_trivial = np.zeros((n * m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m * p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p * n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = trivial_rank

    def compute_error(U, V, W):
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        try:
            T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
            err = np.sum((T_calc - T)**2)
            return err
        except Exception:
            return float('inf')

    def verify_decomposition(U, V, W, tol=1e-10):
        err = compute_error(U, V, W)
        return err < tol

    def build_strassen_rank14():
        """
        Build a rank-14 decomposition using Strassen's 2x2x2 algorithm
        combined with the 1x1x2 trivial structure.
        
        The 2x3x4 multiplication can be viewed as:
        - Split the 3-column input into (2,1) and the 4-row output into (2,2)
        - The 2x2x2 block uses Strassen (rank 7)
        - The 1x1x2 block uses trivial (rank 2)
        - Total: 7 + 7 + 2 = 16? No, let me think more carefully.
        
        Actually, for 2x3x4:
        A is 2x3, B is 3x4, C is 2x4
        
        We can split B into B1 (3x2) and B2 (3x2)
        Then C = A*B1 (2x2) + A*B2 (2x2)
        
        But A*B1 and A*B2 are both 2x3 times 3x2 = 2x2 matrices.
        Each 2x3x2 multiplication has rank at most 6 (trivial) or can be done better.
        
        Let me use a different structural approach:
        Split the 3 intermediate dimension into (2,1):
        A = [A1 | A2] where A1 is 2x2, A2 is 2x1
        B = [B1; B2] where B1 is 2x4, B2 is 1x4
        
        C = A1*B1 + A2*B2
        
        A1*B1 is 2x2 * 2x4 = 2x4 (can use Strassen-like for 2x2x2 blocks)
        A2*B2 is 2x1 * 1x4 = 2x4 (outer product, rank 2 per column pair)
        
        For A1*B1: split B1 into B1a (2x2) and B1b (2x2)
        A1*B1a is 2x2x2 -> Strassen rank 7
        A1*B1b is 2x2x2 -> Strassen rank 7
        Total for A1*B1: 14
        
        For A2*B2: A2 is 2x1, B2 is 1x4
        This is an outer product: C[i,k] = A2[i,0] * B2[0,k]
        This has rank 2 (one per row of the 2x1, but actually it's rank 1 per k)
        Actually, A2*B2[i,k] = A2[i,0]*B2[0,k]
        We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = A2[i,0]*B2[0,k]
        where j is the intermediate (only j=0 matters for A2, k is output col)
        
        Wait, I need to be more careful with the tensor indices.
        
        The multiplication tensor T[i,j,k] = 1 for all i,j,k means:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For the decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        where U has shape (n,m,rank), V has shape (m,p,rank), W has shape (p,n,rank)
        
        Let me just construct it directly using the known Strassen decomposition
        for the 2x2x2 part and handle the rest.
        """
        # Strassen's 7 rank-1 terms for 2x2x2 matrix multiplication
        # C[i,k] = sum_j A[i,j]*B[j,k] for i,j,k in {0,1}
        # Strassen decomposition:
        # m1 = (A00+A11)*(B00+B11)
        # m2 = (A10+A11)*B00
        # m3 = A00*(B01-B11)
        # m4 = A11*(B10-B00)
        # m5 = (A00+A01)*B11
        # m6 = (A10-A00)*(B00+B10)
        # m7 = (A01-A11)*(B01+B11)
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        
        strassen_U = [
            # m1: (A00+A11)*(B00+B11)
            [[1,0],[1,0]],  # U[i,j] for this term
            # m2: (A10+A11)*B00
            [[0,0],[1,1]],
            # m3: A00*(B01-B11)
            [[1,0],[0,0]],
            # m4: A11*(B10-B00)
            [[0,0],[0,1]],
            # m5: (A00+A01)*B11
            [[1,1],[0,0]],
            # m6: (A10-A00)*(B00+B10)
            [[-1,1],[0,0]],
            # m7: (A01-A11)*(B01+B11)
            [[0,1],[0,-1]],
        ]
        strassen_V = [
            # m1: (A00+A11)*(B00+B11)
            [[1,1],[1,1]],  # V[j,k] for this term
            # m2: (A10+A11)*B00
            [[1,0],[0,0]],
            # m3: A00*(B01-B11)
            [[0,1],[0,-1]],
            # m4: A11*(B10-B00)
            [[-1,0],[0,1]],
            # m5: (A00+A01)*B11
            [[0,0],[0,1]],
            # m6: (A10-A00)*(B00+B10)
            [[1,1],[0,0]],
            # m7: (A01-A11)*(B01+B11)
            [[0,1],[1,1]],
        ]
        strassen_W = [
            # m1: (A00+A11)*(B00+B11)
            [[1,0],[1,0]],  # W[k,i] for this term
            # m2: (A10+A11)*B00
            [[0,0],[1,0]],
            # m3: A00*(B01-B11)
            [[0,1],[0,0]],
            # m4: A11*(B10-B00)
            [[0,0],[1,0]],
            # m5: (A00+A01)*B11
            [[0,0],[0,1]],
            # m6: (A10-A00)*(B00+B10)
            [[1,0],[0,0]],
            # m7: (A01-A11)*(B01+B11)
            [[0,1],[0,1]],
        ]
        
        # Now for 2x3x4:
        # n=2, m=3, p=4
        # Split m=3 into m1=2, m2=1
        # Split p=4 into p1=2, p2=2
        
        # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> 2x2x2, use Strassen (rank 7)
        # Block (0,1): A[0:2,0:2] * B[0:2,2:4] -> 2x2x2, use Strassen (rank 7)  
        # Block (1,0): A[0:2,2:3] * B[2:3,0:2] -> 2x1x2, trivial (rank 2)
        # Block (1,1): A[0:2,2:3] * B[2:3,2:4] -> 2x1x2, trivial (rank 2)
        
        # Total rank: 7+7+2+2 = 18? That's not 14.
        
        # Let me reconsider. Maybe I can combine the two 2x1x2 blocks.
        # A[0:2,2:3] * B[2:3,0:4] is 2x1 * 1x4 = 2x4
        # This is an outer product: C[i,k] = A[i,2] * B[2,k]
        # For this to work in the tensor decomposition, we need:
        # sum_r U[i,2,r]*V[2,k,r]*W[k,i,r] = A[i,2]*B[2,k]
        # This can be done with rank 2: 
        #   r=0: U[0,2,0]=1, V[2,0,0]=1, W[0,0,0]=1 (contributes A[0,2]*B[2,0] to C[0,0])
        #   Actually, we need it for all k. Let me think again.
        
        # The term A[i,2]*B[2,k] for fixed i,k is a product of one A element and one B element.
        # In the tensor decomposition, each rank-1 term contributes U[i,j,r]*V[j,k,r]*W[k,i,r].
        # For j=2 (the third column of A / third row of B):
        # We need sum_r U[i,2,r]*V[2,k,r]*W[k,i,r] = A[i,2]*B[2,k]
        # 
        # If we set for each (i,k) pair a separate rank-1 term, that's 8 terms (2*4).
        # But we can do better. Note that A[i,2]*B[2,k] = (A[i,2])*(B[2,k])
        # This is a rank-1 matrix in the (i,k) space.
        # In the tensor decomposition, if we set:
        #   For r: U[i,2,r] = A[i,2], V[2,k,r] = B[2,k], W[k,i,r] = 1
        # But U, V, W must be fixed (not depend on A, B). So we need:
        #   sum_r U[i,2,r]*V[2,k,r]*W[k,i,r] = A[i,2]*B[2,k] for ALL A,B
        # This means the tensor T_sub[i,k] = A[i,2]*B[2,k] must be decomposable.
        # The tensor for this sub-problem has T[i,k] = 1 for all i,k (since it's just the product).
        # The rank of the 2x1x4 multiplication tensor... 
        # Actually, A[i,2]*B[2,k] is just the outer product of A[:,2] and B[2,:].
        # The tensor is T[i,j,k] where j=2 only, so T[i,2,k] = 1.
        # This is a 2x1x4 tensor with all ones. Its rank is 2 (since it's a rank-1 matrix in i,k).
        # We can decompose it as:
        #   r=0: U[0,2,0]=1, V[2,0,0]=1, W[0,0,0]=1; U[1,2,0]=1, V[2,1,0]=1, W[1,0,0]=1... 
        # No, that doesn't work because W[k,i,r] couples i and k.
        
        # Let me just use the trivial decomposition for the 2x1x4 part: rank 8.
        # Total: 7+7+8 = 22. Not great.
        
        # Alternative: Use a different splitting.
        # What if I split differently to get better rank?
        
        # Let me just try to construct a rank-14 decomposition directly.
        # Known: rank(2x3x4) <= 14? Let me check literature.
        # Actually, I recall that rank(2x3x4) might be 14 or 15.
        
        # For now, let me construct the best I can and verify.
        
        # Let me try a direct construction with rank 14.
        # I'll use the following structure:
        # Split m=3 as (1,2) and p=4 as (2,2)
        # Block (0,0): A[:,0:1]*B[0:1,0:2] = 2x1x2, rank 2
        # Block (0,1): A[:,0:1]*B[0:1,2:4] = 2x1x2, rank 2
        # Block (1,0): A[:,1:3]*B[1:3,0:2] = 2x2x2, rank 7 (Strassen)
        # Block (1,1): A[:,1:3]*B[1:3,2:4] = 2x2x2, rank 7 (Strassen)
        # Total: 2+2+7+7 = 18. Still 18.
        
        # Hmm, it seems hard to get below 18 with simple block decomposition.
        # Let me just return the trivial rank-24 for now, or try to find a better one.
        
        # Actually, let me just construct the rank-18 decomposition and see if it works.
        rank = 18
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        # Block (0,0): A[:,0:1]*B[0:1,0:2], trivial rank 2
        # i in {0,1}, j=0, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i * 3 + 0, col] = 1.0
                V[0 * 4 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
        
        # Block (0,1): A[:,0:1]*B[0:1,2:4], trivial rank 2
        # i in {0,1}, j=0, k in {2,3}
        for i in range(2):
            for k in range(2, 4):
                U[i * 3 + 0, col] = 1.0
                V[0 * 4 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
        
        # Block (1,0): A[:,1:3]*B[1:3,0:2], Strassen rank 7
        # i in {0,1}, j in {1,2}, k in {0,1}
        for r in range(7):
            su = strassen_U[r]
            sv = strassen_V[r]
            sw = strassen_W[r]
            for ii in range(2):
                for jj in range(2):
                    U[ii * 3 + (jj + 1), col] = su[ii][jj]
                for kk in range(2):
                    V[(jj + 1) * 4 + kk, col] = sv[jj][kk]
                    W[kk * 2 + ii, col] = sw[kk][ii]
            col += 1
        
        # Block (1,1): A[:,1:3]*B[1:3,2:4], Strassen rank 7
        # i in {0,1}, j in {1,2}, k in {2,3}
        for r in range(7):
            su = strassen_U[r]
            sv = strassen_V[r]
            sw = strassen_W[r]
            for ii in range(2):
                for jj in range(2):
                    U[ii * 3 + (jj + 1), col] = su[ii][jj]
                for kk in range(2):
                    V[(jj + 1) * 4 + (kk + 2), col] = sv[jj][kk]
                    W[(kk + 2) * 2 + ii, col] = sw[kk][ii]
            col += 1
        
        # Verify
        if verify_decomposition(U, V, W):
            return U, V, W, rank
        
        return None

    def discrete_local_search(U, V, W, rank, max_sweeps=5):
        vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        best_err = compute_error(U, V, W)
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 5.0:
                return U, V, W, best_err
                
            improved = False
            
            # Update U
            for r in range(rank):
                for idx in range(n*m):
                    old = U[idx, r]
                    for new_val in vals:
                        if new_val == old: continue
                        U[idx, r] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            U[idx, r] = old
                            
            # Update V
            for r in range(rank):
                for idx in range(m*p):
                    old = V[idx, r]
                    for new_val in vals:
                        if new_val == old: continue
                        V[idx, r] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            V[idx, r] = old
                            
            # Update W
            for r in range(rank):
                for idx in range(p*n):
                    old = W[idx, r]
                    for new_val in vals:
                        if new_val == old: continue
                        W[idx, r] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            W[idx, r] = old
                            
            if not improved:
                break
                
        return U, V, W, best_err

    def search_rank(rank, num_trials=10):
        if time.time() > deadline - 10.0:
            return None
            
        np.random.seed(42 + rank)
        vals_init = [-1.0, 0.0, 1.0]
        
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
                
            U = np.random.choice(vals_init, size=(n * m, rank))
            V = np.random.choice(vals_init, size=(m * p, rank))
            W = np.random.choice(vals_init, size=(p * n, rank))
            
            U_final, V_final, W_final, final_err = discrete_local_search(
                U, V, W, rank, max_sweeps=5
            )
            
            if final_err < 1e-10:
                if verify_decomposition(U_final, V_final, W_final):
                    return (U_final, V_final, W_final)
                    
        return None

    # First, try the structural rank-18 decomposition
    structural = build_strassen_rank14()
    if structural is not None:
        U, V, W, rank = structural
        best_result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        best_rank = rank

    # Try to find even better decompositions
    for rank in [14, 15, 16, 17]:
        if time.time() > deadline - 15.0:
            break
        if rank >= best_rank:
            break
        result = search_rank(rank, num_trials=5)
        if result:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = rank
            break

    if best_rank == trivial_rank:
        return {
            "U": U_trivial.tolist(),
            "V": V_trivial.tolist(),
            "W": W_trivial.tolist()
        }
        
    return best_result
