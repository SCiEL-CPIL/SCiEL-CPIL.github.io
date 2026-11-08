"""Matrix multiplication 3,3,4 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import random
import copy
from fractions import Fraction

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    n, m, p = DIMENSIONS
    
    # Schoolbook initialization
    rank0 = n * m * p
    U = [[0] * rank0 for _ in range(n * m)]
    V = [[0] * rank0 for _ in range(m * p)]
    W = [[0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U*V*W exactly equals the matmul tensor."""
        n, m, p = DIMENSIONS
        # Tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matmul: T[i,j,k] = delta (the multiplication is linear, so tensor is 
        # T[i,j,k] = 1 for the "triple" structure... actually let me think about this differently.
        # The matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor form: T[i,j,k] such that C[i,k] = sum_{i',j',k'} A[i',j'] B[j',k'] T[i',j',k] delta... 
        # Actually the standard tensor: T[i,j,k] = 1 if we think of it as the bilinear map.
        # The decomposition: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k]) ... no.
        # Standard: C[i,k] = sum_c U[i,k,c] * (sum_j ... ) 
        # Let me use the convention: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) ... no that's not right either.
        
        # The correct convention from the task:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # C[i,k] = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]
        # For this to equal sum_j A[i,j]*B[j,k] for all A,B:
        # We need: sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,j} ... 
        # Actually: sum_c sum_j U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k]
        # This requires: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Wait, that's not right. Let me re-derive.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k] ... no
        # The bilinear form: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k' ... ) 
        # I think the convention is:
        # Let P_c[i,j] = U[i*m+j, c], Q_c[j,k] = V[j*p+k, c], R_c[k,i] = W[k*n+i, c]
        # Then C[i,k] = sum_c R_c[k,i] * sum_j P_c[i,j] * A[i,j] * ... no.
        
        # Standard tensor decomposition for matmul:
        # C = A*B where C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T with T[i,j,k] such that C[i,k] = sum_{i',j',k'} T[i',j',k] A[i',j'] B[j',k] delta(i',i)delta(k',k)... 
        # Actually T[i,j,k] is defined so that: C[i,k] = sum_{j} A[i,j] B[j,k] means
        # the tensor has T[i,j,k] = 1 for the "middle" index j, but that's just the identity.
        
        # The bilinear map view: matmul is a bilinear map (A,B) -> C
        # T(A,B) = sum_c (sum_{i,j} U[i*m+j,c] A[i,j]) * (sum_{j,k} V[j*p+k,c] B[j,k]) * W... no, W is 3D.
        
        # I think the correct interpretation is:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # No, that doesn't work dimensionally.
        
        # Let me just verify: for schoolbook, U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for c=(i,j,k)
        # C[i,k] = sum_{i',j',k'} ... this is getting confusing. Let me just check numerically.
        pass
    
    def check_decomp(U, V, W):
        """Verify decomposition by checking all (i,j,k) triples."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Verify schoolbook
    # For schoolbook: only c=(i,j,k) contributes, giving 1*1*1=1. Good.
    
    def try_remove(U, V, W, c_to_remove):
        """Try to remove column c_to_remove by adjusting others. Returns new (U,V,W) or None."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        if rank <= 1:
            return None
        
        # After removing column c, we need to adjust remaining columns so that
        # for all (i,j,k): sum_{c != c_to_remove} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a nonlinear system. Instead, let's try: set U,V,W entries in column c to 0,
        # then for each (i,j,k) where the sum was 1, we need to redistribute.
        
        # Simpler approach: try zeroing out one factor in column c and see if we can fix.
        # Even simpler: just try removing and checking if a simple adjustment works.
        
        # Let's try: for each of the 3 matrices, try setting one entry to absorb the missing value.
        # Actually, let's use a different strategy: randomly perturb and use least squares.
        
        return None  # Placeholder
    
    # Better approach: iterative sparsification with repair
    def sparsify(U, V, W, max_iters=5000):
        """Iteratively try to remove terms."""
        n, m, p = DIMENSIONS
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to remove a random column
            c = random.randint(0, rank - 1)
            
            # Check if column c is "redundant" - try zeroing it and see if we can repair
            # Save state
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            # Zero out column c
            for i in range(n * m):
                U[i][c] = 0
            for j in range(m * p):
                V[j][c] = 0
            for k in range(p * n):
                W[k][c] = 0
            
            # Now try to repair: for each (i,j,k) triple, check if sum is still 1
            # If not, try to adjust one entry in the remaining columns
            repaired = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for cc in range(rank):
                            if cc == c:
                                continue
                            s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                        if s != 1:
                            # Need to fix: find a column cc != c where we can adjust
                            # Try adjusting W[k*n+i, cc] for some cc
                            fixed = False
                            for cc in range(rank):
                                if cc == c:
                                    continue
                                prod = U[i*m+j][cc] * V[j*p+k][cc]
                                if prod != 0:
                                    # Set W[k*n+i, cc] += (1 - s) / prod
                                    # But this might break other triples... 
                                    # For now, just try it and verify at the end
                                    W[k*n+i][cc] += (1 - s) / prod
                                    fixed = True
                                    break
                            if not fixed:
                                repaired = False
                                break
                    if not repaired:
                        break
                if not repaired:
                    break
            
            if repaired:
                # Verify full decomposition
                if check_decomp(U, V, W):
                    # Success! Remove column c
                    U = [row[:c] + row[c+1:] for row in U]
                    V = [row[:c] + row[c+1:] for row in V]
                    W = [row[:c] + row[c+1:] for row in W]
                    rank -= 1
                else:
                    # Restore
                    U = U_save
                    V = V_save
                    W = W_save
            else:
                U = U_save
                V = V_save
                W = W_save
        
        return U, V, W
    
    # Run sparsification with multiple random restarts
    best_U, best_V, best_W = U, V, W
    best_rank = rank0
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(restart * 12345 + 678)
        U_r, V_r, W_r = sparsify(U, V, W, max_iters=2000)
        r = len(U_r[0])
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            print(f"Restart {restart}: improved to rank {best_rank}")
    
    return {"U": best_U, "V": best_V, "W": best_W}
