"""Matrix multiplication 3,3,5 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    # Schoolbook gives rank 45. We attempt to find a lower-rank decomposition.
    # 
    # The tensor T[i,j,k,l] = 1 if j==k, for i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # We want rank-r factors U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k,l]  ... wait, W is indexed by [k*n+i]
    # Actually the convention is W[k*n+i, r] so the output index is (k, i) -> k*n+i
    # But for matrix multiplication C = A*B, C[i,l] = sum_j A[i,j]*B[j,l]
    # So the tensor entry is T[i,j,l,k] where k is the contraction index... 
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # This means W is indexed by [k*n + i] where k and i are... 
    # For (n,m,p) matrix mult: A is n x m, B is m x p, C is n x p
    # U has shape (n*m, rank), indexed by [i*m+j]
    # V has shape (m*p, rank), indexed by [j*p+k]  
    # W has shape (p*n, rank), indexed by [k*n+i] where k is the column of C (0..p-1) and i is the row (0..n-1)
    # So W[k*n+i, c] corresponds to output C[i,k]
    # The decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if j==k else 0
    # Wait, that's not right either. Let me think again.
    # 
    # Standard tensor decomposition for matrix mult (n,m,p):
    # T[i,j,k,l] = delta_{j,k} where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # But here the dimensions are n=3, m=3, p=5, and the tensor is over (i,j) for A, (j,k) for B, (i,k) for C
    # where i in [0,3), j in [0,3), k in [0,5)
    # T[i,j,k] = delta_{j,j'} ... no.
    # 
    # A[i,j] * B[j,k] = C[i,k], summing over j in [0,3)
    # So the tensor has free indices i (row of A/C, 0..2), j (col of A / row of B, 0..2), k (col of B/C, 0..4)
    # T[i,j,k] = 1 (it's a trilinear form)
    # The decomposition: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k] = 1 for all i,j,k
    # Wait no, that's not how it works. The tensor is T[i,j,k] and we want:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 for all valid i,j,k? No!
    # 
    # Actually for matrix multiplication, the tensor is:
    # T[i, j, k] = 1 if the indices are consistent, which they always are in the product
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: sum_{i,j,k} A[i,j] B[j,k] C'[i,k] where C' is the "test" matrix
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # So we need: for all i in [0,3), j in [0,3), k in [0,5):
    # sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1
    #
    # This is a system of 3*3*5 = 45 equations.
    #
    # Let me try to find a rank-r solution. Start with r=45 (schoolbook) and try to reduce.
    
    deadline = time.time() + 130  # safety margin
    
    # Try to find a low-rank decomposition using a greedy/iterative approach
    # First, let's set up the problem as finding U, V, W with r columns
    
    def verify(U, V, W, n=3, m=3, p=5):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook solution
    rank = 45
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(15)]
    W = [[0]*rank for _ in range(15)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U[i*3+j][col] = 1
                V[j*5+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = 45
    
    # Try to reduce rank by combining terms
    # For a fixed (i,k) pair, the constraint is:
    # sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # 
    # Let's try rank 30, 25, 20, 15, etc.
    
    import random
    random.seed(42)
    
    for target_rank in [40, 35, 30, 27, 25, 23, 20, 18, 15]:
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            # Initialize with random values
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(15)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(15)]
            
            # Gradient descent to minimize sum of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 20:
                    break
                # Compute gradients
                grad_U = [[0.0]*target_rank for _ in range(9)]
                grad_V = [[0.0]*target_rank for _ in range(15)]
                grad_W = [[0.0]*target_rank for _ in range(15)]
                
                for i in range(3):
                    for j in range(3):
                        for k in range(5):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*3+j][c] * V[j*5+k][c] * W[k*3+i][c]
                            err = s - 1.0
                            for c in range(target_rank):
                                grad_U[i*3+j][c] += err * V[j*5+k][c] * W[k*3+i][c]
                                grad_V[j*5+k][c] += err * U[i*3+j][c] * W[k*3+i][c]
                                grad_W[k*3+i][c] += err * U[i*3+j][c] * V[j*5+k][c]
                
                for i in range(9):
                    for c in range(target_rank):
                        U[i][c] -= lr * grad_U[i][c]
                for i in range(15):
                    for c in range(target_rank):
                        V[i][c] -= lr * grad_V[i][c]
                        W[i][c] -= lr * grad_W[i][c]
                
                if step % 50 == 0 and verify(U, V, W):
                    found = True
                    break
            
            if verify(U, V, W):
                found = True
                break
        
        if found:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
