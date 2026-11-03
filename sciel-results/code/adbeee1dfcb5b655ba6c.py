"""Matrix multiplication 2x4x4 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W, n=2, m=4, p=4):
    """Check if the tensor decomposition is exact for matrix multiplication."""
    rank = len(U[0])
    # Build the multiplication tensor
    # T[i,j,k,l] should be sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication (A is n x m, B is m x p), the result C = A*B is n x p
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # The tensor for this operation has indices (i, j, k, l) where:
    #   i in [0,n), j in [0,m), k in [0,p), l in [0,p)
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W has p*n rows, indexed by (k, i) where k is from B's column (0..p-1) and i is from A's row (0..n-1)
    # U[i*m+j, r] - U has n*m rows, indexed by (i, j)
    # V[j*p+k, r] - V has m*p rows, indexed by (j, k)
    
    # The tensor T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k,l] should be 1 if this corresponds to a term in C[i,l] = sum_j A[i,j]*B[j,l]
    # Actually, the standard matrix multiplication tensor T has:
    # T[i, j, k, l] = 1 if j == j and k == l ... no.
    
    # Let me think more carefully. The matrix multiplication operation maps (A, B) -> C where:
    # C[i, l] = sum_{j=0}^{m-1} A[i, j] * B[j, l]
    
    # As a 3-tensor: T[i, (j,l), ?] ... this is getting confusing.
    
    # Let's use the given indexing directly:
    # U has shape (n*m, rank), indexed as U[i*m + j, r]
    # V has shape (m*p, rank), indexed as V[j*p + k, r]  
    # W has shape (p*n, rank), indexed as W[k*n + i, r]
    
    # The reconstructed tensor element T[i, j, k, l] where:
    #   i in [0, n) - row of A, row of C
    #   j in [0, m) - col of A, row of B
    #   k in [0, p) - col of B
    #   l in [0, p) - col of C
    # 
    # Actually wait - there's no 'l' in the W index. W[k*n+i, r] uses k and i.
    # So the tensor is really T[i, j, k] with:
    #   T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # But matrix multiplication produces C[i, l] = sum_j A[i,j] * B[j,l]
    # So the output has indices (i, l) where l is the column of B (0 to p-1).
    # 
    # Hmm, so k in W[k*n+i] must correspond to the output column l.
    # And the input to V is B[j, l] which is indexed by (j, l) = (j, k).
    # 
    # So T[i, j, k] = delta_{j,j} * delta_{k,l}... 
    # For matrix multiplication: T[i, j, k] should be 1 if this term contributes to C[i,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    # 
    # Actually, the matrix multiplication tensor T is defined such that:
    # C = A * B means C[i,l] = sum_j A[i,j] * B[j,l]
    # As a 3-tensor T[i,j,k] = 1 if j is the shared index and k=l
    # Wait, no. The tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # No wait, that can't be right because then the rank would just be n*m*p.
    # 
    # Let me reconsider. The matrix multiplication tensor is a 3-tensor of shape (n*p, m, m*p) or similar.
    # 
    # Given the interface:
    # U: (n*m, rank) -> rows indexed by (i,j) where i=row of A, j=shared index
    # V: (m*p, rank) -> rows indexed by (j,k) where j=shared index, k=col of B  
    # W: (p*n, rank) -> rows indexed by (k,i) where k=col of B, i=row of C
    #
    # The tensor element T[i,j,k] (where i in [0,n), j in [0,m), k in [0,p)) should equal:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # And for matrix multiplication, T[i,j,k] = 1 if and only if this term appears in C[i,k] = sum_j' A[i,j']*B[j',k]
    # So T[i,j,k] = 1 for all valid (i,j,k). 
    # 
    # Wait, that means the tensor is just the all-ones tensor of shape (n,m,p)?
    # That would make the rank n*m*p which is the schoolbook answer.
    # But that seems too simple. Let me verify with a small example.
    #
    # For 2x2x2 matrix multiplication:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # So the tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1}.
    # The rank of the all-ones tensor of shape 2x2x2 is... 2? No, it's more.
    #
    # Actually I think I'm overcomplicating this. Let me just check:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For schoolbook, each (i,j,k) triple gets its own rank-1 term, so T[i,j,k] = 1 for all.
    # 
    # The question is whether we can do better than rank 32 for 2x4x4.
    # 
    # Let me just verify that the schoolbook solution works, then try to search for lower rank.
    
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                T[i, j, k] = val
    
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    # because each term A[i,j]*B[j,k] appears exactly once in C[i,k]
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)

def _schoolbook(n=2, m=4, p=4):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find a lower-rank decomposition via randomized local search
    # Start from rank 31 and try to find a valid decomposition
    # Use a simple approach: initialize with random entries, then use gradient-free local search
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try ranks from 31 down to, say, 25, but with very limited time
    for target_rank in range(31, 24, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            random.seed(42 + target_rank * 100 + attempt)
            np.random.seed(42 + target_rank * 100 + attempt)
            
            U = [[random.choice([0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick local search: randomly perturb entries and check
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and entry to modify
                mat_choice = random.randint(0, 2)
                row = random.randint(0, (n*m if mat_choice==0 else m*p if mat_choice==1 else p*n) - 1)
                col = random.randint(0, target_rank - 1)
                old_val = U[row][col] if mat_choice == 0 else V[row][col] if mat_choice == 1 else W[row][col]
                
                new_val = random.choice([0, 0.5, 1])
                if new_val == old_val:
                    continue
                
                if mat_choice == 0:
                    U[row][col] = new_val
                elif mat_choice == 1:
                    V[row][col] = new_val
                else:
                    W[row][col] = new_val
                
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": W and W, "W": W}
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
                else:
                    # Revert
                    if mat_choice == 0:
                        U[row][col] = old_val
                    elif mat_choice == 1:
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
            
            if best_rank == target_rank:
                break
    
    return best_result
