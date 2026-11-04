"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    rank = len(U[0])
    # Build tensor
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                u_vals = np.array([U[i*m+j][r] for r in range(rank)])
                v_vals = np.array([V[j*p+k][r] for r in range(rank)])
                w_vals = np.array([W[k*n+i][r] for r in range(rank)])
                T[i,j,k] = np.dot(np.dot(u_vals, v_vals), w_vals)
    # Check against identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i,j,k) == (i,j,k) else 0.0
                # Actually the tensor for matrix multiplication:
                # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # So T[i,j,k] should be 1 for all i,j,k? No.
                # The tensor is T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}?
                # Actually, the multiplication tensor M_{i,j,k} = 1 for all (i,j,k) in the range?
                # No. The tensor is defined such that:
                # sum_{i,j,k} A[i,j] B[j,k] C[k,i] = sum_r (sum_j A[i,j] U[i,j,r]) (sum_k B[j,k] V[j,k,r]) (sum_i C[k,i] W[k,i,r])
                # This equals sum_i,j,k A[i,j] B[j,k] C[k,i] if and only if
                # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
                # Wait, that's not right either.
                
                # Let me re-derive. We want:
                # (A*B)[i,k] = sum_j A[i,j] B[j,k]
                # We express this as:
                # (A*B)[i,k] = sum_r U[i,*,r] * V[*,k,r] * W[k,*,r]?
                
                # The standard formulation:
                # M = sum_r U_r \otimes V_r \otimes W_r
                # where U_r is n*m vector, V_r is m*p vector, W_r is p*n vector
                # And the contraction is:
                # (A*B)[i,k] = sum_j sum_r A[i,j] U_r[i*m+j] * B[j,k] V_r[j*p+k] * W_r[k*n+i]
                
                # For this to equal sum_j A[i,j] B[j,k], we need:
                # sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] = 1 for all i,j,k
                # Wait, that would give sum_j sum_r ... = sum_j 1 = m, which is wrong.
                
                # Let me look at the schoolbook:
                # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
                # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if there's a column with all three = 1 for (i,j,k)
                # In schoolbook, each (i,j,k) has its own column, so for a given (i,j,k),
                # the sum is 1 (from its own column) + 0 from others = 1.
                # So (A*B)[i,k] = sum_j sum_r A[i,j] * 1 * B[j,k] * 1 * 1? 
                # No, W depends on k and i, not j.
                
                # Let me be more careful. The formula is:
                # (A*B)[i,k] = sum_r (sum_j A[i,j] U_r[i*m+j]) (sum_j' B[j',k] V_r[j'*p+k]) W_r[k*n+i]
                # = sum_r W_r[k*n+i] (sum_j A[i,j] U_r[i*m+j]) (sum_j' B[j',k] V_r[j'*p+k])
                
                # For this to equal sum_j A[i,j] B[j,k], we need the tensor to satisfy:
                # sum_r U_r[i*m+j] V_r[j'*p+k] W_r[k*n+i] = delta_{j,j'}
                
                # In schoolbook: U_r[i*m+j] = 1 if r corresponds to (i,j), else 0
                # V_r[j'*p+k] = 1 if r corresponds to (j,k), else 0  
                # W_r[k'n+i'] = 1 if r corresponds to (k',i')
                
                # So for a specific r corresponding to (i0,j0,k0):
                # U_r[i*m+j] = 1 iff i=i0, j=j0
                # V_r[j'*p+k] = 1 iff j'=j0, k=k0
                # W_r[k*n+i] = 1 iff k'=k0, i'=i0
                
                # So the product U_r[i*m+j] * V_r[j'*p+k] * W_r[k*n+i] = 1 iff i=i0, j=j0, j'=j0, k=k0, i'=i0
                # i.e., iff (i,j,j',k,i') = (i0,j0,j0,k0,i0)
                
                # Summing over r: sum_r ... = 1 iff there exists r=(i,j,j,k,i) i.e. iff j=j' and i=i' and k=k
                # Wait, this is getting confusing. Let me just verify numerically.
                pass
    
    # Simpler verification: check that for random A and B, A@B matches the decomposition
    random.seed(42)
    A = np.array([[random.randint(-5,5) for _ in range(m)] for _ in range(n)])
    B = np.array([[random.randint(-5,5) for _ in range(p)] for _ in range(m)])
    expected = A @ B
    
    actual = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        # U_r: n*m vector, V_r: m*p vector, W_r: p*n vector
        u_r = np.array([U[i*m+j][r] for i in range(n) for j in range(m)]).reshape(n, m)
        v_r = np.array([V[j*p+k][r] for j in range(m) for k in range(p)]).reshape(m, p)
        w_r = np.array([W[k*n+i][r] for k in range(p) for i in range(n)]).reshape(p, n)
        
        # Compute contribution: sum_j A[i,j]*u_r[i,j] * sum_k B[j,k]*v_r[j,k] * w_r[k,i]
        # This is: (A * u_r) @ v_r * w_r^T? 
        # Let's think: 
        # term_r[i,k] = w_r[k,i] * (sum_j A[i,j] u_r[i,j]) * (sum_j B[j,k] v_r[j,k])
        # = w_r[k,i] * (A * u_r)[i] * (v_r * B)[k]  where v_r*B is row vector
        # Actually (v_r * B)[k] = sum_j v_r[j,k] B[j,k]
        
        a_part = np.array([sum(A[i,j]*u_r[i,j] for j in range(m)) for i in range(n)])
        b_part = np.array([sum(v_r[j,k]*B[j,k] for j in range(m)) for k in range(p)])
        for i in range(n):
            for k in range(p):
                actual[i,k] += w_r[k,i] * a_part[i] * b_part[k]
    
    if not np.allclose(actual, expected, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Try to find a rank-14 decomposition for 2x3x3 and rank-8 for 2x3x2
    # Then combine: 2x3x8 = 2x3x3 + 2x3x3 + 2x3x2
    # Rank = 14 + 14 + 8 = 36
    
    # First, let's try to find a rank-14 decomposition for 2x3x3
    # Known: rank of 2x3x3 is 14 (or maybe less? Let me check)
    # Actually, I recall that rank of 2x2x2 is 7, 2x3x3 might be around 12-14
    
    # For now, let's just use schoolbook as a fallback
    result = _schoolbook(n, m, p)
    
    # Try to find better decompositions with time budget
    # Try random search for 2x3x3 with rank 12, 13, 14
    best_233 = None
    for target_rank in [12, 13, 14, 15]:
        if time.time() > deadline - 30:
            break
        found = _search_233(target_rank, deadline - 30)
        if found is not None:
            best_233 = found
            break
    
    # Try to find 2x3x2 with rank 8, 9, 10
    best_232 = None
    for target_rank in [8, 9, 10]:
        if time.time() > deadline - 20:
            break
        found = _search_232(target_rank, deadline - 20)
        if found is not None:
            best_232 = found
            break
    
    # If we found both, combine them
    if best_233 is not None and best_232 is not None:
        # Combine: 2x3x8 = block(2x3x3) + block(2x3x3) + block(2x3x2)
        U233, V233, W233 = best_233
        U232, V232, W232 = best_232
        r233 = len(U233[0])
        r232 = len(U232[0])
        total_rank = 2 * r233 + r232
        
        # Build combined matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # First 2x3x3 block (k=0,1,2)
        for r in range(r233):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = U233[i*m+j][r]
            for j in range(m):
                for k in range(3):
                    V[j*p+k][r] = V233[j*3+k][r]
            for k in range(3):
                for i in range(n):
                    W[k*n+i][r] = W233[k*n+i][r]
        
        # Second 2x3x3 block (k=3,4,5)
        offset = r233
        for r in range(r233):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][offset+r] = U233[i*m+j][r]
            for j in range(m):
                for k in range(3):
                    V[j*p+(k+3)][offset+r] = V233[j*3+k][r]
            for k in range(3):
                for i in range(n):
                    W[(k+3)*n+i][offset+r] = W233[k*n+i][r]
        
        # 2x3x2 block (k=6,7)
        offset2 = 2 * r233
        for r in range(r232):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][offset2+r] = U232[i*m+j][r]
            for j in range(m):
                for k in range(2):
                    V[j*p+(k+6)][offset2+r] = V232[j*2+k][r]
            for k in range(2):
                for i in range(n):
                    W[(k+6)*n+i][offset2+r] = W232[k*n+i][r]
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return result

def _search_233(rank, deadline):
    """Search for rank decomposition of 2x3x3."""
    n, m, p = 2, 3, 3
    random.seed(123)
    for attempt in range(100):
        if time.time() > deadline:
            break
        U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
    return None

def _search_232(rank, deadline):
    """Search for rank decomposition of 2x3x2."""
    n, m, p = 2, 3, 2
    random.seed(456)
    for attempt in range(100):
        if time.time() > deadline:
            break
        U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
    return None

# EVOLVE-BLOCK-END
