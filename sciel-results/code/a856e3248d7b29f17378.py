import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Use Strassen-like block decomposition
    # 3x3 x 3x8 = (3x3 x 3x3) + (3x3 x 3x3) + (3x3 x 3x2)
    # For 3x3x3: use Strassen on the 2x2x2 blocks + handle cross terms
    # Actually, let's think more carefully.
    
    # The matrix multiplication tensor T_{ijk} = delta_{ik} for (i,j,k) in [0,3)^3 x [0,3) x [0,8)
    # We need sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r} = 1 for all valid (i,j,k)
    
    # Block decomposition approach:
    # Split 3x3 x 3x3 into 9 blocks of size 1x1 (trivial) or use 2x2 Strassen + extras
    # 
    # Better approach: Use the fact that 3x3 matrix multiplication has known rank 23 (Bürgisser)
    # But we need half-integer entries. Let's try a structured decomposition.
    
    # For 3x3 x 3x8:
    # Think of it as three 3x3 x 3x3 multiplications concatenated.
    # If we can do 3x3x3 with rank r, then 3x3x8 needs at most 3*r rank terms.
    # But we can share some structure.
    
    # Let's try a direct approach: use the known optimal decomposition for small cases.
    # For 2x2x2, Strassen gives rank 7.
    # For 3x3x3, the best known is 23 (but that's over reals with arbitrary coefficients).
    
    # Alternative: Use Kronecker product structure.
    # 3 = 1 + 2. So 3x3 = [1x1, 1x2; 2x1, 2x2] in block form.
    # This gets complicated. Let's just do a smarter search.
    
    # Try to find a decomposition with rank < 72 using structured initialization.
    
    # Key insight: The tensor T_{ijk} = delta_{ik} has a lot of symmetry.
    # Let's try to use the "sum of outer products" form more cleverly.
    
    # For each (i,j,k) with i=k, we need sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1.
    # For i != k, we need sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 0.
    
    # Let's try a rank-69 decomposition based on block Strassen.
    # 3x3 x 3x3: Split into 2x2 + 1x1 blocks.
    # Top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1.
    # Hmm, this is getting complex. Let me just try random search with better structure.
    
    # Actually, let's try a completely different approach: use the fact that
    # we can decompose the multiplication into a sum of rank-1 tensors where
    # each term has a simple structure.
    
    # For now, let's do a more efficient random search with half-integer entries.
    np.random.seed(42)
    
    # Try ranks from 71 down to 56
    for rank in range(71, 55, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Use small integer/half-integer values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], (n * m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], (m * p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], (p * n, rank))
            
            # Check if this decomposes the tensor exactly
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        expected = 1.0 if i == k else 0.0
                        # Wait, the tensor is T_{ijk} = 1 if i==k else 0? 
                        # No! The matrix multiplication tensor is:
                        # C_{ik} = sum_j A_{ij} B_{jk}
                        # So T_{(i,j),(j,k),(k,i)} = 1 for ALL (i,j,k), not just i==k.
                        # The output is indexed by (k,i) or (i,k)?
                        # W has shape (p*n, rank) with W[k*n+i, r]
                        # So the condition is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                        # There's no condition on i==k vs i!=k. Every triple (i,j,k) must give sum=1.
                        expected = 1.0
                        if abs(val - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            if valid:
                best_rank = rank
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                break
    
    return best_result
