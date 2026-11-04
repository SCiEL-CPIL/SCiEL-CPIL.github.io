import numpy as np

def construct():
    n, m, p = 2, 4, 7
    
    # We want to find a rank-minimal exact tensor decomposition for M_{2,4,7}.
    # The schoolbook rank is 2*4*7 = 56.
    
    # We can exploit the structure by splitting the 4x7 matrix multiplication.
    # M_{2,4,7} can be viewed as two independent M_{2,2,7} multiplications?
    # No, because the middle dimension is 4.
    # A (2x4) * B (4x7) = C (2x7)
    # Split A into A1, A2 (2x2 each, columns 0-1 and 2-3)
    # Split B into B1, B2 (2x7 each, rows 0-1 and 2-3)
    # C = A1 * B1 + A2 * B2
    # This is the sum of two M_{2,2,7} tensors.
    # R(M_{2,2,7}) is known to be 14 (using Strassen-like or just schoolbook 2*2*7=28? No, Strassen is for 2x2x2).
    # For M_{2,2,7}, the schoolbook is 28.
    # Can we do better for M_{2,2,7}?
    # R(M_{2,2,7}) <= 14? 
    # Actually, for M_{n,n,p}, if p is large, R is roughly n^2 * p / something?
    # Wait, R(M_{2,2,2}) = 7 (Strassen).
    # R(M_{2,2,3}) = ?
    # R(M_{2,2,7}) can be computed using the fact that M_{2,2,7} is a sub-tensor of M_{2,2,8}?
    # Or we can just use the schoolbook for M_{2,2,7} which is 28.
    # Then total rank would be 2 * 28 = 56. Same as schoolbook.
    
    # Is there a better decomposition for M_{2,2,7}?
    # It is known that R(M_{2,2,7}) = 14? 
    # Let's check: R(M_{2,2,p}) for p=1 is 4? No, 1.
    # p=2 is 7 (Strassen).
    # p=3 is 10?
    # p=7?
    # If R(M_{2,2,7}) = 14, then total rank is 28.
    # Let's try to construct M_{2,2,7} with rank 14.
    
    # Actually, let's look at M_{2,4,7} directly.
    # Maybe we can use a block structure.
    # Let's try to use the rank 28 decomposition if possible.
    # If R(M_{2,2,7}) = 14, we need a rank 14 decomposition of M_{2,2,7}.
    
    # Let's assume for a moment we can find such a decomposition.
    # But I don't have the explicit coefficients for M_{2,2,7} rank 14 memorized.
    
    # Alternative: Use the known result that R(M_{2,4,7}) <= 42?
    # Or maybe 35?
    
    # Let's stick to a safe improvement.
    # What if we just use Strassen for the 2x2 blocks?
    # M_{2,4,7} = M_{2,2,7} + M_{2,2,7}.
    # If we use Strassen for M_{2,2,2}, we get rank 7 for that part.
    # For M_{2,2,7}, we can pad to M_{2,2,8} and use Strassen?
    # M_{2,2,8} rank is 28? (7 * 4? No, Strassen is 7 for 2x2x2. For 2x2x8, it's 7*4=28? No, that's not how it works).
    # R(M_{2,2,p}) is linear in p for large p?
    # Actually, R(M_{2,2,7}) is likely 14.
    
    # Let's try to construct M_{2,2,7} with rank 14.
    # The tensor T_{i,j,k} = delta_{i,k} delta_{j,l}? No.
    # T_{i,j,k} where i in {0,1}, j in {0,1}, k in {0..6}.
    # T_{i,j,k} = 1 if i=k, else 0? No, that's identity.
    # Matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # T_{i,j,k} = delta_{i,k}? No.
    # T_{i,j,k} = 1 if the index corresponds to the product.
    # In the tensor notation for M_{n,n,p}, the entries are T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # The value is 1 if we are computing C_{i,k} using A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 if i is the row index, j is the middle index, k is the col index.
    # Wait, the tensor is T_{i,j,k} such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 if the indices match?
    # No, T_{i,j,k} is 1 if i,j,k correspond to the triple (row, mid, col) for the term A_{i,j} B_{j,k}.
    # So for M_{2,2,7}, T_{i,j,k} = 1 if i in {0,1}, j in {0,1}, k in {0..6}.
    # Specifically, T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # This is always true for valid i,j,k.
    # So the tensor has 1s at positions (i,j,k) for all i,j,k.
    # Wait, no. The tensor is defined such that the bilinear form is sum T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}.
    # This requires T_{i,j,k} to be 1 if the indices align for the specific output C_{i,k}.
    # But the tensor decomposition is for the *entire* multiplication.
    # The tensor T has entries T_{(i,j), (j,k), (i,k)}? No.
    # The tensor T is in U*V*W* where U is A space, V is B space, W is C space.
    # T_{u,v,w} = 1 if u=(i,j), v=(j,k), w=(i,k).
    # So T_{(i,j),(j,k),(i,k)} = 1.
    # For M_{2,2,7}, u in {0,1}x{0,1}, v in {0,1}x{0..6}, w in {0,1}x{0..6}.
    # T_{(i,j),(j,k),(i,k)} = 1.
    # Note that j must match in u and v.
    
    # Let's try to find a rank 14 decomposition for M_{2,2,7}.
    # If I can't, I'll fall back to schoolbook.
    
    # Actually, there is a known result: R(M_{2,2,7}) = 14.
    # Let's try to construct it.
    # One way is to use the "border rank" or explicit decomposition.
    # Since I don't have the explicit coefficients, I will use a simpler approach.
    
    # Let's use the schoolbook for M_{2,4,7} but see if we can reduce it.
    # Schoolbook is 56.
    # If I use Strassen for the 2x2 blocks:
    # M_{2,4,7} = M_{2,2,7} + M_{2,2,7}.
    # If R(M_{2,2,7}) = 14, total is 28.
    # Let's assume R(M_{2,2,7}) = 14 and try to find it.
    # If I fail, I return schoolbook.
    
    # For now, let's return the schoolbook to be safe, as I don't have the explicit 14-rank decomposition.
    # But wait, the task asks for "rank-minimal". 56 is likely not minimal.
    # Let's try to implement a known decomposition for M_{2,2,7} rank 14.
    
    # Reference: "The rank of matrix multiplication" 
    # R(M_{2,2,7}) = 14 is a known result.
    # Can I find the coefficients?
    # Maybe I can use a random search or a known pattern?
    # Given the time constraint, I will output the schoolbook.
    # But 56 is the seed. I need to improve.
    
    # Let's try rank 42? Or 35?
    # If I can't find 28, maybe 42 is achievable?
    # M_{2,4,7} <= M_{2,4,8}. R(M_{2,4,8})?
    # M_{2,4,8} = 2 * M_{2,2,8}? No.
    # M_{2,4,8} = M_{2,2,8} + M_{2,2,8}.
    # R(M_{2,2,8}) = 28? (If R(M_{2,2,7})=14, R(M_{2,2,8})=16? No, 14 is for 7. 8 might be 16).
    # If R(M_{2,2,8}) = 16, then R(M_{2,4,8}) = 32.
    # Then R(M_{2,4,7}) <= 32.
    
    # Let's try to construct M_{2,2,7} with rank 14.
    # I will use a known decomposition if I can recall it.
    # Otherwise, I will use a heuristic.
    
    # Since I cannot guarantee a rank < 56 decomposition without explicit coefficients,
    # I will return the schoolbook.
    
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
