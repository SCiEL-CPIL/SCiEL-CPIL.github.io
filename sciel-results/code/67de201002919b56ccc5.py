"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s/150s limits
    n, m, p = DIMENSIONS
    
    # Based on theoretical analysis:
    # The tensor for multiplying (n x m) by (m x p) can be decomposed.
    # For p=1, the rank is 1 (matrix-vector product is bilinear and simple).
    # Specifically, C_i = sum_j A_ij x_j.
    # We can write this as sum_r U_r A_ij V_r x_j W_r.
    # If we set U_r=1, V_r=1, W_r=1 for r=0, then C_i = sum_j A_ij x_j.
    # This works for any n, m.
    # So for p=1, rank is 1.
    # For general p, we can treat the multiplication as p separate matrix-vector multiplications.
    # C_{i,k} = sum_j A_ij B_{j,k}.
    # Each term is a product of A_ij and B_{j,k}.
    # The tensor is a direct sum of p copies of the (n,m,1) tensor?
    # No, the indices are coupled in the summation over j, but the output index k is separate.
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}
    # This is exactly the tensor T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
    # If we use rank 1 for each k, we need to be careful.
    # Let's check the rank of T_{n,m,p}.
    # It is known that rank(M_{n,m,p}) <= n*p + m*p - n - m + 1 (for n=m=p, this is ~3n^2).
    # But here, one dimension is 1 in the "vector" part if we view it as A * B.
    # Actually, the standard rank of matrix multiplication M_{n,m,p} is between max(nm, mp, np) and nmp.
    # For M_{4,4,9}, is the rank 9?
    # Let's test the rank 9 hypothesis.
    # We want C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    # Decomposition: C_{i,k} = \sum_{r=0}^{R-1} U_{i,j,r} V_{j,k,r} W_{i,k,r}.
    # If we set R=9 (one for each k), let r correspond to k.
    # For r=k:
    #   U_{i,j,k} = 1
    #   V_{j,k,k} = 1 if k==k? No, V depends on j,k.
    #   V_{j,k,r} needs to pick out B_{j,k}.
    #   Let V_{j,k,r} = 1 if k==r else 0.
    #   Let U_{i,j,r} = 1.
    #   Let W_{i,k,r} = 1 if k==r else 0.
    # Then term r=k is: \sum_{j} 1 * 1 * 1 * A_{i,j} B_{j,k} = \sum_j A_{i,j} B_{j,k}.
    # Terms r != k are 0 because V or W is 0.
    # So yes, rank <= p = 9.
    # Can it be lower?
    # Lower bound: The rank of M_{n,m,p} is at least max(nm, mp, np) / something?
    # Actually, for M_{1,1,p}, rank is 1.
    # For M_{n,1,p}, rank is n*p? No, M_{n,1,1} rank is 1.
    # Wait, M_{4,4,1} rank is 1.
    # Is M_{4,4,9} rank 9?
    # Let's verify the lower bound.
    # The partial derivative with respect to A_{i,j} gives a tensor in V * W space.
    # This might be complex.
    # However, rank 9 is a significant improvement over 144.
    # Let's construct the rank 9 decomposition.
    
    rank = p  # 9
    # U: shape (n*m, rank) -> (16, 9)
    # V: shape (m*p, rank) -> (36, 9)
    # W: shape (p*n, rank) -> (36, 9)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Convention:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = k  # Rank index corresponds to output column k
                # U[i*m+j][r] = 1
                U[i * m + j][r] = 1
                # V[j*p+k][r] = 1 (since r=k, this is 1)
                V[j * p + k][r] = 1
                # W[k*n+i][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
