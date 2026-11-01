import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Rank 7 Strassen decomposition
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U indices: i*2 + j
    # V indices: j*2 + k
    # W indices: k*2 + i
    
    # Product 1: (a00 + a11)(b00 + b11)
    # U: a00 (0), a11 (3)
    # V: b00 (0), b11 (3)
    # W: c00 (0), c11 (3)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # Product 2: (a10 + a11)b00
    # U: a10 (2), a11 (3)
    # V: b00 (0)
    # W: c10 (1), c11 (3) with sign -1 for c11
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    W[3][1] = -1
    
    # Product 3: a00(b01 - b11)
    # U: a00 (0)
    # V: b01 (1), b11 (3) with sign -1
    # W: c01 (2), c11 (3)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] = 1
    W[3][2] = 1
    
    # Product 4: a11(b10 - b00)
    # U: a11 (3)
    # V: b10 (2), b00 (0) with sign -1
    # W: c00 (0), c10 (1)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[1][3] = 1
    
    # Product 5: (a00 + a01)b11
    # U: a00 (0), a01 (1)
    # V: b11 (3)
    # W: c00 (0) with sign -1, c01 (2)
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[2][4] = 1
    
    # Product 6: (a10 - a00)(b00 + b01)
    # U: a10 (2), a00 (0) with sign -1
    # V: b00 (0), b01 (1)
    # W: c11 (3)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # Product 7: (a01 - a11)(b10 + b11)
    # U: a01 (1), a11 (3) with sign -1
    # V: b10 (2), b11 (3)
    # W: c00 (0)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    
    # Verify the decomposition exactly
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check that the tensor decomposition is correct
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For matrix multiplication: T[i,j,k] = delta_{i,k} (standard identity)
        # Wait, let me reconsider the tensor structure.
        
        # The matrix multiplication tensor M_{i,j,k} where:
        # (AB)_{ik} = sum_j A_{ij} B_{jk}
        # So M_{i,j,k} = 1 if the multiplication involves A_{ij} * B_{jk} = C_{ik}
        # Actually, the tensor is M_{i,j,k} = delta_{i,k} in some conventions,
        # but more precisely, the multiplication tensor has entries:
        # T_{(i,j),k,?} ... let me think more carefully.
        
        # The standard tensor decomposition of matrix multiplication:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # This can be written as a sum of rank-1 tensors.
        # Each rank-r term contributes: U_{(i,j),r} * V_{j',r} ... 
        # 
        # Actually, the convention here is:
        # U has shape (n*m, rank) = (4, rank), indexed by (i,j) -> i*2+j
        # V has shape (m*p, rank) = (4, rank), indexed by (j,k) -> j*2+k  
        # W has shape (p*n, rank) = (4, rank), indexed by (k,i) -> k*2+i
        #
        # The decomposition satisfies:
        # sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = delta_{i,k} ... no
        #
        # Actually for matrix multiplication C = A * B:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor is: sum_j A_{ij} B_{jk} C_{ik}^* (dual)
        # So T_{(i,j),(j,k),(k,i)} = 1 for each valid (i,j,k) triple
        #
        # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T_{(i,j),(j,k),(k,i)}
        # where T_{(i,j),(j,k),(k,i)} = 1 iff this term appears in the sum, which is always
        # the case for matrix multiplication. So T = 1 for all valid indices.
        
        # Let's verify: for each (i,j,k), sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] should equal 1
        
        max_err = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(rank):
                        s += U_np[i*2+j, r] * V_np[j*2+k, r] * W_np[k*2+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fallback: use trivial rank-8 decomposition if Strassen fails
            rank = 8
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(4)]
            W = [[0] * rank for _ in range(4)]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = i * 4 + j * 2 + k
                        if r < rank:
                            U[i*2+j][r] = 1
                            V[j*2+k][r] = 1
                            W[k*2+i][r] = 1
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
