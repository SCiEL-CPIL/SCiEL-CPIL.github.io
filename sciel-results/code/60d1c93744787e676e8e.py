"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin
    
    # Recursive decomposition
    def decompose(n, m, p):
        # Returns U, V, W for n x m * m x p
        # U: n*m x rank
        # V: m*p x rank
        # W: p*n x rank
        
        # Base case: Schoolbook
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            # U, V, W for Strassen
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A10)B11
            # M6 = (A11+A01)(B00+B10)
            # M7 = (A01+A11)(B01+B11)
            
            # Indices for A: (i,j) -> i*m+j
            # Indices for B: (j,k) -> j*p+k
            # Indices for C: (i,k) -> k*n+i
            
            U = [
                [1, 0, 0, 0, 0, 0, 0], # A00
                [0, 0, 0, 0, 0, 0, 0], # A01
                [0, 0, 0, 0, 0, 0, 0], # A10
                [1, 1, 1, 1, 1, 1, 1], # A11
            ]
            # Wait, Strassen uses combinations.
            # U should have rows for A indices.
            # The rank-1 tensors are defined by the products.
            # We need to map the Strassen products to the U, V, W format.
            
            # Product 1: (A00+A11) * (B00+B11)
            # U col 0: A00=1, A11=1. Others 0.
            # V col 0: B00=1, B11=1. Others 0.
            # W col 0: C00+=1, C11+=1.
            
            # This is getting complex to hardcode.
            # I will use a generic schoolbook for base cases to ensure correctness,
            # and only use Strassen if I can easily construct it.
            # Given the complexity, I will stick to schoolbook for all base cases
            # and rely on the split for improvement.
            pass

        # Splitting logic
        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
        
        # Try splitting n
        if n >= 2 and time.time() < deadline:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = decompose(n1, m, p)
                U2, V2, W2 = decompose(n2, m, p)
                
                # Combine
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                new_rank = rank1 + rank2
                
                if new_rank < best_rank:
                    # U1 is n1*m x rank1. U2 is n2*m x rank2.
                    # New U is (n1+m)*m? No, n*m.
                    # U rows are indexed by i*m+j.
                    # U1 covers i in [0, n1). U2 covers i in [n1, n).
                    # We need to offset the indices for U2?
                    # No, U is just a matrix. We stack them vertically?
                    # U shape: (n*m, rank).
                    # U1 shape: (n1*m, rank1).
                    # U2 shape: (n2*m, rank2).
                    # New U: 
                    # Rows 0..n1*m-1 come from U1.
                    # Rows n1*m..n*m-1 come from U2.
                    
                    new_U = [row[:] for row in U1]
                    for row in U2:
                        new_U.append(row)
                    
                    # V is m*p x rank.
                    # V1 and V2 have same shape (m*p, rank).
                    # We stack them horizontally.
                    new_V = []
                    for r in range(len(V1)):
                        new_V.append(V1[r] + V2[r])
                    
                    # W is p*n x rank.
                    # W1 shape: (p*n1, rank1).
                    # W2 shape: (p*n2, rank2).
                    # W indices are k*n+i.
                    # W1 covers i in [0, n1).
                    # W2 covers i in [n1, n).
                    # In the combined W, the index for i is k*n + i.
                    # For W1, the index is k*n1 + i.
                    # We need to map k*n1 + i to k*n + i.
                    # This is not a simple stack.
                    # The W matrix rows are indexed by (k, i).
                    # W1 has rows for i < n1.
                    # W2 has rows for i >= n1.
                    # We need to interleave or permute?
                    # The output C is n x p.
                    # The tensor W is p x n.
                    # The decomposition sums over rank.
                    # The W matrix for the combined problem must have rows for all i.
                    # Row (k, i) of W should be the combination of W1 and W2.
                    # If i < n1, it comes from W1.
                    # If i >= n1, it comes from W2.
                    
                    new_W = [ [0] * new_rank for _ in range(p * n) ]
                    for k in range(p):
                        for i in range(n1):
                            # W1 index: k * n1 + i
                            # New index: k * n + i
                            new_W[k * n + i] = W1[k * n1 + i] + [0] * rank2
                        for i in range(n1, n):
                            # W2 index: k * n2 + (i - n1)
                            # New index: k * n + i
                            new_W[k * n + i] = [0] * rank1 + W2[k * n2 + (i - n1)]
                    
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W
        
        # Try splitting m
        if m >= 2 and time.time() < deadline:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = decompose(n, m1, p)
                U2, V2, W2 = decompose(n, m2, p)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                new_rank = rank1 + rank2
                
                if new_rank < best_rank:
                    # U: n*m x rank.
                    # U1: n*m1 x rank1.
                    # U2: n*m2 x rank2.
                    # U indices: i*m + j.
                    # U1: j < m1.
                    # U2: j >= m1.
                    
                    new_U = [ [0] * new_rank for _ in range(n * m) ]
                    for i in range(n):
                        for j in range(m1):
                            new_U[i * m + j] = U1[i * m1 + j] + [0] * rank2
                        for j in range(m1, m):
                            new_U[i * m + j] = [0] * rank1 + U2[i * m2 + (j - m1)]
                    
                    # V: m*p x rank.
                    # V1: m1*p x rank1.
                    # V2: m2*p x rank2.
                    # V indices: j*p + k.
                    
                    new_V = [ [0] * new_rank for _ in range(m * p) ]
                    for j in range(m1):
                        for k in range(p):
                            new_V[j * p + k] = V1[j * p + k] + [0] * rank2
                    for j in range(m1, m):
                        for k in range(p):
                            new_V[j * p + k] = [0] * rank1 + V2[(j - m1) * p + k]
                    
                    # W: p*n x rank.
                    # W1, W2 have same shape (p*n, rank).
                    new_W = []
                    for r in range(len(W1)):
                        new_W.append(W1[r] + W2[r])
                    
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting p
        if p >= 2 and time.time() < deadline:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = decompose(n, m, p1)
                U2, V2, W2 = decompose(n, m, p2)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                new_rank = rank1 + rank2
                
                if new_rank < best_rank:
                    # U: n*m x rank.
                    # U1, U2 have same shape.
                    new_U = []
                    for r in range(len(U1)):
                        new_U.append(U1[r] + U2[r])
                    
                    # V: m*p x rank.
                    # V1: m*p1 x rank1.
                    # V2: m*p2 x rank2.
                    # V indices: j*p + k.
                    
                    new_V = [ [0] * new_rank for _ in range(m * p) ]
                    for j in range(m):
                        for k in range(p1):
                            new_V[j * p + k] = V1[j * p1 + k] + [0] * rank2
                        for k in range(p1, p):
                            new_V[j * p + k] = [0] * rank1 + V2[j * p2 + (k - p1)]
                    
                    # W: p*n x rank.
                    # W1: p1*n x rank1.
                    # W2: p2*n x rank2.
                    # W indices: k*n + i.
                    
                    new_W = [ [0] * new_rank for _ in range(p * n) ]
                    for k in range(p1):
                        for i in range(n):
                            new_W[k * n + i] = W1[k * n + i] + [0] * rank2
                    for k in range(p1, p):
                        for i in range(n):
                            new_W[k * n + i] = [0] * rank1 + W2[(k - p1) * n + i]
                    
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W

        # If no split improved, use schoolbook
        if best_U is None:
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
            return U, V, W

        return best_U, best_V, best_W

    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
