import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
class TensorDecomposition:
    def __init__(self):
        self.deadline = time.time() + 110

    def construct(self, n, m, p):
        # For 3x5x6, we use the "peeling" / block decomposition strategy.
        # We split n=3 into 1+2.
        # Sub-problem 1: 1x5x6 (rank 30)
        # Sub-problem 2: 2x5x6 (rank 48)
        # Total rank: 78
        
        # Construct sub-decomposition for 1x5x6
        U1, V1, W1, r1 = self._construct_1x5x6()
        
        # Construct sub-decomposition for 2x5x6
        U2, V2, W2, r2 = self._construct_2x5x6()
        
        total_rank = r1 + r2
        
        # Assemble U (3*5=15 rows, total_rank cols)
        U = [[0] * total_rank for _ in range(15)]
        # Assemble V (5*6=30 rows, total_rank cols)
        V = [[0] * total_rank for _ in range(30)]
        # Assemble W (6*3=18 rows, total_rank cols)
        W = [[0] * total_rank for _ in range(18)]
        
        # Copy sub-1 (1x5x6)
        # U1 is 5 x r1. These correspond to i=0.
        for j in range(5):
            for c in range(r1):
                U[0*5 + j][c] = U1[j][c]
        
        # V1 is 30 x r1. Same j,k indices.
        for idx in range(30):
            for c in range(r1):
                V[idx][c] = V1[idx][c]
        
        # W1 is 18 x r1. W index is k*1 + i. Since n1=1, i is always 0.
        # Wait, W index for sub-problem (n1, m, p) is k*n1 + i.
        # For n1=1, W1 row index is k.
        # In the full problem (n=3), W row index is k*3 + i.
        # For i=0, index is k*3.
        # So W1[k][c] goes to W[k*3 + 0][c].
        for k in range(6):
            for c in range(r1):
                W[k*3 + 0][c] = W1[k][c]
        
        # Copy sub-2 (2x5x6)
        # U2 is 10 x r2. These correspond to i=1,2.
        # U2 row index is i2*5 + j, where i2 in {0,1}.
        # Full U row index is (1+i2)*5 + j.
        for i2 in range(2):
            for j in range(5):
                for c in range(r2):
                    U[(1+i2)*5 + j][r1 + c] = U2[i2*5 + j][c]
        
        # V2 is 30 x r2. Same j,k indices.
        for idx in range(30):
            for c in range(r2):
                V[idx][r1 + c] = V2[idx][c]
        
        # W2 is 18 x r2. W2 row index is k*2 + i2.
        # Full W row index is k*3 + (1+i2).
        for k in range(6):
            for i2 in range(2):
                for c in range(r2):
                    W[k*3 + (1+i2)][r1 + c] = W2[k*2 + i2][c]
        
        return U, V, W, total_rank

    def _construct_1x5x6(self):
        # 1x5x6 matrix multiplication: (1x5) * (5x6) -> (1x6)
        # Rank = 30.
        # U: 5x30. U[j][c] = 1 if c == j*6 + k? No, wait.
        # Tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication term.
        # C = A B. C_{i,l} = sum_j A_{i,j} B_{j,l}.
        # Here i is fixed to 0 (size 1).
        # We need C_{0,l} = sum_{j=0}^4 A_{0,j} B_{j,l}.
        # The terms are indexed by (j, l). There are 5*6=30 terms.
        # Let c index (j, l) with c = j*6 + l.
        # U row j, col c: U[j][c] = 1 if c==j*6+l for any l? No.
        # U corresponds to A. A is 1x5. Row 0.
        # U[j][c] should be A_{0,j} coefficient for product c.
        # Product c corresponds to A_{0,j} * B_{j,l}.
        # So U[j][c] = 1 if c corresponds to j (i.e., c // 6 == j) and the A part is A_{0,j}.
        # Actually, the standard form:
        # U_{(i,j), c} = coeff of A_{i,j} in product c.
        # V_{(j,k), c} = coeff of B_{j,k} in product c.
        # W_{(k,i), c} = coeff of C_{i,k} in product c.
        #
        # For 1x5x6:
        # i=0.
        # Product c = j*6 + l.
        # A_{0,j} is used. So U[j][c] = 1 if c//6 == j.
        # B_{j,l} is used. So V[j*6+l][c] = 1 if c == j*6+l. (Row index of V is j*6+k, where k is col of B, l is col of B).
        # Wait, V is (m*p) x rank. m=5, p=6. Row index is j*6 + l.
        # So V[j*6+l][c] = 1 if c == j*6+l.
        # W is (p*n) x rank. p=6, n=1. Row index is k*n + i = l*1 + 0 = l.
        # C_{0,l} gets the term. So W[l][c] = 1 if c%6 == l.
        
        r = 30
        U = [[0]*r for _ in range(5)]
        V = [[0]*r for _ in range(30)]
        W = [[0]*r for _ in range(6)]
        
        for j in range(5):
            for l in range(6):
                c = j*6 + l
                U[j][c] = 1
                V[j*6+l][c] = 1
                W[l][c] = 1
        
        return U, V, W, r

    def _construct_2x5x6(self):
        # 2x5x6 matrix multiplication.
        # Split m=5 into 2+3.
        # Sub-problem 1: 2x2x6 (rank 18)
        # Sub-problem 2: 2x3x6 (rank 30)
        # Total rank: 48.
        
        U1, V1, W1, r1 = self._construct_2x2x6()
        U2, V2, W2, r2 = self._construct_2x3x6()
        
        r = r1 + r2
        U = [[0]*r for _ in range(10)]
        V = [[0]*r for _ in range(30)]
        W = [[0]*r for _ in range(12)]
        
        # Copy Sub-1 (2x2x6)
        # U1: 2*2=4 rows. i in {0,1}, j in {0,1}.
        # Full U: i in {0,1}, j in {0,4}.
        for i in range(2):
            for j in range(2):
                for c in range(r1):
                    U[i*5 + j][c] = U1[i*2 + j][c]
        
        # V1: 2*6=12 rows. j in {0,1}, k in {0,5}.
        # Full V: j in {0,4}, k in {0,5}.
        for j in range(2):
            for k in range(6):
                for c in range(r1):
                    V[j*6 + k][c] = V1[j*6 + k][c]
        
        # W1: 6*2=12 rows. k in {0,5}, i in {0,1}.
        # Full W: k in {0,5}, i in {0,1}.
        for k in range(6):
            for i in range(2):
                for c in range(r1):
                    W[k*2 + i][c] = W1[k*2 + i][c]
        
        # Copy Sub-2 (2x3x6)
        # U2: 2*3=6 rows. i in {0,1}, j in {0,2} (local).
        # Full U: i in {0,1}, j in {2,4} (global).
        for i in range(2):
            for j in range(3):
                for c in range(r2):
                    U[i*5 + (2+j)][r1 + c] = U2[i*3 + j][c]
        
        # V2: 3*6=18 rows. j in {0,2} (local), k in {0,5}.
        # Full V: j in {2,4} (global), k in {0,5}.
        for j in range(3):
            for k in range(6):
                for c in range(r2):
                    V[(2+j)*6 + k][r1 + c] = V2[j*6 + k][c]
        
        # W2: 6*2=12 rows. k in {0,5}, i in {0,1}.
        # Full W: k in {0,5}, i in {0,1}.
        for k in range(6):
            for i in range(2):
                for c in range(r2):
                    W[k*2 + i][r1 + c] = W2[k*2 + i][c]
        
        return U, V, W, r

    def _construct_2x2x6(self):
        # 2x2x6. Split p=6 into 3+3.
        # Sub-problem 1: 2x2x3 (rank 9)
        # Sub-problem 2: 2x2x3 (rank 9)
        # Total rank: 18.
        
        U1, V1, W1, r1 = self._construct_2x2x3()
        U2, V2, W2, r2 = self._construct_2x2x3()
        
        r = r1 + r2
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(12)]
        W = [[0]*r for _ in range(12)]
        
        # Copy Sub-1 (2x2x3)
        # U1: 4 rows.
        for idx in range(4):
            for c in range(r1):
                U[idx][c] = U1[idx][c]
        
        # V1: 2*3=6 rows. j in {0,1}, k in {0,2}.
        # Full V: j in {0,1}, k in {0,5}.
        for j in range(2):
            for k in range(3):
                for c in range(r1):
                    V[j*6 + k][c] = V1[j*3 + k][c]
        
        # W1: 3*2=6 rows. k in {0,2}, i in {0,1}.
        # Full W: k in {0,5}, i in {0,1}.
        for k in range(3):
            for i in range(2):
                for c in range(r1):
                    W[k*2 + i][c] = W1[k*2 + i][c]
        
        # Copy Sub-2 (2x2x3)
        # U2: 4 rows.
        for idx in range(4):
            for c in range(r2):
                U[idx][r1 + c] = U2[idx][c]
        
        # V2: 2*3=6 rows. j in {0,1}, k in {0,2} (local).
        # Full V: j in {0,1}, k in {3,5} (global).
        for j in range(2):
            for k in range(3):
                for c in range(r2):
                    V[j*6 + (3+k)][r1 + c] = V2[j*3 + k][c]
        
        # W2: 3*2=6 rows. k in {0,2} (local), i in {0,1}.
        # Full W: k in {3,5} (global), i in {0,1}.
        for k in range(3):
            for i in range(2):
                for c in range(r2):
                    W[(3+k)*2 + i][r1 + c] = W2[k*2 + i][c]
        
        return U, V, W, r

    def _construct_2x2x3(self):
        # 2x2x3. Split p=3 into 1+2.
        # Sub-problem 1: 2x2x1 (rank 4)
        # Sub-problem 2: 2x2x2 (rank 7, Strassen)
        # Total rank: 11? Wait.
        # Rank of 2x2x1 is 4?
        # C = A B. C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # k is fixed.
        # Terms: (i,j). 4 terms.
        # Yes, rank 4.
        # Rank of 2x2x2 is 7 (Strassen).
        # Total rank: 11.
        
        U1, V1, W1, r1 = self._construct_2x2x1()
        U2, V2, W2, r2 = self._strassen_2x2x2()
        
        r = r1 + r2
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(6)]
        
        # Copy Sub-1 (2x2x1)
        for idx in range(4):
            for c in range(r1):
                U[idx][c] = U1[idx][c]
        
        # V1: 2*1=2 rows. j in {0,1}, k=0.
        # Full V: j in {0,1}, k in {0,2}.
        for j in range(2):
            for c in range(r1):
                V[j*3 + 0][c] = V1[j][c]
        
        # W1: 1*2=2 rows. k=0, i in {0,1}.
        # Full W: k in {0,2}, i in {0,1}.
        for i in range(2):
            for c in range(r1):
                W[0*2 + i][c] = W1[i][c]
        
        # Copy Sub-2 (2x2x2)
        for idx in range(4):
            for c in range(r2):
                U[idx][r1 + c] = U2[idx][c]
        
        # V2: 2*2=4 rows. j in {0,1}, k in {0,1} (local).
        # Full V: j in {0,1}, k in {1,2} (global).
        for j in range(2):
            for k in range(2):
                for c in range(r2):
                    V[j*3 + (1+k)][r1 + c] = V2[j*2 + k][c]
        
        # W2: 2*2=4 rows. k in {0,1} (local), i in {0,1}.
        # Full W: k in {1,2} (global), i in {0,1}.
        for k in range(2):
            for i in range(2):
                for c in range(r2):
                    W[(1+k)*2 + i][r1 + c] = W2[k*2 + i][c]
        
        return U, V, W, r

    def _construct_2x2x1(self):
        # 2x2x1. Rank 4.
        r = 4
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(2)]
        W = [[0]*r for _ in range(2)]
        
        # C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}
        # Term 0: i=0, j=0
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Term 1: i=0, j=1
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        # Term 2: i=1, j=0
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1
        
        # Term 3: i=1, j=1
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1
        
        return U, V, W, r

    def _construct_2x3x6(self):
        # 2x3x6. Split m=3 into 1+2.
        # Sub-problem 1: 2x1x6 (rank 12)
        # Sub-problem 2: 2x2x6 (rank 18)
        # Total rank: 30.
        
        U1, V1, W1, r1 = self._construct_2x1x6()
        U2, V2, W2, r2 = self._construct_2x2x6()
        
        r = r1 + r2
        U = [[0]*r for _ in range(6)]
        V = [[0]*r for _ in range(18)]
        W = [[0]*r for _ in range(12)]
        
        # Copy Sub-1 (2x1x6)
        # U1: 2*1=2 rows. i in {0,1}, j=0.
        for i in range(2):
            for c in range(r1):
                U[i*3 + 0][c] = U1[i][c]
        
        # V1: 1*6=6 rows. j=0, k in {0,5}.
        for k in range(6):
            for c in range(r1):
                V[0*6 + k][c] = V1[k][c]
        
        # W1: 6*2=12 rows. k in {0,5}, i in {0,1}.
        for idx in range(12):
            for c in range(r1):
                W[idx][c] = W1[idx][c]
        
        # Copy Sub-2 (2x2x6)
        # U2: 2*2=4 rows. i in {0,1}, j in {0,1} (local).
        # Full U: i in {0,1}, j in {1,2} (global).
        for i in range(2):
            for j in range(2):
                for c in range(r2):
                    U[i*3 + (1+j)][r1 + c] = U2[i*2 + j][c]
        
        # V2: 2*6=12 rows. j in {0,1} (local), k in {0,5}.
        # Full V: j in {1,2} (global), k in {0,5}.
        for j in range(2):
            for k in range(6):
                for c in range(r2):
                    V[(1+j)*6 + k][r1 + c] = V2[j*6 + k][c]
        
        # W2: 6*2=12 rows. k in {0,5}, i in {0,1}.
        for idx in range(12):
            for c in range(r2):
                W[idx][r1 + c] = W2[idx][c]
        
        return U, V, W, r

    def _construct_2x1x6(self):
        # 2x1x6. Rank 12.
        # C_{i,k} = A_{i,0} B_{0,k}
        # Terms: (i,k).
        r = 12
        U = [[0]*r for _ in range(2)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(12)]
        
        for i in range(2):
            for k in range(6):
                c = i*6 + k
                U[i][c] = 1
                V[k][c] = 1
                W[k*2 + i][c] = 1
        
        return U, V, W, r

    def _strassen_2x2x2(self):
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[2][2] = 1; V[0][2] = -1
        W[0][2] = -1; W[2][2] = 1
        
        U[3][3] = 1
        V[1][3] = 1; V[3][3] = -1
        W[1][3] = 1; W[3][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1
        
        return U, V, W, 7

def construct():
    n, m, p = DIMENSIONS
    solver = TensorDecomposition()
    U, V, W, rank = solver.construct(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
