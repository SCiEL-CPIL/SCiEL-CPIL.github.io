import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 130

    def make_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify_fast(U, V, W, A, B, tol=1e-6):
        """Vectorized verification."""
        C = A @ B
        n, m, p = DIMENSIONS
        C_decomp = np.zeros((n, p))
        for t in range(U.shape[1]):
            V_t = V[:, t].reshape(m, p)
            for i in range(n):
                U_t_i = U[i*m:(i+1)*m, t]
                contrib = np.dot(U_t_i * A[i, :], V_t)
                W_t = W[:, t]
                for k in range(p):
                    C_decomp[i, k] += W_t[k*n + i] * contrib[k]
        return np.allclose(C_decomp, C, atol=tol)

    # Strategy: Karatsuba on m=4
    # Split A (2x4) into A0 (2x2), A1 (2x2)
    # Split B (4x8) into B0 (2x8), B1 (2x8)
    # C = A0 B0 + A1 B1
    # Use Karatsuba: M1 = A0 B0, M2 = A1 B1, M3 = (A0+A1)(B0+B1)
    # C = M1 + M3 - M2
    
    # Each term is a (2x2) * (2x8) multiplication.
    # This can be viewed as 4 independent (2x2)*(2x2) multiplications.
    # Rank of (2x2)*(2x2) is 7 (Strassen).
    # So rank of (2x2)*(2x8) is 4 * 7 = 28.
    # Total rank = 3 * 28 = 84? No, wait.
    
    # Let's refine the decomposition of (2x2)*(2x8).
    # Let A_s be 2x2, B_s be 2x8.
    # B_s = [B_s1, B_s2, B_s3, B_s4] where each B_sj is 2x2.
    # A_s B_s = [A_s B_s1, A_s B_s2, A_s B_s3, A_s B_s4].
    # Each A_s B_sj is a 2x2*2x2 product.
    # If we use Strassen for each, we need 7 * 4 = 28 bilinear forms.
    # However, can we share the A_s factors?
    # Strassen forms:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # For 4 different B blocks, the A factors are the same.
    # So we compute the A-side Strassen intermediates once?
    # No, the bilinear form is sum_t U_t A V_t.
    # For a specific (A_s, B_s) pair, the rank is 28.
    # But wait, the Kronecker product structure:
    # Tensor T_{2,2,2} has rank 7.
    # We are computing T_{2,2,2} \otimes I_4 ? No.
    # We are computing 4 independent T_{2,2,2}.
    # The tensor for (2x2)*(2x8) is T_{2,2,2} \otimes I_4 (where I_4 is identity on the 4 output columns of B).
    # Rank(T \otimes I) <= Rank(T) * Rank(I) = 7 * 4 = 28.
    # Is it exactly 28? Yes, likely.
    # So each of the 3 Karatsuba terms has rank 28.
    # Total rank = 3 * 28 = 84.
    
    # This is WORSE than 56.
    # Why? Because the 56 construction used 8 independent (2x2)*(2x2) blocks.
    # (2x4)*(4x8) can be split into 2*(2x2)*(2x8) + ... no.
    # (2x4)*(4x8) = [A0 A1] * [B0; B1] = A0 B0 + A1 B1.
    # A0 is 2x2, B0 is 2x8. Rank 28.
    # A1 is 2x2, B1 is 2x8. Rank 28.
    # Total rank 56.
    # Karatsuba adds a third term (A0+A1)(B0+B1) of rank 28.
    # But we can recover A0 B0 + A1 B1 from M1, M2, M3.
    # M1 = A0 B0
    # M2 = A1 B1
    # M3 = (A0+A1)(B0+B1) = A0 B0 + A0 B1 + A1 B0 + A1 B1
    # We need A0 B0 + A1 B1.
    # We have M1, M2, M3.
    # M1 + M2 = A0 B0 + A1 B1. This is exactly what we want!
    # So we don't need M3?
    # Wait, Karatsuba is for multiplication of polynomials or larger matrices where you want to reduce the number of multiplications.
    # Here, A0 B0 and A1 B1 are already computed.
    # The "product" is the sum.
    # The standard decomposition is just sum of two products.
    # Rank(A0 B0) + Rank(A1 B1) = 28 + 28 = 56.
    # Karatsuba doesn't help here because we are summing two independent products, not multiplying two polynomials where cross terms are needed.
    # Unless... we can compute A0 B0 + A1 B1 with fewer than 56 ranks?
    # The tensor is T = T_{A0,B0} + T_{A1,B1}.
    # These tensors have disjoint supports in the index space (different i,j,k ranges? No, same i,k, different j).
    # Actually, the supports are disjoint in the 'j' index of U and V.
    # U for A0 B0 uses j=0,1. U for A1 B1 uses j=2,3.
    # So the tensors are orthogonal in the j-index.
    # Rank(T1 + T2) = Rank(T1) + Rank(T2) if supports are disjoint in a way that prevents cancellation/merging.
    # So 56 is likely the minimum for this "split and sum" approach.
    
    # Can we do better than 56?
    # The theoretical lower bound for (2,4,8) is not known to be < 56, but 56 is the "obvious" upper bound.
    # Is there a rank-48 or rank-40 construction?
    # Laderman's algorithm for (4,4,4) has rank 20.
    # (2,4,8) is not a cube.
    
    # Let's check the rank of the (2,2,8) tensor.
    # T_{2,2,8}.
    # We established Rank(T_{2,2,8}) = 28.
    # Is it possible that Rank(T_{2,2,8}) < 28?
    # T_{2,2,2} has rank 7.
    # T_{2,2,8} = T_{2,2,2} \otimes I_4.
    # Rank(T \otimes I) = Rank(T) * Rank(I) is a standard upper bound.
    # Is it a lower bound?
    # For T_{2,2,2}, rank is 7.
    # For T_{2,2,4}, rank is 14?
    # Let's assume 56 is the best structural construction available without complex search.
    
    # However, the current program returns 56.
    # The prompt asks for a DIFFERENT strategy.
    # If I cannot find a lower rank, I should try to find a rank-56 construction that is more robust or verify it faster, OR try a random search for rank < 56.
    
    # Given the time limit, a random search for rank 55 or 54 is unlikely to succeed quickly.
    # But maybe there is a rank-49 construction?
    # 7 * 7 = 49.
    # Can (2,4,8) be done with 49?
    # If we view it as (2,4,2) * (2,2,4)? No.
    
    # Let's try a different structural decomposition:
    # Use the fact that (2,4,8) can be embedded in (4,4,8) or similar? No.
    
    # Let's stick to the 56 construction but ensure it's correct and fast.
    # The current code has a bug in the Strassen block construction or verification?
    # The current code scores 1.0 (which means it matches the seed, likely schoolbook rank 64? No, seed is 1.0, current is 1.0. 
    # Wait, "score is the rank". "1.0 equals the seed program".
    # If seed is schoolbook (rank 64), score 1.0 means rank 64?
    # "score is the rank" -> lower is better.
    # If seed is rank 64, and current is rank 56, score should be 56/64 = 0.875?
    # "search score is seed/value, so 1.0 equals the seed program".
    # This implies Value = Rank. Score = SeedRank / Rank.
    # If SeedRank = 64, and CurrentRank = 56, Score = 64/56 = 1.14.
    # But the current program scores 1.0000.
    # This implies the current program is returning rank 64?
    # Let's check the code.
    # It builds rank 56.
    # It verifies. If verification fails, it returns schoolbook (rank 64).
    # It is highly likely that the verification in the current code FAILS due to a bug in the Strassen block or the assembly.
    # Therefore, it falls back to schoolbook rank 64.
    # So the "improvement" is to fix the verification or the construction so it actually returns 56.
    
    # Let's debug the Strassen block in the current code.
    # U, V, W for 2x2*2x2.
    # Indices: U is (4, 7), V is (4, 7), W is (4, 7).
    # U[i*2+j, t] corresponds to A[i,j].
    # V[j*2+k, t] corresponds to B[j,k].
    # W[k*2+i, t] corresponds to C[i,k].
    
    # Strassen:
    # m1 = (a+d)(e+h) -> U: a,d; V: e,h; W: contributes to C?
    # C = [[ae+bi, af+bj], [ce+di, cf+dj]]
    # m1 = (a+d)(e+h) = ae+ah+de+dh
    # m2 = (c+d)e = ce+de
    # m3 = a(f-j) = af-aj
    # m4 = d(g-i) = dg-di
    # m5 = (a+c)(i+l) = ai+al+ci+cl -- Wait, standard Strassen is:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m7
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's map to U, V, W.
    # t=0: m1.
    # U: a11=1, a22=1. (Indices 0, 3 in U)
    # V: b11=1, b22=1. (Indices 0, 3 in V)
    # W: C11 += 1, C22 += 1. (Indices 0, 3 in W)
    # Current code:
    # U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    # W[0,0]=1; W[3,0]=1
    # This matches m1.
    
    # t=1: m2.
    # U: a21=1, a22=1. (Indices 2, 3)
    # V: b11=1. (Index 0)
    # W: C21 += 1, C22 -= 1. (Indices 2, 3)
    # Current code:
    # U[2,1]=1; U[3,1]=1; V[0,1]=1
    # W[2,1]=1; W[3,1]=-1
    # Matches m2.
    
    # t=2: m3.
    # U: a11=1. (Index 0)
    # V: b21=1, b22=-1. (Indices 2, 3)
    # W: C12 += 1, C22 += 1. (Indices 1, 3)
    # Current code:
    # U[0,2]=1; V[1,2]=1; V[3,2]=-1  <-- ERROR: V[1,2] is b12? No, V index is j*p+k.
    # For 2x2, V[j*2+k].
    # b21 is j=1, k=0 -> Index 2.
    # b22 is j=1, k=1 -> Index 3.
    # Current code has V[1,2]=1 (b12) and V[3,2]=-1 (b22).
    # m3 needs b21 and b22. So V[2,2]=1, V[3,2]=-1.
    # The current code has V[1,2]=1. This is WRONG.
    
    # t=3: m4.
    # U: a22=1. (Index 3)
    # V: b12=1, b11=-1. (Indices 1, 0)
    # W: C11 -= 1, C21 += 1. (Indices 0, 2)
    # Current code:
    # U[3,3]=1; V[2,3]=1; V[0,3]=-1
    # V[2,3] is b21? No, b12 is index 1. b21 is index 2.
    # m4 needs b12 and b11. So V[1,3]=1, V[0,3]=-1.
    # Current code has V[2,3]=1. This is WRONG.
    
    # So the current Strassen block is incorrect, causing verification to fail, leading to fallback to rank 64.
    # My new strategy: Fix the Strassen block and the assembly to correctly produce rank 56.
    
    def make_strassen_block():
        """2x2*2x2 Strassen, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a11+a22)(b11+b22)
        # U: 0,3; V: 0,3; W: 0(+), 3(+)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # m2 = (a21+a22)b11
        # U: 2,3; V: 0; W: 2(+), 3(-)
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        
        # m3 = a11(b21-b22)
        # U: 0; V: 2,3; W: 1(+), 3(+)
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # m4 = a22(b12-b11)
        # U: 3; V: 1,0; W: 0(-), 2(+)
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        W[0,3]=-1; W[2,3]=1
        
        # m5 = (a11+a12)b22
        # U: 0,1; V: 3; W: 0(+), 1(+)
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=1; W[1,4]=1
        
        # m6 = (a21-a11)(b11+b12)
        # U: 2,0; V: 0,1; W: 3(+)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # m7 = (a12-a22)(b21+b22)
        # U: 1,3; V: 2,3; W: 1(+), 3(-)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[1,6]=1; W[3,6]=-1
        
        return U, V, W

    # Build 8 Strassen blocks: rank 56
    U_block, V_block, W_block = make_strassen_block()
    rank_total = 8 * 7  # 56
    U_full = np.zeros((U_dim, rank_total))
    V_full = np.zeros((V_dim, rank_total))
    W_full = np.zeros((W_dim, rank_total))

    col_offset = 0
    # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3).
    # B is 4x8. Split into B0 (rows 0,1) and B1 (rows 2,3).
    # C is 2x8.
    # C = A0 B0 + A1 B1.
    # A0 is 2x2. B0 is 2x8.
    # B0 has 4 columns of 2x2.
    # So for A0 B0, we need 4 Strassen blocks.
    # For A1 B1, we need 4 Strassen blocks.
    # Total 8 blocks.
    
    for a_block in range(2):
        # a_block 0: A0 (cols 0,1). a_block 1: A1 (cols 2,3).
        a_col_base = a_block * 2
        b_row_base = a_block * 2
        for c_block in range(4):
            # c_block 0..3: columns 0,1; 2,3; 4,5; 6,7 of B.
            c_col_base = c_block * 2
            for t in range(7):
                # U: A indices. i in 0..1, j in 0..1 (local).
                # Global j = a_col_base + j_local.
                for i in range(2):
                    for j in range(2):
                        U_full[i*4 + a_col_base + j, col_offset] += U_block[i*2+j, t]
                # V: B indices. j in 0..1 (local), k in 0..1 (local).
                # Global row = b_row_base + j_local. Global col = c_col_base + k_local.
                for j in range(2):
                    for k in range(2):
                        V_full[(b_row_base+j)*8 + c_col_base + k, col_offset] += V_block[j*2+k, t]
                # W: C indices. i in 0..1, k in 0..1 (local).
                # Global row = i. Global col = c_col_base + k_local.
                # W index is k*n + i.
                for i in range(2):
                    for k in range(2):
                        W_full[(c_col_base+k)*2 + i, col_offset] += W_block[k*2+i, t]
                col_offset += 1

    # Verify
    rng = np.random.RandomState(42)
    A_test = rng.randint(-3, 4, size=(n, m)).astype(float)
    B_test = rng.randint(-3, 4, size=(m, p)).astype(float)
    
    if not verify_fast(U_full, V_full, W_full, A_test, B_test):
        U0, V0, W0, r0 = make_schoolbook()
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
