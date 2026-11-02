"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Try to find lower rank via integer/rational search
    # The tensor for matrix multiplication is T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # Rank of matrix multiplication tensor for (n,m,p) is at most n*m*p
    # For 4x4x6, the known lower bound is at least max(nm, mp, pn) = max(16, 24, 24) = 24
    # But the actual rank could be lower than the schoolbook rank of 96.
    
    # Let's try a random search for lower rank decompositions with integer/half-integer entries
    deadline = time.time() + 120
    random.seed(42)
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        # For each output entry (i,k), sum_c U[i*m+0:c... no
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k]=1
        # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] (or similar)
        # 
        # The convention says W[k*n+i, r], so W is indexed by (k,i)
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # Tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This must equal 1 for all (i,j,k) since matrix mult is "1" for each term
        # Wait, no. The tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k? 
        # No, that's not right either.
        #
        # Actually, the matrix multiplication tensor: to compute C = A*B where A is n x m and B is m x p,
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The bilinear form: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
        # where T[i,j,k] = 1 for all i,j,k (each triple (i,j,k) contributes to exactly one output)
        # Wait, that's not right. Let me think again.
        #
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This is a linear combination. The "tensor" here is the structure of how indices are contracted.
        # In tensor decomposition terms: the multiplication tensor mu_{n,m,p} has entries mu[i,j,k] = 1 
        # for all valid i,j,k, and we want to write this as sum_c U_c V_c W_c where
        # mu[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try ranks from lower bound upward
    # Lower bound: rank >= max(nm, mp, pn) = 24
    # Try a few ranks below schoolbook
    for target_rank in range(24, min(rank_school, 60)):
        if time.time() > deadline - 30:
            break
        
        # Random search with integer entries in {-2,-1,0,1,2}
        found = False
        attempts = 0
        max_attempts = 5000000
        
        while attempts < max_attempts and not found and time.time() < deadline - 30:
            attempts += 1
            if attempts % 100000 == 0 and time.time() > deadline - 30:
                break
            
            # Generate random sparse matrices with entries in {-1, 0, 1}
            U_r = [[0]*target_rank for _ in range(nm)]
            V_r = [[0]*target_rank for _ in range(mp)]
            W_r = [[0]*target_rank for _ in range(pn)]
            
            # Make them sparse: each column has few non-zero entries
            for c in range(target_rank):
                # Randomly choose which rows are non-zero in U
                u_nonzero = random.sample(range(nm), random.randint(2, 4))
                for idx in u_nonzero:
                    U_r[idx][c] = random.choice([-1, 1])
                
                v_nonzero = random.sample(range(mp), random.randint(2, 4))
                for idx in v_nonzero:
                    V_r[idx][c] = random.choice([-1, 1])
                
                w_nonzero = random.sample(range(pn), random.randint(2, 4))
                for idx in w_nonzero:
                    W_r[idx][c] = random.choice([-1, 1])
            
            # Quick check: for a random subset of (i,j,k) triples
            if random.random() < 0.01:  # Full check only 1% of the time
                if verify(U_r, V_r, W_r, target_rank, n, m, p):
                    found = True
                    best = {'U': U_r, 'V': V_r, 'W': W_r, 'rank': target_rank}
                    break
            else:
                # Partial check
                ok = True
                for _ in range(20):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    s = 0
                    for c in range(target_rank):
                        s += U_r[i*m+j][c] * V_r[j*p+k][c] * W_r[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if ok:
                    # Full verification
                    if verify(U_r, V_r, W_r, target_rank, n, m, p):
                        found = True
                        best = {'U': U_r, 'V': V_r, 'W': W_r, 'rank': target_rank}
                        break
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
