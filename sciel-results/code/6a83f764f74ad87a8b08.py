"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    # Build the tensor from U, V, W: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # For each output element C[k], verify it equals sum over A[i]*B[j]
    # C[k] = sum_i,j T[i,j,k] * A[i] * B[j]
    # We need T[i,j,k] = 1 if C[k] = A[i]*B[j], 0 otherwise
    
    # A[i] where i = a*n + b (a=row, b=col)
    # B[j] where j = b'*p + c (b'=row, c=col)
    # C[k] where k = c'*n + a' (c'=col, a'=row)
    
    # C[a',c'] = sum_b A[a,b] * B[b,c']
    # So T[i,j,k] should be 1 iff:
    #   i = a*n + b, j = b*p + c', k = c'*n + a'
    #   i.e., a' = a, b' = b, c' = c'
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                # C[a,c] = A[a,b]*B[b,c]
                i = a * m + b  # A index
                j = b * p + c  # B index
                k = c * n + a  # C index
                
                # Check T[i,j,k]
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Also check that T[i,j,k] = 0 for non-matching triples
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                # Check if this is a valid multiplication term
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue  # Already checked above
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Start with the known rank-14 solution
    best_result = _build_rank14()
    
    # Verify the base solution
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"]):
        # Try to find a lower-rank solution using random search
        # This is a heuristic search; we keep the rank-14 if we can't do better
        
        # Try rank 12 and 13 with random initializations
        for target_rank in [13, 12]:
            if time.time() > deadline - 10:
                break
            
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization with small values
                U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(4)]
                V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(8)]
                W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(8)]
                
                # Try a few gradient-free refinement steps
                for step in range(20):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute error
                    error = 0.0
                    for a in range(2):
                        for b in range(2):
                            for c in range(4):
                                i = a * 2 + b
                                j = b * 4 + c
                                k = c * 2 + a
                                t_val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(target_rank))
                                error += (t_val - 1.0) ** 2
                    
                    for i in range(4):
                        for j in range(8):
                            for k in range(8):
                                if i // 2 == k % 2 and i % 2 == j // 4:
                                    continue
                                t_val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(target_rank))
                                error += t_val ** 2
                    
                    if error < 1e-6:
                        if _verify_decomposition(U, V, W):
                            return {"U": U, "V": V, "W": W}
                    
                    # Small perturbations
                    if random.random() < 0.3:
                        idx = random.randint(0, 3)
                        r = random.randint(0, target_rank - 1)
                        U[idx][r] += random.choice([-1, 1])
                    if random.random() < 0.3:
                        idx = random.randint(0, 7)
                        r = random.randint(0, target_rank - 1)
                        V[idx][r] += random.choice([-1, 1])
                    if random.random() < 0.3:
                        idx = random.randint(0, 7)
                        r = random.randint(0, target_rank - 1)
                        W[idx][r] += random.choice([-1, 1])
    
    # Return the best verified solution
    return best_result
# EVOLVE-BLOCK-END
