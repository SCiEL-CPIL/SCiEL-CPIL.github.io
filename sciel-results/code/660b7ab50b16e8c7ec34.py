import random
import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        # Check T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try randomized search for lower rank
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from low to schoolbook
    for target_rank in range(1, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Multiple attempts per rank
        max_attempts = 50 if target_rank < 40 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 1000 + target_rank)
            
            # Generate random matrices with small integer/half-integer entries
            # Use values from {-1, -0.5, 0, 0.5, 1} for simplicity
            def rand_val():
                vals = [-1, -0.5, 0, 0.5, 1]
                return random.choice(vals)
            
            U = [[rand_val() for _ in range(target_rank)] for _ in range(nm)]
            V = [[rand_val() for _ in range(target_rank)] for _ in range(mp)]
            W = [[rand_val() for _ in range(target_rank)] for _ in range(pn)]
            
            if verify(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    # Verify with random matrices
    random.seed(42)
    A = [[random.randint(0, 10) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(0, 10) for _ in range(p)] for _ in range(m)]
    
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                s += A[i][j] * B[j][k]
            C[i][k] = s
    
    C_prime = [[0] * p for _ in range(n)]
    for c in range(rank):
        sum_u = sum(U[i * m + j][c] * A[i][j] for i in range(n) for j in range(m))
        sum_v = sum(V[j * p + k][c] * B[j][k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_prime[i][k] += sum_u * sum_v * W[k * n + i][c]
                
    for i in range(n):
        for k in range(p):
            assert abs(C[i][k] - C_prime[i][k]) < 1e-9, f"Mismatch at ({i},{k}): {C[i][k]} vs {C_prime[i][k]}"
                
    print("Decomposition verified. Rank =", rank)
