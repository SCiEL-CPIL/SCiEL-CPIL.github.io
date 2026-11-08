import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    # First, verify the Strassen rank-28 decomposition works
    def verify_strassen():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # Strassen's algorithm for 2x2 block matrices
        # Each 2x2 block multiplication uses 7 multiplications
        # We have 4 such block multiplications: (A11*A21) x (B11*B21), (A11*A21) x (B12*B22), 
        # (A12*A22) x (B11*B21), (A12*A22) x (B12*B22)
        # Wait, for 2x4 @ 4x4:
        # A is 2x4, split into 1x2 blocks of size 1x2: A1 (row 0), A2 (row 1)
        # B is 4x4, split into 2x2 blocks of size 2x2: B11, B12, B21, B22
        # C is 2x4, split into 1x2 blocks of size 1x2: C1 (row 0), C2 (row 1)
        # C1 = A1 @ B11 + A1 @ B21 = A1 @ (B11 + B21)
        # C2 = A1 @ B12 + A1 @ B22 = A1 @ (B12 + B22)
        
        # Actually, let me use a simpler approach: direct decomposition
        # For 2x4 @ 4x4, the trivial decomposition has rank 2*4*4 = 32? No, that's not right.
        # The trivial (schoolbook) multiplication has rank n*m*p = 2*4*4 = 32? 
        # No, the tensor rank of matrix multiplication is the minimum number of scalar multiplications.
        # For n x m @ m x p, the trivial rank is n*m*p.
        # But we can do better with Strassen-like methods.
        
        # Let me construct a known-good decomposition.
        # For 2x4 @ 4x4, we can view it as two independent 2x4 @ 4x4... no, it's just one.
        # Let me try a direct construction.
        
        # Actually, let me just use the schoolbook algorithm with rank 8 (since n=2, m=4, p=4, 
        # but we can optimize by noting the structure).
        
        # Schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
        # This requires 2*4*4 = 32 scalar multiplications if done naively.
        # But we can group them. For each pair (i,k), we sum over j.
        # The rank is the minimum number of rank-1 tensors whose sum gives the multiplication tensor.
        
        # Let me just return the Strassen-style decomposition that I know works.
        # For 2x4 @ 4x4, we can split B into two 4x2 blocks: B_left (cols 0,1) and B_right (cols 2,3)
        # Then C = A @ [B_left | B_right] = [A @ B_left | A @ B_right]
        # Each A @ B_left is 2x4 @ 4x2, which has trivial rank 2*4*2 = 16
        # Total rank = 16 + 16 = 32? That's worse.
        
        # Let me think differently. The known lower bound for 2x4x4 is at least 16.
        # Let me try to construct a rank 16 decomposition directly.
        
        # Actually, let me just return a valid decomposition with some rank.
        # The simplest is the schoolbook with rank n*m*p = 32, but we can do better.
        
        # Let me construct a rank 8 decomposition using the following idea:
        # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        # For rank 8, we can use 4 terms per row of C.
        # But that's still 8 terms total (2 rows * 4 terms).
        
        # Wait, the rank is the number of scalar multiplications. Each term in the sum 
        # A[i,j]*B[j,k] is one multiplication. So the schoolbook rank is 2*4*4 = 32.
        
        # But we can combine terms! For example:
        # A[i,0]*B[0,k] + A[i,1]*B[1,k] = A[i,0]*(B[0,k]) + A[i,1]*(B[1,k])
        # This doesn't help directly.
        
        # Let me just implement the schoolbook decomposition with rank 32 and verify it works.
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Map r to (i, j, k) where r = i * m * p + j * p + k
            # But we only need r up to n*m*p = 32
            i = r // (m * p)
            j = (r // p) % m
            k = r % p
            if i < n:
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
        
        # Verify
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        
        if not np.allclose(C, Cf):
            return None
        return {"U": U, "V": V, "W": W}
    
    # First try the schoolbook rank 32
    result = verify_strassen()
    if result:
        return {
            "U": result["U"].tolist(),
            "V": result["V"].tolist(),
            "W": result["W"].tolist()
        }
    
    # Fallback: return an empty dict (shouldn't happen)
    return {"U": [], "V": [], "W": []}

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = 2, 4, 4
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
    else:
        print("Verification Passed")
    
    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
