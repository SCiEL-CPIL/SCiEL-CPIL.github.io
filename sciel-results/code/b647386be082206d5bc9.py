import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # We will use 4 Strassen blocks of size 2x2x2
    # Covering i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Blocks:
    # 1. i0=0, j0=0, k0=0 -> covers i:0-1, j:0-1, k:0-1
    # 2. i0=0, j0=2, k0=0 -> covers i:0-1, j:2-3, k:0-1
    # 3. i0=0, j0=0, k0=2 -> covers i:0-1, j:0-1, k:2-3
    # 4. i0=0, j0=2, k0=2 -> covers i:0-1, j:2-3, k:2-3
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    # Terms covered by Strassen are exactly 4 * 8 = 32
    # However, we must be careful not to double count or miss.
    # The blocks are disjoint in index space.
    # Covered set C = { (i,j,k) | exists (i0,j0,k0) in blocks s.t. i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1] }
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    schoolbook_terms_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    schoolbook_terms_count += 1
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i0, j0, k0, col_offset):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    r = total_rank
    tensor_size = n * m * p
    
    # Build the tensor representation: for each rank-1 term c, the tensor is outer(U[:,c], V[:,c], W[:,c])
    # We want to find subsets S of {0,...,r-1} such that sum_{c in S} outer(U[:,c], V[:,c], W[:,c]) = 0
    
    # Precompute each rank-1 term as a flattened vector
    terms = []
    for c in range(r):
        t = np.outer(np.outer(U_np[:, c], V_np[:, c]), W_np[:, c]).reshape(-1)
        terms.append(t)
    
    # Try to find small subsets that sum to zero
    # Start with pairs
    removed = set()
    
    # Check pairs
    for c1 in range(r):
        if c1 in removed:
            continue
        if time.time() > deadline:
            break
        for c2 in range(c1 + 1, r):
            if c2 in removed:
                continue
            if time.time() > deadline:
                break
            # Check if terms[c1] + terms[c2] = 0 (i.e., terms[c1] = -terms[c2])
            if np.allclose(terms[c1] + terms[c2], 0, atol=1e-10):
                removed.add(c1)
                removed.add(c2)
                break
    
    # Check triples among remaining
    remaining = [c for c in range(r) if c not in removed]
    for i1 in range(len(remaining)):
        if time.time() > deadline:
            break
        c1 = remaining[i1]
        for i2 in range(i1 + 1, len(remaining)):
            c2 = remaining[i2]
            for i3 in range(i2 + 1, len(remaining)):
                c3 = remaining[i3]
                if time.time() > deadline:
                    break
                if np.allclose(terms[c1] + terms[c2] + terms[c3], 0, atol=1e-10):
                    removed.add(c1)
                    removed.add(c2)
                    removed.add(c3)
                    break
            if time.time() > deadline:
                break
    
    # Rebuild without removed terms
    if removed:
        keep = [c for c in range(r) if c not in removed]
        U_final = U_np[:, keep]
        V_final = V_np[:, keep]
        W_final = W_np[:, keep]
        
        # Verify the decomposition is still correct
        # Reconstruct the tensor
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for idx, c in enumerate(keep):
            tensor += np.einsum('i,j,k->ijk', U_final[:, idx], V_final[:, idx], W_final[:, idx])
        
        # The correct tensor for matrix multiplication has T[i,j,k] = 1 if this is a valid term
        # Actually, the tensor T[i,j,k] should be 1 for all (i,j,k) since it's the identity mapping
        # Wait - let me reconsider. The tensor for matrix multiplication C = AB has
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k)
        
        # Check that our decomposition gives the correct tensor
        expected = np.ones((n, m, p), dtype=np.float64)
        if not np.allclose(tensor, expected, atol=1e-10):
            # Fallback: use the original
            pass
        else:
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
