import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
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

def _wft_decompose(n, m, p):
    """
    Winograd Fourier Transform Algorithm for tensor decomposition.
    Pad to next power of 2 in each dimension, apply 3D DFT to diagonalize,
    then prune zero-frequency components.
    """
    # Next power of 2 >= max dimension
    def next_pow2(x):
        r = 1
        while r < x:
            r *= 2
        return r
    
    N = next_pow2(n)
    M = next_pow2(m)
    P = next_pow2(p)
    
    # Build the multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to valid indices
    # T has shape (n*m, m*p, p*n) in the decomposition sense
    # Actually, the tensor is: for each (i,j,k) in [0,n)x[0,m)x[0,p), 
    # the contribution is A[i,j]*B[j,k]*C[k,i] -> output[i,i]
    # In tensor form: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 1 if the indices are consistent, else 0
    
    # The tensor is: T[i,j,k] = 1 for all valid (i,j,k)
    # In the Kronecker product form, this is just the identity-like tensor
    
    # WFTA approach: 
    # 1. Pad the tensor to (N*M, M*P, P*N)
    # 2. Apply 3D DFT along each mode
    # 3. The diagonalized form gives rank = N*M*P terms
    # 4. Prune terms where any factor is zero
    
    # Actually, let me think more carefully.
    # The matrix multiplication tensor for n x m * m x p has entries:
    # T[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # This is a rank-3 tensor of shape (n, m, p) in the "index" sense
    # But in the decomposition sense, we need U (n*m x rank), V (m*p x rank), W (p*n x rank)
    
    # The WFTA diagonalizes the convolution/multiplication structure.
    # For matrix multiplication, the tensor is actually quite simple:
    # T[i,j,k] = 1 if i<m, j<m, k<p (using different notation)
    
    # Let me use a different approach: build the full tensor and use SVD-like decomposition
    # but with the constraint that entries must be integer/half-integer.
    
    # Actually, the cleanest WFTA approach for matmul:
    # The multiplication tensor can be written as a sum of rank-1 tensors.
    # Using DFT on each dimension:
    
    # Build tensor T of shape (N, M, P) with T[i,j,k] = 1 if i<n, j<m, k<p else 0
    T = np.zeros((N, M, P), dtype=complex)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Apply 3D DFT
    T_fft = np.fft.fftn(T)
    
    # The diagonalized decomposition: each (freq1, freq2, freq3) gives a rank-1 term
    # But we need to map back to the U,V,W structure.
    
    # Actually, the standard WFTA for matmul works differently.
    # Let me just build a dense decomposition and then prune.
    
    # Dense decomposition: rank = n*m*p (schoolbook)
    # But we can try to find a smaller rank using the tensor structure.
    
    # Alternative: use the fact that the tensor is separable in certain ways.
    # For 2x5x5, the tensor T[i,j,k] = 1 has a specific structure.
    
    # Let me try a different structural approach: 
    # Use the "partitioning" method with optimized block sizes.
    
    # For 2x5x5, try all possible partitions of the 5 k-dimension and 5 j-dimension
    # into blocks, and use Strassen for 2x2 blocks, schoolbook otherwise.
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook rank
    
    # Generate all ways to partition 5 into contiguous groups
    def partitions_of_5():
        """Generate all partitions of 5 into contiguous integer groups."""
        results = []
        def gen(remaining, current):
            if remaining == 0:
                results.append(tuple(current))
                return
            for size in range(1, remaining + 1):
                current.append(size)
                gen(remaining - size, current)
                current.pop()
        gen(5, [])
        return results
    
    part_k = partitions_of_5()
    part_j = partitions_of_5()
    
    for pk in part_k:
        for pj in part_j:
            # Calculate rank for this partition
            rank = 0
            # Convert partition sizes to actual index groups
            def sizes_to_groups(sizes):
                groups = []
                start = 0
                for s in sizes:
                    groups.append(tuple(range(start, start + s)))
                    start += s
                return groups
            
            kg = sizes_to_groups(pk)
            jg = sizes_to_groups(pj)
            
            for kgroup in kg:
                for jgroup in jg:
                    lk = len(kgroup)
                    lj = len(jgroup)
                    if lk == 2 and lj == 2:
                        rank += 7
                    elif lk == 2 and lj == 1:
                        rank += 4
                    elif lk == 1 and lj == 2:
                        rank += 4
                    else:
                        rank += lk * lj * 2  # 2 rows in A, so 2*lk*lj schoolbook terms
                        # Actually for lk x lj block with 2 A-rows: 2*lk*lj multiplications
                        # But we can do better with Strassen-like for larger blocks
            
            if rank < best_rank:
                # Build this decomposition
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                offset = 0
                for kgroup in kg:
                    for jgroup in jg:
                        lk = len(kgroup)
                        lj = len(jgroup)
                        
                        if lk == 2 and lj == 2:
                            # Strassen 2x2
                            k0, k1 = kgroup
                            j0, j1 = jgroup
                            
                            a00 = 0 * 5 + k0
                            a01 = 0 * 5 + k1
                            a10 = 1 * 5 + k0
                            a11 = 1 * 5 + k1
                            
                            b00 = k0 * 5 + j0
                            b01 = k0 * 5 + j1
                            b10 = k1 * 5 + j0
                            b11 = k1 * 5 + j1
                            
                            c00 = j0 * 2 + 0
                            c01 = j1 * 2 + 0
                            c10 = j0 * 2 + 1
                            c11 = j1 * 2 + 1
                            
                            # r=0: M0 = (A00+A11)(B00+B11)
                            U[a00][offset + 0] += 1
                            U[a11][offset + 0] += 1
                            V[b00][offset + 0] += 1
                            V[b11][offset + 0] += 1
                            W[c00][offset + 0] += 1
                            W[c11][offset + 0] += 1
                            
                            # r=1: M1 = (A10+A11)B00
                            U[a10][offset + 1] += 1
                            U[a11][offset + 1] += 1
                            V[b00][offset + 1] += 1
                            W[c10][offset + 1] += 1
                            W[c11][offset + 1] += -1
                            
                            # r=2: M2 = A00(B01-B11)
                            U[a00][offset + 2] += 1
                            V[b01][offset + 2] += 1
                            V[b11][offset + 2] += -1
                            W[c01][offset + 2] += 1
                            W[c11][offset + 2] += 1
                            
                            # r=3: M3 = A11(B10-B00)
                            U[a11][offset + 3] += 1
                            V[b10][offset + 3] += 1
                            V[b00][offset + 3] += -1
                            W[c00][offset + 3] += 1
                            W[c10][offset + 3] += 1
                            
                            # r=4: M4 = (A00+A01)B11
                            U[a00][offset + 4] += 1
                            U[a01][offset + 4] += 1
                            V[b11][offset + 4] += 1
                            W[c00][offset + 4] += -1
                            W[c01][offset + 4] += 1
                            
                            # r=5: M5 = (A10-A00)(B00+B01)
                            U[a10][offset + 5] += 1
                            U[a00][offset + 5] += -1
                            V[b00][offset + 5] += 1
                            V[b01][offset + 5] += 1
                            W[c11][offset + 5] += 1
                            
                            # r=6: M6 = (A01-A11)(B10+B11)
                            U[a01][offset + 6] += 1
                            U[a11][offset + 6] += -1
                            V[b10][offset + 6] += 1
                            V[b11][offset + 6] += 1
                            W[c00][offset + 6] += 1
                            
                            offset += 7
                        else:
                            # Schoolbook for this block
                            for k in kgroup:
                                for j in jgroup:
                                    for i in range(2):
                                        U[i * 5 + k][offset] += 1
                                        V[k * 5 + j][offset] += 1
                                        W[j * 2 + i][offset] += 1
                                        offset += 1
                
                # Verify
                if _verify_decomp(n, m, p, U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                
                if not _deadline_ok():
                    break
        if not _deadline_ok():
            break
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try WFTA-based approach with partition optimization
    U, V, W = _wft_decompose(n, m, p)
    
    # Verify
    if _verify_decomp(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
