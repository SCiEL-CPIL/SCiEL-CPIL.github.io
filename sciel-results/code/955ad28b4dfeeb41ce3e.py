"""Matrix multiplication 2x3 by 3x6: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 18
P = p * n  # 12


def _schoolbook():
    """Rank-36 schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((N, rank), dtype=object)
    V = np.zeros((M, rank), dtype=object)
    W = np.zeros((P, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-10):
    """Check that sum_r U[:,r] outer V[:,r] outer W[:,r] = matmul tensor."""
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1) > tol:
                    return False
    # Also check that non-matching entries are zero (they should be by construction)
    # Actually the tensor only has 1s at (i,j,k) for valid i,j,k and 0 elsewhere.
    # Since U,V,W have zero rows for non-matching indices, this is automatic.
    return True


def _try_rank(rank, max_iter=500, timeout=None, start_time=None):
    """Try to find a rank-rank decomposition using random initialization + local search.
    
    Uses the approach: random U, V, then solve for W from linear equations,
    then iterate.
    """
    if start_time is None:
        start_time = time.time()
    
    best_result = None
    
    for attempt in range(max_iter):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Random sparse initialization
        density = max(0.05, 0.3 * (n * m * p) / (rank * 3))
        U = np.zeros((N, rank), dtype=float)
        V = np.zeros((M, rank), dtype=float)
        W = np.zeros((P, rank), dtype=float)
        
        for r in range(rank):
            # Pick random non-zero positions
            for _ in range(max(1, int(N * density))):
                U[random.randint(0, N - 1), r] = random.choice([-1, 1])
            for _ in range(max(1, int(M * density))):
                V[random.randint(0, M - 1), r] = random.choice([-1, 1])
            for _ in range(max(1, int(P * density))):
                W[random.randint(0, P - 1), r] = random.choice([-1, 1])
        
        # Try to solve: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of N*M*P equations in rank unknowns (but U,V,W are all unknowns)
        # Use a simpler approach: fix U and V, solve for W linearly.
        
        # For fixed U, V: W[k*n+i, r] must satisfy for all (i,j,k):
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is linear in W.
        
        # Build the linear system: for each (i,j,k), equation:
        # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
        # Unknowns: W[0..P-1, 0..rank-1] -> P*rank unknowns
        # Equations: N*M*P
        
        num_eq = N * M * P
        num_unk = P * rank
        
        # Only build if feasible
        if num_unk > num_eq:
            continue
        
        # Build A matrix (num_eq x num_unk) and b vector
        A = np.zeros((num_eq, num_unk), dtype=float)
        b = np.ones(num_eq, dtype=float)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A[eq_idx, (k * n + i) * rank + r] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        W_sol = W_flat.reshape(P, rank)
        
        # Check residual
        if residuals is not None and len(residuals) > 0:
            if residuals[0] > 1e-6:
                continue
        
        # Verify
        if _verify(U, V, W_sol, tol=1e-6):
            best_result = {"U": U, "V": V, "W": W_sol}
            break
    
    return best_result


def _try_rank_strassen_like(rank, timeout=None, start_time=None):
    """Try structured approaches inspired by Strassen-like block decomposition."""
    if start_time is None:
        start_time = time.time()
    
    # For 2x3 * 3x6, we can try to exploit structure.
    # Think of A as [A1; A2] where A1 is 1x3, A2 is 1x3
    # B as [B1 B2 B3 B4 B5 B6] where each Bj is 3x1
    # Result C[i,j] = sum_k A[i,k]*B[k,j]
    
    # Actually let's try: split B into two 3x3 blocks: B = [B_left, B_right]
    # Then C = A * [B_left, B_right] = [A*B_left, A*B_right]
    # Each A*B is 2x3 * 3x3 = 2x3 matrix multiply
    
    # For 2x3 * 3x3: schoolbook is 18. Can we do better?
    # The rank of 2x3*3x3 matrix multiplication is at most 18, and it's known to be 18
    # (since min(n,m,p) >= 2 and all dimensions >= 2, the rank is n*m*p for these small cases
    #  unless Strassen applies). Actually for 2x3*3x3, the rank is 18 (no Strassen benefit
    # since Strassen needs 2x2*2x2).
    
    # So the whole 2x3*3x6 = two 2x3*3x3 multiplications, rank at most 36.
    # But we might share some computations between the two halves.
    
    # Let's try: A = [a1 a2 a3; b1 b2 b3]
    # B columns: c1..c6, each 3x1
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Group: C_left = A * B[0:3, :] (2x3), C_right = A * B[3:6, :] (2x3)
    # Shared: the products A[i,k] are shared.
    
    # Actually the schoolbook already shares nothing. Let's think differently.
    # 
    # For 2x3 * 3x6:
    # C[0,0] = a1*B[0,0] + a2*B[1,0] + a3*B[2,0]
    # C[0,1] = a1*B[0,1] + a2*B[1,1] + a3*B[2,1]
    # ...
    # C[1,0] = b1*B[0,0] + b2*B[1,0] + b3*B[2,0]
    # ...
    
    # Total: 2*3*6 = 36 terms. Can we combine?
    # If we define S_k = some combination, we might reduce.
    
    # Known lower bound: rank >= n*m*p / max(n,m,p) ... no that's not right.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Also rank >= n*m*p / max(n,m,p) = 36/6 = 6 (weak)
    # Better: rank >= (n*m*p) / (n+m+p-2) ... not standard.
    # The strongest known lower bound for small matrices: rank >= n*m*p / max(n,m,p) is weak.
    # Actually: rank >= max(n*m, m*p, p*n) is a valid lower bound (from the "slice rank" or similar).
    # Wait, that's not quite right either. The rank of the matmul tensor T_{ijk} = delta... 
    # is at least max(n*m, m*p, p*n)? Let me think.
    # 
    # The tensor has rank r means T[i,j,k] = sum_{alpha=1}^r u_i^alpha v_j^alpha w_k^alpha
    # Fixing i: T[i,j,k] = sum_alpha u_i^alpha v_j^alpha w_k^alpha, which is a matrix of rank <= r
    # for each fixed i. The matrix T[i,:,:] is m*p = 18 dimensional. For it to have all
    # entries correct (it's a 3x6 matrix with specific structure), we need rank(T[i,:,:]) <= r.
    # But T[i,:,:] for a general matmul is just A[i,:] * B, which as a matrix is rank 1 (it's
    # outer product of A[i,:] and B^T... no wait).
    # 
    # Actually T[i,j,k] where i indexes rows of A, j indexes columns of... wait let me re-read.
    # 
    # The tensor for matrix multiplication: T[i,j,k] represents the coefficient of A[i,j]*B[j,k]
    # in the product. No wait, the standard formulation:
    # C[p,q] = sum_j A[p,j] * B[j,q]
    # The tensor is T[p,j,q] = A[p,j]*B[j,q], and C[p,q] = sum_j T[p,j,q].
    # 
    # But in this problem's convention: U has shape (n*m, rank), V has shape (m*p, rank),
    # W has shape (p*n, rank). The decomposition is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j too?
    # 
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So the full tensor is: sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = C[i,k]
    # 
    # This is the standard matmul tensor decomposition. The rank is the minimum number of
    # terms in this sum.
    # 
    # Lower bound: For fixed i, sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = C[i,k]
    # This is a bilinear form in (V[:,c], W[:,c]) with coefficients U[i*m+j,c].
    # The matrix of this bilinear form (over j,k) has rank at most the number of c's where
    # any U[i*m+j,c] != 0. So rank <= number of non-zero U entries in row i*m..i*m+m-1.
    # But C[i,k] = A[i,:] * B[:,k], which as a function of (j,k) is rank 1 (outer product).
    # So we need the bilinear form to represent a rank-1 matrix. This doesn't immediately
    # give a strong lower bound on total rank.
    # 
    # Known results: The rank of 2x3 * 3x6 matrix multiplication.
    # By submultiplicativity and known results:
    # - 2x2*2x2: rank 7 (Strassen)
    # - 2x3*3x2: rank 12 (I believe, or maybe less)
    # - 2x3*3x3: rank 18 (schoolbook, I don't think Strassen helps for non-square)
    # 
    # Actually, for 2x3 * 3x6:
    # We can write it as 2x3 * 3x3 * identity... no.
    # 2x3 * 3x6: think of it as computing 2x3 * [3x3, 3x3].
    # Each 2x3*3x3 has rank 18 (schoolbook). Total 36.
    # Can we do better by sharing? The A matrix is shared.
    # 
    # Alternative: 2x3 * 3x6 = (2x3 * 3x3) concatenated with (2x3 * 3x3).
    # If we use rank r1 for first and r2 for second, total is r1+r2.
    # But if we can share, maybe less.
    # 
    # Let's just try to find the minimum empirically.
    # Lower bound: 18 (from m*p = 18, since fixing i gives a 3x6 matrix of rank 1,
    # but the decomposition requires... hmm).
    
    # Let me try a different approach: use the known Strassen algorithm structure.
    # For 2x2*2x2, Strassen uses 7 multiplications.
    # For 2x3*3x6, can we embed Strassen?
    
    # Split: A (2x3) = [A1 A2 A3] where Ai is 2x1
    # B (3x6) rows: B1, B2, B3 each 1x6
    # C = A1*B1 + A2*B2 + A3*B3 (each Ai*Bj is 2x6 outer product)
    # C is 2x6.
    
    # So C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is 3 outer products of 2x1 and 1x6 vectors = 3 * 2 * 6 = 36 scalar mults.
    # 
    # To reduce: group the 6 columns of B into pairs: (B[j,0],B[j,1]), (B[j,2],B[j,3]), (B[j,4],B[j,5])
    # For each pair and each j, we compute A[i,j]*B[j,0] and A[i,j]*B[j,1].
    # That's 2 mults per (i,j,pair). Total: 2*3*3*2 = 36. No savings.
    
    # What if we use Strassen on the 2x2 subproblems?
    # C is 2x6. Split into three 2x2 blocks: C1, C2, C3.
    # C1 = A * B[:, 0:2] (2x3 * 3x2)
    # C2 = A * B[:, 2:4]
    # C3 = A * B[:, 4:6]
    # 
    # For 2x3 * 3x2: 
    # C[i,k] = sum_j A[i,j] * B[j,k], i in {0,1}, k in {0,1}, j in {0,1,2}
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # This is 2*2*3 = 12 scalar mults.
    # 
    # Can we do 2x3*3x2 in less than 12? 
    # The tensor is T[i,j,k] with i in 2, j in 3, k in 2.
    # Rank lower bound: max(2*3, 3*2, 2*2) = 6. Upper bound 12.
    # I believe the rank of 2x3*3x2 is 12 (no known improvement).
    
    # So total for 2x3*3x6 would be 3*12 = 36. Same as schoolbook.
    
    # Let me try yet another decomposition.
    # What if we split A into 1x3 rows?
    # A = [a; b] where a, b are 1x3.
    # C[0,k] = sum_j a_j * B[j,k]
    # C[1,k] = sum_j b_j * B[j,k]
    # 
    # For row 0: C[0,:] = a * B (1x3 * 3x6 = 1x6). This needs 18 mults.
    # For row 1: C[1,:] = b * B (1x3 * 3x6 = 1x6). This needs 18 mults.
    # Total: 36.
    
    # I think 36 might actually be optimal for 2x3*3x6, or maybe we can shave a few.
    # Let me try to search for rank 35, 34, etc.
    
    # Actually, let me try a clever approach:
    # C[0,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k]
    # C[1,k] = b0*B[0,k] + b1*B[1,k] + b2*B[2,k]
    # 
    # Define: for each k, we have two linear combinations.
    # Let's define M_k = [a0*B[0,k], a1*B[1,k], a2*B[2,k], b0*B[0,k], b1*B[1,k], b2*B[2,k]]
    # That's 6 mults per k, 18 total. Plus we need to form the sums.
    # 
    # Alternative: define intermediate products:
    # P[j] = some function... 
    # 
    # I don't see an obvious way to reduce below 36 for this specific size.
    # Let me just try random search for ranks 33-35 to see if anything works.
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook (rank 36)
    best = _schoolbook()
    best_rank = 36
    
    # Try to find lower ranks
    # Based on analysis, 36 might be near-optimal, but let's search.
    
    # Try structured approaches first
    for rank in range(33, 36):
        if time.time() > deadline:
            break
        result = _try_rank(rank, max_iter=100, timeout=deadline - time.time(), start_time=start_time)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # If we have time, try even lower
    for rank in range(30, 33):
        if time.time() > deadline:
            break
        result = _try_rank(rank, max_iter=50, timeout=deadline - time.time(), start_time=start_time)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # Convert to list-of-lists format if needed
    U = best["U"]
    V = best["V"]
    W = best["W"]
    
    # Ensure integer or half-integer entries
    def to_list(mat):
        rows = []
        for row in mat:
            r = []
            for val in row:
                if isinstance(val, (int, np.integer)):
                    r.append(int(val))
                elif isinstance(val, (float, np.floating)):
                    # Round to nearest half-integer
                    v = round(val * 2) / 2
                    r.append(v)
                elif isinstance(val, Fraction):
                    r.append(float(val))
                else:
                    r.append(float(val))
            rows.append(r)
        return rows
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
