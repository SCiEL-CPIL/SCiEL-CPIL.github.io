import numpy as np
import random
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_dim = n * m * p
    
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0 # 10s margin
    
    # Target tensor vectorization
    # T[i, j, k] = 1 if (i, j, k) is a valid index, else 0? 
    # Wait, the tensor for matrix multiplication is T_{i,j,k} = delta_{j, j'} ... no.
    # The tensor is defined such that C = A B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has indices i, j, k. T_{i,j,k} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    # Actually, the standard tensor decomposition form is:
    # C_{ik} = sum_r U_{i,r} V_{j,r} W_{k,r} ... wait, the indices are mixed.
    # Standard form: C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as sum_r (sum_i' A_{i' j} ... )
    # Let's stick to the vectorization of the tensor T where T is a 3D array of shape (n, m, p)
    # such that T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    # In the schoolbook, for each (i,j,k), we have a term A[i,j]*B[j,k].
    # So the tensor T has T[i, j, k] = 1 for all valid i, j, k?
    # No, that's not right. The tensor T is the "structure tensor".
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the term A_{i,j} B_{j,k} appears in C_{i,k} with coefficient 1.
    # So T[i, j, k] = 1 for all i, j, k?
    # Let's check the dimensions.
    # U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    # W is (p*n, rank). Index k*n+i.
    # The decomposition is:
    # C_{k*n+i, r} ... wait, the prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W is a matrix of shape (p*n, rank).
    # The reconstruction is likely:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # Using the factors:
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} ... this sums over r, but j is fixed?
    # No, the standard decomposition of the matrix multiplication tensor T is:
    # T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r} ?
    # If U, V, W are (n, rank), (m, rank), (p, rank).
    # But the prompt specifies shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # This suggests a different unfolding or a specific convention.
    # Let's look at the seed code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
    # The "index" in U is (i,j), in V is (j,k), in W is (k,i).
    # The product U_{i,j, r} V_{j,k, r} W_{k,i, r} is summed over r.
    # For the schoolbook, for a fixed (i,j,k), we have a rank-1 tensor that is 1 at (i,j), (j,k), (k,i) and 0 elsewhere.
    # The sum of these rank-1 tensors over all (i,j,k) gives the identity tensor for the multiplication?
    # Let's verify:
    # Sum_{i,j,k} (A_{i,j} B_{j,k} C_{k,i}^T?)
    # Actually, the evaluator likely checks:
    # For all A, B: C = A B.
    # Is it true that C_{i,k} = sum_j sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} A_{i,j} B_{j,k} ?
    # No, that would sum over j twice.
    # Let's assume the standard definition:
    # The tensor T is a 3D array T[i, j, k].
    # The decomposition is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    # But the shapes are (n*m, rank).
    # This implies U is indexed by (i,j).
    # So the tensor being decomposed is likely T_{i, j, k} where the "input" is a vector of length n*m (flattened A),
    # "middle" is a vector of length m*p (flattened B), and "output" is p*n (flattened C^T?).
    # Let's look at the seed again.
    # U[i*m+j] = 1. V[j*p+k] = 1. W[k*n+i] = 1.
    # If we take inner product of U_col, V_col, W_col with A_flat, B_flat, C_flat?
    # No, it's a rank-1 tensor in the space of matrices?
    # A rank-1 tensor in the space of (n x m) x (m x p) x (p x n) matrices?
    # If U is a matrix of size n*m, it can be seen as a vector.
    # Let's assume the evaluator constructs the tensor T of size (n*m, m*p, p*n) ?
    # No, the rank is the number of terms.
    # The seed has rank n*m*p = 72.
    # The tensor is the "matrix multiplication tensor".
    # The standard rank of the matrix multiplication tensor M_{n,m,p} is denoted R(n,m,p).
    # For 3x4x6, the rank is known to be <= 72 (schoolbook).
    # Can we do better?
    # Strassen's algorithm works for 2x2.
    # For 3x3, rank is 23 (Lader).
    # For 3x4x6, we can try to find a decomposition.
    
    # Let's define the target tensor T as a 3D array of shape (n*m, m*p, p*n) ?
    # No, that would be huge (12 * 24 * 18 = 5184).
    # The tensor T_{i, j, k} usually refers to the coefficients.
    # Let's stick to the vectorized form of the tensor of shape (n, m, p) flattened to (n*m*p).
    # T_flat[idx] = 1 if the term exists.
    # In the seed, for each (i,j,k), there is a term.
    # So the target vector is all ones?
    # Let's check:
    # If T is all ones, then sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k?
    # And 0 otherwise?
    # The indices i, j, k are coupled.
    # In the schoolbook, for a fixed (i,j,k), U has 1 at i*m+j, V at j*p+k, W at k*n+i.
    # The "tensor" is a set of such triplets.
    # We want to find a set of rank-1 tensors (triplets of vectors) that sum to the target tensor.
    # What is the target tensor?
    # It is the tensor that computes C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a multilinear map.
    # The tensor T has components T_{i,j,k} such that C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}?
    # No, T_{i,j,k} is 1 if j is the middle index.
    # So T_{i,j,k} = 1 for all i, j, k?
    # If we flatten A to a, B to b, C to c.
    # c = M a b ?
    # The tensor T is of size (n*m) x (m*p) x (p*n)?
    # Let's assume the target tensor is T of shape (n*m, m*p, p*n) where
    # T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise.
    # This is a sparse tensor.
    # We want to decompose this sparse tensor into a sum of rank-1 tensors.
    # A rank-1 tensor is u v w^T (outer product).
    # We want sum_r u_r v_r w_r^T = T.
    
    # This is a sparse tensor decomposition problem.
    # We can use a randomized algorithm to find a low-rank decomposition.
    # 1. Identify the support of T. The support is the set of (a, b, c) indices where T[a,b,c] = 1.
    #    Support size is n*m*p = 72.
    # 2. We want to find vectors u_r, v_r, w_r such that sum_r u_r v_r w_r^T = T.
    #    This implies that for any (a,b,c) not in support, sum_r u_r[a] v_r[b] w_r[c] = 0.
    #    And for (a,b,c) in support, sum_r u_r[a] v_r[b] w_r[c] = 1.
    
    # We can formulate this as a system of linear equations if we fix the sparsity pattern of u, v, w.
    # Or we can use a greedy approach.
    
    # Given the small size, we can try to find a decomposition of rank R.
    # We start with R = 20, 25, 30... and try to solve.
    # If we find a solution, we return it.
    
    # To solve for rank R:
    # We need to find R vectors u_r (len n*m), v_r (len m*p), w_r (len p*n).
    # This is a non-linear system.
    # However, we can fix the values of u and v, and solve for w?
    # No, it's coupled.
    
    # Alternative: Use the fact that the tensor is sparse.
    # We can try to find a decomposition where u, v, w are also sparse.
    
    # Let's try a simple heuristic:
    # 1. Generate random sparse vectors for u, v, w.
    # 2. Check if they can form a decomposition.
    
    # Actually, there is a known result for 3x4x6.
    # The rank is likely around 20-30.
    # Let's try to find a valid decomposition by searching.
    
    # We will use a "peeling" method on the support.
    # The support is a set of 72 points in 3D grid (12, 24, 18).
    # We want to cover these points with rank-1 tensors.
    # A rank-1 tensor u v w^T has support (support(u) x support(v) x support(w)).
    # We want the sum of these supports to match the target support, and the values to match.
    
    # Since the target values are all 1, and we can use integers, we can try to find a partition of the support into "boxes" (Cartesian products of sets) such that the sum of indicators of these boxes is the target indicator.
    # This is equivalent to finding a 0-1 decomposition if we restrict u,v,w to be 0-1 vectors.
    # If we allow -1, we can have cancellations.
    
    # Let's try to find a 0-1 decomposition first.
    # This is the "Boolean rank" or "Combinatorial Rank".
    # If we find a 0-1 decomposition of rank R, it is a valid integer decomposition.
    
    # Algorithm for 0-1 decomposition:
    # 1. Start with the set of target points S.
    # 2. Find a box B = I x J x K such that B is a subset of S? No, B can overlap.
    #    We want sum of boxes = S.
    #    This is difficult.
    
    # Let's try a different approach:
    # Use the known decomposition for smaller matrices and pad?
    # 3x3x3 rank 23.
    # 3x4x6 can be seen as 3x4x2 (3 times) + 3x4x2 (3 times)?
    # No.
    
    # Let's try to run a simple local search / greedy algorithm to find a low rank decomposition.
    # We will represent the decomposition as a list of triplets (u, v, w).
    # We will start with the schoolbook (72 triplets, all unit vectors).
    # We will try to merge or replace triplets to reduce the count.
    
    # Given the time constraint, a robust approach is to use a known low-rank algorithm for small tensors.
    # For 3x4x6, a rank of 24 is achievable?
    # Let's try to construct a rank 24 decomposition using a heuristic.
    
    # Heuristic:
    # 1. Create a pool of candidate rank-1 tensors.
    # 2. Greedily select tensors that cover the most "uncovered" mass of the target tensor.
    # 3. Adjust coefficients to exactly match.
    
    # Since the target is sparse (1s), and we want exact match, we can use a linear programming approach if we fix the support of u, v, w.
    
    # Let's try a specific structured decomposition.
    # For 3x4x6, we can use the fact that 6 = 2*3.
    # Split the 6 columns into two groups of 3.
    # C1 = A B1 (3x3), C2 = A B2 (3x3).
    # Rank(3x3x3) = 23.
    # Total rank = 23 + 23 = 46?
    # Can we share factors?
    # If we use the same U for both, we save rank?
    # U is (n*m, rank).
    # If we use the same U, then the rank is sum of ranks of V, W parts?
    # No, the rank is the number of terms in the sum.
    # If we have sum_{r=1}^{23} u_r v_{1,r} w_{1,r} + sum_{r=1}^{23} u_r v_{2,r} w_{2,r}.
    # This is 46 terms.
    # Can we do better?
    
    # Let's try to find a rank 30 decomposition.
    
    # We will implement a simple search for a valid decomposition of rank R.
    # We will try R = 20, 25, 30, 35, 40...
    # For each R, we will try to find u, v, w.
    
    # To make it fast, we will use a randomized greedy method.
    
    def get_target_tensor():
        # Returns a dictionary mapping (a, b, c) to value 1.
        # a in 0..n*m-1, b in 0..m*p-1, c in 0..p*n-1.
        target = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[(a, b, c)] = 1
        return target

    target = get_target_tensor()
    support = list(target.keys())
    
    # We want to find R triplets (u_r, v_r, w_r) such that
    # sum_r u_r[a] v_r[b] w_r[c] = 1 if (a,b,c) in target, else 0.
    
    # We will try to find a decomposition with small integer entries.
    
    # Strategy:
    # 1. Generate random sparse vectors for u, v, w.
    # 2. Use a greedy algorithm to select R vectors that best approximate the target.
    # 3. Solve for the exact coefficients using least squares (or exact linear algebra) on the selected vectors.
    
    best_rank = n * m * p
    best_decomp = None
    
    # Fallback: Schoolbook
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_decomp = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better ranks
    for rank in range(20, 50):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Generate random sparse candidates
            # We want u, v, w to be sparse to make the system easier.
            num_candidates = 200
            
            # We will pick R candidates from a pool and solve for them.
            # To ensure we can solve, we need the "tensor" formed by the candidates to span the target.
            
            # Let's generate a pool of random sparse vectors.
            # Sparsity: 1-2 non-zeros per vector.
            
            pool_u = []
            pool_v = []
            pool_w = []
            
            for _ in range(num_candidates):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                
                # Pick random indices
                idx_u = random.sample(range(n*m), k=random.randint(1, 2))
                idx_v = random.sample(range(m*p), k=random.randint(1, 2))
                idx_w = random.sample(range(p*n), k=random.randint(1, 2))
                
                for idx in idx_u:
                    u[idx] = random.choice([-1, 1])
                for idx in idx_v:
                    v[idx] = random.choice([-1, 1])
                for idx in idx_w:
                    w[idx] = random.choice([-1, 1])
                    
                pool_u.append(u)
                pool_v.append(v)
                pool_w.append(w)
            
            # Select R candidates
            selected_indices = random.sample(range(num_candidates), k=rank)
            
            u_sel = np.array([pool_u[i] for i in selected_indices]) # (R, n*m)
            v_sel = np.array([pool_v[i] for i in selected_indices]) # (R, m*p)
            w_sel = np.array([pool_w[i] for i in selected_indices]) # (R, p*n)
            
            # We want to find coefficients c_r such that
            # sum_r c_r (u_sel[r] outer v_sel[r] outer w_sel[r]) = T
            # This is a linear system in c_r?
            # No, c_r multiplies the whole tensor.
            # So we are looking for c_r such that
            # sum_r c_r T_r = T_target
            # This is a linear system: A c = b
            # Where A is a matrix of size (num_nonzeros, R).
            # b is the target values.
            
            # Build the system
            # For each (a,b,c) in the union of supports of selected tensors and target:
            # sum_r c_r u_sel[r][a] v_sel[r][b] w_sel[r][c] = T_target[a,b,c]
            
            # This is a linear system in c.
            # We can solve it using least squares.
            
            # To make it faster, we only consider the points in the target support and the points in the selected supports.
            # Actually, we must ensure that for points NOT in target, the sum is 0.
            
            # Let's collect all unique points (a,b,c) that appear in any selected tensor or the target.
            points = set()
            for r in range(rank):
                for a in range(n*m):
                    if u_sel[r][a] != 0:
                        for b in range(m*p):
                            if v_sel[r][b] != 0:
                                for c in range(p*n):
                                    if w_sel[r][c] != 0:
                                        points.add((a,b,c))
            
            for pt in target.keys():
                points.add(pt)
            
            points = list(points)
            
            if len(points) == 0:
                continue
                
            A = np.zeros((len(points), rank))
            b = np.zeros(len(points))
            
            for idx, pt in enumerate(points):
                a, b_idx, c_idx = pt
                for r in range(rank):
                    val = u_sel[r][a] * v_sel[r][b_idx] * w_sel[r][c_idx]
                    A[idx, r] = val
                if pt in target:
                    b[idx] = target[pt]
                else:
                    b[idx] = 0
            
            # Solve A c = b
            # Use least squares
            try:
                c, residuals, rank_sys, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check if the solution is exact
            # We need integer or half-integer entries.
            # If the solution c is close to integers, we can round and verify.
            
            # Check residuals
            if residuals is None or len(residuals) == 0:
                # Perfect fit?
                pass
            else:
                if residuals[0] > 1e-9:
                    continue
            
            # Verify exactness with rounded integers
            c_rounded = np.round(c)
            if not np.allclose(A @ c_rounded, b, atol=1e-9):
                # Try half-integers?
                c_half = np.round(c * 2) / 2.0
                if not np.allclose(A @ c_half, b, atol=1e-9):
                    continue
            
            # If we get here, we have a valid decomposition
            # Construct U, V, W
            U_new = np.zeros((n*m, rank))
            V_new = np.zeros((m*p, rank))
            W_new = np.zeros((p*n, rank))
            
            for r in range(rank):
                # The tensor term is c_r * u_sel[r] outer v_sel[r] outer w_sel[r]
                # So U[:, r] = c_r * u_sel[r]
                U_new[:, r] = c_rounded[r] * u_sel[r]
                V_new[:, r] = c_rounded[r] * v_sel[r]
                W_new[:, r] = c_rounded[r] * w_sel[r]
                
            # Final verification
            # Reconstruct T and compare
            T_recon = {}
            for a in range(n*m):
                if np.any(U_new[a] != 0):
                    for b_idx in range(m*p):
                        if np.any(V_new[b_idx] != 0):
                            for c_idx in range(p*n):
                                if np.any(W_new[c_idx] != 0):
                                    val = 0
                                    for r in range(rank):
                                        val += U_new[a, r] * V_new[b_idx, r] * W_new[c_idx, r]
                                    if val != 0:
                                        T_recon[(a, b_idx, c_idx)] = val
                                    
            # Check if T_recon matches target
            match = True
            for pt in target:
                if T_recon.get(pt, 0) != 1:
                    match = False
                    break
            if match:
                for pt in T_recon:
                    if pt not in target:
                        match = False
                        break
            
            if match:
                best_rank = rank
                best_decomp = {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
                found = True
                break
        
        if found:
            break

    return best_decomp
# EVOLVE-BLOCK-END
