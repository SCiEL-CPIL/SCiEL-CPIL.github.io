"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check all valid (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1) > tol:
                        return False
        # Check zero entries - sample some
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(dim_u)
            b = rng.randint(dim_v)
            c = rng.randint(dim_w)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if not (j == j2 and k == k2 and i == i2):
                val = np.sum(U[a,:] * V[b,:] * W[c,:])
                if abs(val) > tol:
                    return False
        return True
    
    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2
    
    def check_full_zero(U, V, W, tol=1e-10):
        """Check all zero entries are actually zero."""
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    # Strategy 1: Try structured decompositions based on tensor structure
    # The key insight: for fixed i, the problem involves U block i (m entries), 
    # all of V, and W indices {k*n+i for k in 0..p-1}
    
    # Try rank = m*p for each i independently, but share V
    # This gives total rank = n * m * p = schoolbook, so no improvement
    
    # Better: try to find a shared decomposition
    # For the 2x3 x 3x9 case, let's try specific structured forms
    
    # Strategy 2: Use the fact that we can factor through intermediate dimensions
    # Try rank = n * p + m * p + n * m or similar
    
    # Let's try systematic rank reduction from schoolbook
    # For each rank, try multiple random initializations
    
    # Precompute the valid (i,j,k) triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                valid_triples.append((a, b, c))
    
    # Precompute invalid triples (sample)
    rng = np.random.RandomState(12345)
    invalid_triples = []
    for _ in range(500):
        a = rng.randint(dim_u)
        b = rng.randint(dim_v)
        c = rng.randint(dim_w)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (j == j2 and k == k2 and i == i2):
            invalid_triples.append((a, b, c))
    
    def loss_and_grad(U, V, W):
        """Compute loss and gradients for the decomposition."""
        r = U.shape[1]
        # Loss = sum over valid (sum - 1)^2 + sum over invalid (sum)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Valid triples
        for (a, b, c) in valid_triples:
            s = np.sum(U[a,:] * V[b,:] * W[c,:])
            err = s - 1.0
            loss += err * err
            # Gradient
            grad_U[a,:] += 2 * err * V[b,:] * W[c,:]
            grad_V[b,:] += 2 * err * U[a,:] * W[c,:]
            grad_W[c,:] += 2 * err * U[a,:] * V[b,:]
        
        # Invalid triples (sample)
        for (a, b, c) in invalid_triples:
            s = np.sum(U[a,:] * V[b,:] * W[c,:])
            err = s
            loss += err * err
            grad_U[a,:] += 2 * err * V[b,:] * W[c,:]
            grad_V[b,:] += 2 * err * U[a,:] * W[c,:]
            grad_W[c,:] += 2 * err * U[a,:] * V[b,:]
        
        return loss, grad_U, grad_V, grad_W
    
    def optimize(U, V, W, lr=0.01, max_iter=1000):
        """Simple gradient descent optimization."""
        for it in range(max_iter):
            if time.time() > deadline:
                break
            loss, gU, gV, gW = loss_and_grad(U, V, W)
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            if loss < 1e-12:
                break
        return U, V, W
    
    # Try ranks from schoolbook down
    # First, try some specific structured ranks
    candidate_ranks = [schoolbook_rank]
    # Try ranks that are multiples or related to the dimensions
    for r in range(schoolbook_rank - 1, max(0, schoolbook_rank - 20), -1):
        candidate_ranks.append(r)
    
    # Also try some smaller ranks based on structural insights
    # For 2x3 x 3x9, the tensor rank might be related to n*p + m*p + ... 
    # Let's try a few more
    for r in [27, 26, 25, 24, 23, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 5, 4, 3]:
        if r not in candidate_ranks and r < schoolbook_rank:
            candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks), reverse=True)
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        found = False
        max_trials = 30 if rank >= schoolbook_rank - 5 else 50
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            rng = np.random.RandomState(trial * 7 + 13)
            
            if trial == 0 and rank == schoolbook_rank:
                # Use schoolbook initialization
                U = U_sb[:, :rank].copy()
                V = V_sb[:, :rank].copy()
                W = W_sb[:, :rank].copy()
            elif trial == 1:
                # Structured initialization: try to exploit block structure
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                # For each i, try to use a subset of columns
                cols_per_i = rank // n
                for i in range(n):
                    start_c = i * cols_per_i
                    end_c = (i + 1) * cols_per_i if i < n - 1 else rank
                    for j in range(m):
                        for k in range(p):
                            c_idx = start_c + (j * p + k) % (end_c - start_c)
                            if c_idx < rank:
                                U[i*m+j, c_idx] = 1.0
                                V[j*p+k, c_idx] = 1.0
                                W[k*n+i, c_idx] = 1.0
            elif trial == 2:
                # Random with structure: initialize U and V, solve for W
                U = rng.randn(dim_u, rank) * 0.5
                V = rng.randn(dim_v, rank) * 0.5
                # Solve for W: for each (a,b,c) valid, W[c,:] should satisfy
                # sum_d W[c,d] * U[a,d] * V[b,d] = 1
                # This is a system: for each c, we have equations from all (a,b) that map to c
                W = np.zeros((dim_w, rank))
                for c in range(dim_w):
                    # Find all (a,b) such that (a,b,c) is valid
                    equations = []
                    rhs = []
                    for (a, b, c2) in valid_triples:
                        if c2 == c:
                            row = U[a,:] * V[b,:]
                            equations.append(row)
                            rhs.append(1.0)
                    if equations:
                        A = np.array(equations)
                        b_vec = np.array(rhs)
                        try:
                            W[c,:], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                        except:
                            W[c,:] = rng.randn(rank) * 0.1
                # Optimize
                U, V, W = optimize(U, V, W, lr=0.005, max_iter=200)
            elif trial == 3:
                # Another structured init: use Kronecker-like structure
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                # Try to use rank = m*p for the "main" part
                for j in range(m):
                    for k in range(p):
                        c_idx = j * p + k
                        if c_idx < rank:
                            for i in range(n):
                                U[i*m+j, c_idx] = 1.0
                                V[j*p+k, c_idx] = 1.0
                                W[k*n+i, c_idx] = 1.0
                # Add remaining columns with random values
                for c in range(m*p, rank):
                    U[:, c] = rng.randn(dim_u) * 0.1
                    V[:, c] = rng.randn(dim_v) * 0.1
                    W[:, c] = rng.randn(dim_w) * 0.1
                U, V, W = optimize(U, V, W, lr=0.005, max_iter=200)
            else:
                # Random initialization
                scale = 0.3
                U = rng.randn(dim_u, rank) * scale
                V = rng.randn(dim_v, rank) * scale
                W = rng.randn(dim_w, rank) * scale
                # Optimize
                U, V, W = optimize(U, V, W, lr=0.01, max_iter=300)
            
            # Check if solution is valid
            loss, _, _ = loss_and_grad(U, V, W)
            if loss < 1e-8:
                # Round to half-integers
                U_r = round_to_half_int(U)
                V_r = round_to_half_int(V)
                W_r = round_to_half_int(W)
                
                # Verify rounded version
                if verify_decomposition(U_r, V_r, W_r, tol=1e-10):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
        else:
            # If we couldn't find for this rank, and we've tried several, break
            if rank < schoolbook_rank - 10:
                break
    
    # Final verification
    if not verify_decomposition(
        np.array(best_result["U"]), 
        np.array(best_result["V"]), 
        np.array(best_result["W"]), 
        tol=1e-10
    ):
        # Fallback to schoolbook
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        best_rank = schoolbook_rank
    
    return best_result
# EVOLVE-BLOCK-END

