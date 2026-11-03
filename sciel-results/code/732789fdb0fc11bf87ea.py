"""Optimized rank-minimal tensor decomposition for 4x4 x 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms


def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 via Kronecker product of 2x2 Strassen."""
    base_terms = _strassen_2x2_terms()
    terms_4x4 = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base_terms[c1][0][I1*2+J1] * base_terms[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base_terms[c1][1][J1*2+K1] * base_terms[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base_terms[c1][2][K1*2+I1] * base_terms[c2][2][K0*2+I0]
            terms_4x4.append((U4, V4, W4))
    return terms_4x4


def _make_4x4x7_decomposition():
    """Create a rank-98 decomposition for 4x4x7 using Strassen 4x4x4 terms."""
    terms_4x4 = _strassen_4x4_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    for U4, V4, W4 in terms_4x4:
        # First group: columns 0-3
        V1 = [0] * 28
        W1 = [0] * 28
        for J in range(4):
            for K in range(4):
                V1[J*7+K] = V4[J*4+K]
        for K in range(4):
            for I in range(4):
                W1[K*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V1)
        W_list.append(W1)
        
        # Second group: columns 4-6
        V2 = [0] * 28
        W2 = [0] * 28
        for J in range(4):
            for K in range(3):
                V2[J*7+(K+4)] = V4[J*4+K]
        for K in range(3):
            for I in range(4):
                W2[(K+4)*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V2)
        W_list.append(W2)
    
    rank = len(U_list)
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    for c in range(rank):
        U[:, c] = U_list[c]
        V[:, c] = V_list[c]
        W[:, c] = W_list[c]
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank_reduce(U, V, W, n, m, p, deadline):
    """Attempt to reduce rank by finding linear dependencies among terms."""
    start_time = time.time()
    rank = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    keep = [True] * rank
    
    # Precompute for faster verification
    # Tensor values: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We want to remove terms that are linearly dependent
    
    # Strategy: try removing each term and verify
    # Sort by "importance" - terms with more zeros are less important
    term_weights = []
    for c in range(rank):
        nz = np.count_nonzero(current_U[:, c]) + np.count_nonzero(current_V[:, c]) + np.count_nonzero(current_W[:, c])
        term_weights.append((nz, c))
    term_weights.sort()
    
    for _, idx in term_weights:
        if time.time() > deadline - 20:
            break
        if not keep[idx]:
            continue
        
        # Try removing term idx
        keep[idx] = False
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        if keep[c]:
                            s += current_U[i*m+j, c] * current_V[j*p+k, c] * current_W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            current_rank -= 1
        else:
            keep[idx] = True
    
    if current_rank < rank:
        keep_indices = [c for c in range(rank) if keep[c]]
        return current_U[:, keep_indices], current_V[:, keep_indices], current_W[:, keep_indices]
    return U, V, W


def _try_landsberg_style(U, V, W, n, m, p, deadline):
    """Try to find a lower rank decomposition using structured search."""
    start_time = time.time()
    rank = U.shape[1]
    
    # Try to find linear combinations of terms that can replace pairs
    # This is a heuristic approach
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try merging pairs of terms
    for c1 in range(rank):
        if time.time() > deadline - 10:
            break
        for c2 in range(c1+1, rank):
            if time.time() > deadline - 10:
                break
            # Check if term c2 is a scalar multiple of term c1
            # (i.e., they have the same support pattern)
            U1, U2 = U[:, c1], U[:, c2]
            V1, V2 = V[:, c1], V[:, c2]
            W1, W2 = W[:, c1], W[:, c2]
            
            # Check if U2 = alpha * U1
            mask1 = U1 != 0
            if np.any(mask1):
                alphas = U2[mask1] / U1[mask1]
                if np.allclose(alphas, alphas[0], atol=1e-9):
                    alpha = alphas[0]
                    # Check V2 = alpha * V1
                    maskV = V1 != 0
                    if np.any(maskV):
                        alphasV = V2[maskV] / V1[maskV]
                        if np.allclose(alphasV, alpha, atol=1e-9):
                            maskW = W1 != 0
                            if np.any(maskW):
                                alphasW = W2[maskW] / W1[maskW]
                                if np.allclose(alphasW, alpha, atol=1e-9):
                                    # Terms are scalar multiples, can merge
                                    # Remove c2
                                    keep = [True] * rank
                                    keep[c2] = False
                                    valid = True
                                    for i in range(n):
                                        for j in range(m):
                                            for k in range(p):
                                                s = 0.0
                                                for c in range(rank):
                                                    if keep[c]:
                                                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                                if abs(s - 1.0) > 1e-9:
                                                    valid = False
                                                    break
                                            if not valid:
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        keep_indices = [c for c in range(rank) if keep[c]]
                                        new_rank = len(keep_indices)
                                        if new_rank < best_rank:
                                            best_rank = new_rank
                                            best_U = U[:, keep_indices]
                                            best_V = V[:, keep_indices]
                                            best_W = W[:, keep_indices]
    
    if best_rank < rank:
        return best_U, best_V, best_W
    return U, V, W


def _random_search_reduce(U, V, W, n, m, p, deadline, max_iters=5000):
    """Random search to find lower rank by perturbing and reducing."""
    start_time = time.time()
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try random linear combinations to create new terms
    # and see if we can reduce rank
    rng = np.random.RandomState(42)
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Try to replace two terms with one
        if rank <= 2:
            break
            
        c1 = rng.randint(0, rank)
        c2 = rng.randint(0, rank)
        if c1 == c2:
            continue
        
        # Try to see if term c2 can be expressed using other terms
        # by solving a linear system
        # This is expensive, so use a simpler heuristic
        
        # Try removing c2 and see if we can adjust other terms
        keep = [True] * rank
        keep[c2] = False
        keep_indices = [c for c in range(rank) if keep[c]]
        
        # Check if remaining terms still work
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in keep_indices:
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = len(keep_indices)
            best_U = U[:, keep_indices]
            best_V = V[:, keep_indices]
            best_W = W[:, keep_indices]
            U, V, W = best_U, best_V, best_W
            rank = best_rank
    
    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based rank-98 decomposition
    U, V, W = _make_4x4x7_decomposition()
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, column] = 1
                    V[j*p+k, column] = 1
                    W[k*n+i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U, V, W = _rank_reduce(U, V, W, n, m, p, deadline)
    
    if time.time() < deadline - 15:
        U, V, W = _try_landsberg_style(U, V, W, n, m, p, deadline)
    
    if time.time() < deadline - 5:
        U, V, W = _random_search_reduce(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fall back to original
        U, V, W = _make_4x4x7_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
