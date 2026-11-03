"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V3 = V_arr.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # Compute sum over rank dimension
    # For each (i, j, k), we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Let's build the full tensor T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # This is a bit tricky with numpy broadcasting.
    # U3[i,j,c], V3[j,k,c], W3[k,i,c]
    # We want to sum over c.
    # Let's iterate over i, k and check all j? Or just compute the full tensor.
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Note that W3 depends on k and i, not j.
    # So for fixed i, k:
    # T[i,j,k] = W3[k,i,:] dot (U3[i,j,:] * V3[j,k,:])? No.
    # T[i,j,k] = sum_c (U3[i,j,c] * V3[j,k,c]) * W3[k,i,c]
    
    # Let's just use a loop for i, k and vectorize over j, c?
    # Or simply:
    # T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # This might be memory heavy if rank is large, but rank <= 100, n,m,p <= 5.
    # 5*5*5*100 = 12500 floats. Very small.
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if T is all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _try_rank(r, n, m, p, max_iters=1000, timeout=10, use_numpy_verify=True):
    """Try to find a rank-r decomposition using random initialization and verification."""
    start_time = time.time()
    verify_func = _verify_numpy if use_numpy_verify else _verify
    
    # Pre-compute indices for potential optimization, but random is fine
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Random entries from {0, 1, -1}
        # Using numpy for generation speed
        U = np.random.randint(0, 3, size=(n * m, r)) - 1
        V = np.random.randint(0, 3, size=(m * p, r)) - 1
        W = np.random.randint(0, 3, size=(p * n, r)) - 1
        
        # Convert to list of lists for interface if needed, or pass to numpy verify
        if use_numpy_verify:
            if _verify_numpy(U, V, W, n, m, p):
                # Convert back to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        else:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def _try_rank_structured(r, n, m, p, max_iters=100, timeout=10):
    """
    Try a structured approach: 
    Start with schoolbook and try to merge terms or use sparsity.
    This is heuristic.
    """
    start_time = time.time()
    
    # For 4x5x5, rank 100 is schoolbook.
    # Can we do better?
    # Bini et al. showed rank 48 for 3x3x3.
    # For 4x5x5, known bounds?
    # Let's try a sparse random search with a bias towards sparsity.
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Each column is a "term".
        # A term is defined by a pattern of 1s and -1s.
        # To be efficient, terms should be sparse.
        # Let's pick random subsets of indices for U, V, W.
        
        for c in range(r):
            # Randomly decide sparsity
            # Probability of non-zero
            p_nonzero = 0.2 + 0.3 * random.random()
            
            # Pick indices for U
            # U has n*m entries.
            num_u = max(1, int(n * m * p_nonzero))
            u_indices = np.random.choice(n * m, size=num_u, replace=False)
            u_vals = np.random.choice([-1, 1], size=num_u)
            U[u_indices, c] = u_vals
            
            num_v = max(1, int(m * p * p_nonzero))
            v_indices = np.random.choice(m * p, size=num_v, replace=False)
            v_vals = np.random.choice([-1, 1], size=num_v)
            V[v_indices, c] = v_vals
            
            num_w = max(1, int(p * n * p_nonzero))
            w_indices = np.random.choice(p * n, size=num_w, replace=False)
            w_vals = np.random.choice([-1, 1], size=num_w)
            W[w_indices, c] = w_vals
            
        if _verify_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best['U'][0])
    
    # We want to find a rank < best_rank
    # Lower bound for rank of 4x5x5?
    # Rank is at least max(n*m, m*p, p*n) = max(20, 25, 20) = 25? No, that's for border rank or something?
    # Actually rank >= max(n*m, m*p, p*n) is not true.
    # But rank >= n*m + m*p + p*n - ...?
    # Let's just search downwards from 99.
    
    # Random search is very unlikely to find a solution for rank close to 100.
    # We need smarter search or known decompositions.
    
    # However, the task asks for "rank-minimal". 
    # For 4x5x5, the minimal rank is likely significantly lower than 100.
    # But finding it randomly is hard.
    
    # Let's try a few ranks.
    # If we can't find a better one, we return schoolbook.
    
    # Try ranks 90 down to 80 with random search
    # This is a long shot, but maybe we get lucky with a small improvement?
    # Or maybe we should try to verify if a specific known low rank exists?
    # No, we have to generate it.
    
    # Let's try a few iterations of random search for slightly lower ranks.
    # And structured sparse search.
    
    # Time budget allocation:
    # 60s for random dense
    # 60s for random sparse
    # 20s buffer
    
    # Try dense random
    for r in range(99, 95, -1):
        if time.time() > deadline - 60:
            break
        result = _try_rank(r, n, m, p, max_iters=2000, timeout=5, use_numpy_verify=True)
        if result is not None:
            best = result
            best_rank = r
            break
            
    # Try sparse random
    for r in range(95, 85, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank_structured(r, n, m, p, max_iters=500, timeout=5)
        if result is not None:
            best = result
            best_rank = r
            break

    return best
# EVOLVE-BLOCK-END
