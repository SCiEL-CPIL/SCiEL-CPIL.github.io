import numpy as np
import time
import random
import itertools

DIMENSIONS = [4, 4, 5]

def _time_left(deadline):
    return deadline - time.time()

def _tensor_values(U, V, W):
    """Compute the tensor from factor matrices."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = np.array(U)[:, r]
        v = np.array(V)[:, r]
        w = np.array(W)[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    return T

def _is_valid(U, V, W, tolerance=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    T = _tensor_values(U, V, W)
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k} (for identity in the middle)
    # Actually, for multiplying A (n×m) by B (m×p): C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 if this term contributes, which is always 1 for the product
    # Wait, the multiplication tensor T_{ijk} satisfies: C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} = 1 for all i,j,k (it's the all-ones tensor in this case)
    # Actually no. Let me reconsider.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is a trilinear form. The tensor T has T[i,j,k] = 1 for all i,j,k.
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tolerance, rtol=0)

def _structural_lower_bound(n, m, p):
    return max(n * p, m * p, n * m)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _greedy_build(deadline):
    """Try to build a low-rank decomposition greedily."""
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try random greedy approaches
    attempts = 0
    while _time_left(deadline) > 5.0 and attempts < 50:
        attempts += 1
        random.seed(attempts * 12345 + 678)
        
        # Start with a random subset of schoolbook terms
        all_terms = list(range(n * m * p))
        random.shuffle(all_terms)
        
        # Try building up from scratch
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        
        # Greedily add terms
        used_terms = set()
        for term_idx in all_terms:
            i = term_idx // (m * p)
            j = (term_idx // p) % m
            k = term_idx % p
            
            col = len(used_terms)
            if col >= best_rank:
                break
            
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            used_terms.add(term_idx)
        
        if _is_valid(U, V, W):
            if len(used_terms) < best_rank:
                best_rank = len(used_terms)
                best_result = {"U": U, "V": V, "W": W}
        
        if _time_left(deadline) < 10.0:
            break
    
    return best_result

def construct():
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try greedy search
    if _time_left(deadline) > 10.0:
        result = _greedy_build(deadline)
        if len(result["U"][0]) < best_rank:
            best = result
            best_rank = len(result["U"][0])
    
    return best

