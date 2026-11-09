"""Program for matrix multiplication tensor rank minimization (3x4, 4x5).

Approach:
1. Initialize with the schoolbook decomposition (rank = n*m*p = 60).
2. Attempt a greedy iterative merging procedure: for each pair of components,
   check whether their sum can be represented by a single rank-1 tensor with
   half-integer entries. If so, replace the pair with that single tensor.
3. After each merge, re-normalize and continue until no further merges are possible.

The W matrix follows the trace convention W[k*n+i, r] for the (k, i) output slot.

This is a heuristic search; it is guaranteed to terminate and produce a valid
decomposition (possibly the original schoolbook one if no merges succeed).
"""

import numpy as np
from fractions import Fraction
import itertools
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # --- Step 1: Schoolbook initialization ---
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    components = []
    for c in range(rank):
        components.append((
            U[:, c].copy(),
            V[:, c].copy(),
            W[:, c].copy()
        ))

    def to_half_integer(x, tol=1e-9):
        if abs(x - round(2 * x) * 0.5) < tol:
            return round(2 * x) * 0.5
        return None

    def try_merge_two(comp_a, comp_b):
        u_a, v_a, w_a = comp_a
        u_b, v_b, w_b = comp_b
        
        T = (np.einsum('i,j,k->ijk', u_a, v_a, w_a) + 
             np.einsum('i,j,k->ijk', u_b, v_b, w_b))
        
        nonzero_mask = T != 0
        if not np.any(nonzero_mask):
            return (np.zeros(nm), np.zeros(mp), np.zeros(pn))
        
        idx = np.argwhere(nonzero_mask)[0]
        i0, j0, k0 = idx
        s = T[i0, j0, k0]
        
        u = T[:, j0, k0] / s
        v = T[i0, :, k0] / s
        w = T[i0, j0, :] / s
        
        T_reconstructed = np.einsum('i,j,k->ijk', u, v, w)
        if not np.allclose(T, T_reconstructed, atol=1e-8):
            return None
        
        for arr in [u, v, w]:
            for val in arr:
                if to_half_integer(val) is None:
                    return None
        
        u = np.array([to_half_integer(val) for val in u])
        v = np.array([to_half_integer(val) for val in v])
        w = np.array([to_half_integer(val) for val in w])
        
        return (u, v, w)

    def greedy_reduce(comps, max_passes=20):
        components = [c for c in comps]
        merged = True
        passes = 0
        while merged and passes < max_passes:
            merged = False
            passes += 1
            
            valid_merges = []
            for a in range(len(components)):
                for b in range(a + 1, len(components)):
                    result = try_merge_two(components[a], components[b])
                    if result is not None:
                        u, v, w = result
                        if not (np.allclose(u, 0) and np.allclose(v, 0) and np.allclose(w, 0)):
                            valid_merges.append((a, b, result))
            
            if not valid_merges:
                break
                
            random.shuffle(valid_merges)
            
            used = [False] * len(components)
            new_components = []
            
            for a, b, result in valid_merges:
                if not used[a] and not used[b]:
                    new_components.append(result)
                    used[a] = True
                    used[b] = True
                    merged = True
            
            for idx in range(len(components)):
                if not used[idx]:
                    new_components.append(components[idx])
            
            components = new_components
            
        return components

    best_components = components
    best_rank = len(components)
    
    for _ in range(10):
        current_components = greedy_reduce(components)
        if len(current_components) < best_rank:
            best_rank = len(current_components)
            best_components = current_components
            
    final_rank = len(best_components)
    U_out = np.zeros((nm, final_rank), dtype=np.float64)
    V_out = np.zeros((mp, final_rank), dtype=np.float64)
    W_out = np.zeros((pn, final_rank), dtype=np.float64)

    for c, (u, v, w) in enumerate(best_components):
        U_out[:, c] = u
        V_out[:, c] = v
        W_out[:, c] = w

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
