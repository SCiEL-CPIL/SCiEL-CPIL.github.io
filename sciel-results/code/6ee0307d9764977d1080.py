import numpy as np
from fractions import Fraction
import itertools
import time

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

    # Store as list of (U_col, V_col, W_col) for easier manipulation
    components = []
    for c in range(rank):
        components.append((
            U[:, c].copy(),
            V[:, c].copy(),
            W[:, c].copy()
        ))

    # --- Step 2: Greedy merge procedure ---
    # For each pair of components, check if their sum is rank-1 with half-integer entries.
    # We iterate multiple passes to allow chain merges.
    
    def to_half_integer(x, tol=1e-9):
        """Check if x is a half-integer (multiple of 0.5) within tolerance."""
        if abs(x - round(2 * x) * 0.5) < tol:
            return round(2 * x) * 0.5
        return None

    def try_merge_two(comp_a, comp_b):
        """
        Try to find a single rank-1 tensor (u, v, w) with half-integer entries
        such that u⊗v⊗w = comp_a + comp_b.
        Returns (u, v, w) or None.
        """
        u_a, v_a, w_a = comp_a
        u_b, v_b, w_b = comp_b

        # Sum tensors: T = T_a + T_b
        # We need T = u ⊗ v ⊗ w
        # Approach: find u, v, w by solving linear equations.
        # If T is rank-1, then any nonzero slice should be proportional.
        
        # Collect all nonzero entries of T
        # T[i,j,k] = u_a[i,j]*v_a[j,k]*w_a[k,i] + u_b[i,j]*v_b[j,k]*w_b[k,i]
        # But we don't have T explicitly; we have the outer products.
        
        # Let's build T as a 3D array of size (n*m, m*p, p*n) — too large? 
        # n*m=12, m*p=20, p*n=15 => 12*20*15 = 3600 entries. Fine.
        
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i_m in range(nm):
            for j_p in range(mp):
                for k_n in range(pn):
                    T[i_m, j_p, k_n] = (u_a[i_m] * v_a[j_p] * w_a[k_n] +
                                         u_b[i_m] * v_b[j_p] * w_b[k_n])
        
        # Check if T is rank-1: all 2x2 minors of any unfolding are zero.
        # Simpler: pick a nonzero entry T[i0,j0,k0] = s. Then u[i,j] should be T[i,j,k0]/(s/w[k0])... 
        # Actually, if T = u⊗v⊗w, then T[i,j,k] = u[i]*v[j]*w[k].
        # Find a nonzero entry.
        nonzero_mask = T != 0
        if not np.any(nonzero_mask):
            # T is zero — can merge into nothing? Just return a zero component.
            return (np.zeros(nm), np.zeros(mp), np.zeros(pn))
        
        # Find indices of a nonzero entry
        idx = np.argwhere(nonzero_mask)[0]
        i0, j0, k0 = idx
        s = T[i0, j0, k0]
        
        # Assume T = u ⊗ v ⊗ w with u[i0]=1 (scale choice), then:
        # w[k] = T[i0, j0, k] / (u[i0]*v[j0]) = T[i0,j0,k] / v[j0] if u[i0]=1
        # v[j] = T[i0, j, k0] / (u[i0]*w[k0]) = T[i0,j,k0] / w[k0]
        # u[i] = T[i, j0, k0] / (v[j0]*w[k0])
        
        # But we don't know the scale. Let's try: set u[i0] = 1.
        # Then v[j0] = T[i0,j0,k0] / w[k0] ... circular.
        
        # Better: use the fact that for rank-1, T[:,j0,k0] = u * (v[j0]*w[k0])
        # and T[i0,:,k0] = v * (u[i0]*w[k0])
        # and T[i0,j0,:] = w * (u[i0]*v[j0])
        
        # Let a = u[i0]*v[j0]*w[k0] = s
        # T[i, j0, k0] = u[i] * v[j0] * w[k0] = u[i] * (a / u[i0]) = u[i] * a (since u[i0]=1)
        # So u[i] = T[i, j0, k0] / a
        
        # Similarly, T[i0, j, k0] = v[j] * a => v[j] = T[i0,j,k0] / a
        # T[i0, j0, k] = w[k] * a => w[k] = T[i0,j0,k] / a
        
        u = T[:, j0, k0] / s
        v = T[i0, :, k0] / s
        w = T[i0, j0, :] / s
        
        # Verify: check that T[i,j,k] ≈ u[i]*v[j]*w[k] for all i,j,k
        T_reconstructed = np.einsum('i,j,k->ijk', u, v, w)
        if not np.allclose(T, T_reconstructed, atol=1e-8):
            return None
        
        # Check half-integer entries
        for arr in [u, v, w]:
            for val in arr:
                if to_half_integer(val) is None:
                    return None
        
        # Convert to half-integers
        u = np.array([to_half_integer(val) for val in u])
        v = np.array([to_half_integer(val) for val in v])
        w = np.array([to_half_integer(val) for val in w])
        
        return (u, v, w)

    # Greedy merge: repeatedly try to merge pairs
    merged = True
    while merged:
        merged = False
        new_components = []
        used = [False] * len(components)
        
        for a in range(len(components)):
            if used[a]:
                continue
            for b in range(a + 1, len(components)):
                if used[b]:
                    continue
                result = try_merge_two(components[a], components[b])
                if result is not None:
                    # Check if the result is non-trivial (not all zeros)
                    u, v, w = result
                    if not (np.allclose(u, 0) and np.allclose(v, 0) and np.allclose(w, 0)):
                        new_components.append(result)
                        used[a] = True
                        used[b] = True
                        merged = True
                        break
            if merged:
                break
        
        if merged:
            # Keep unused components
            for idx in range(len(components)):
                if not used[idx]:
                    new_components.append(components[idx])
            components = new_components
        else:
            break

    # --- Step 3: Build output matrices from components ---
    final_rank = len(components)
    U_out = np.zeros((nm, final_rank), dtype=np.float64)
    V_out = np.zeros((mp, final_rank), dtype=np.float64)
    W_out = np.zeros((pn, final_rank), dtype=np.float64)

    for c, (u, v, w) in enumerate(components):
        U_out[:, c] = u
        V_out[:, c] = v
        W_out[:, c] = w

    # Convert to lists for the required return format
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
