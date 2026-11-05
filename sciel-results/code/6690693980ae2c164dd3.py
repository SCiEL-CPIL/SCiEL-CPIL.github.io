"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _get_rank17_solution():
    """Returns the known rank-17 solution using Strassen block."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_rank16_search(U17, V17, W17, T, deadline, max_trials=20000):
    """
    Search for a rank-16 decomposition by perturbing the rank-17 solution.
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Get supports for each term in rank-17
    supports_U = [set(np.nonzero(U17[:, r])[0]) for r in range(17)]
    supports_V = [set(np.nonzero(V17[:, r])[0]) for r in range(17)]
    supports_W = [set(np.nonzero(W17[:, r])[0]) for r in range(17)]
    
    all_U = set(range(n_a))
    all_V = set(range(n_b))
    all_W = set(range(n_c))
    
    trial = 0
    while trial < max_trials and time.time() < deadline:
        trial += 1
        
        # Remove one random term
        remove_idx = random.randint(0, 16)
        keep = [r for r in range(17) if r != remove_idx]
        
        # Start with 16 kept terms
        U_sup = [supports_U[r].copy() for r in keep]
        V_sup = [supports_V[r].copy() for r in keep]
        W_sup = [supports_W[r].copy() for r in keep]
        
        # Random perturbations
        n_pert = random.randint(0, 6)
        for _ in range(n_pert):
            j = random.randint(0, 15)
            mat = random.randint(0, 2)
            
            if mat == 0:
                sup = U_sup[j]
                all_idx = all_U
            elif mat == 1:
                sup = V_sup[j]
                all_idx = all_V
            else:
                sup = W_sup[j]
                all_idx = all_W
            
            if sup and random.random() < 0.4:
                to_rm = random.choice(list(sup))
                sup.discard(to_rm)
            else:
                cands = list(all_idx - sup)
                if cands:
                    sup.add(random.choice(cands))
        
        # Build design matrix
        A = np.zeros((num_entries, 16))
        for j in range(16):
            u_vec = np.zeros(n_a)
            v_vec = np.zeros(n_b)
            w_vec = np.zeros(n_c)
            for idx in U_sup[j]:
                u_vec[idx] = 1
            for idx in V_sup[j]:
                v_vec[idx] = 1
            for idx in W_sup[j]:
                w_vec[idx] = 1
            A[:, j] = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec).flatten()
        
        try:
            if np.linalg.matrix_rank(A) < 16:
                continue
            alpha, res, rk, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            if np.allclose(A @ alpha, T_flat, atol=1e-9):
                U16 = np.zeros((n_a, 16))
                V16 = np.zeros((n_b, 16))
                W16 = np.zeros((n_c, 16))
                for j in range(16):
                    u_vec = np.zeros(n_a)
                    v_vec = np.zeros(n_b)
                    w_vec = np.zeros(n_c)
                    for idx in U_sup[j]:
                        u_vec[idx] = 1
                    for idx in V_sup[j]:
                        v_vec[idx] = 1
                    for idx in W_sup[j]:
                        w_vec[idx] = 1
                    U16[:, j] = alpha[j] * u_vec
                    V16[:, j] = alpha[j] * v_vec
                    W16[:, j] = alpha[j] * w_vec
                return U16, V16, W16
        except:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    U17, V17, W17 = _get_rank17_solution()
    
    # Try rank-16 search
    result = _try_rank16_search(U17, V17, W17, T, deadline, max_trials=20000)
    if result is not None:
        U16, V16, W16 = result
        if _verify_decomposition(U16, V16, W16, T):
            return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Fallback to rank 17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
