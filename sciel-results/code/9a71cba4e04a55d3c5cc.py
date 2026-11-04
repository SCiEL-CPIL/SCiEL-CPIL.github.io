import numpy as np
import time
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix.
    
    Strategy:
    1. Construct the target tensor T of shape (n*m, m*p, p*n).
    2. Use a greedy search algorithm to find a low-rank decomposition.
    3. Start with a known rank-46 solution (from block splitting) as a fallback.
    4. Attempt to find a rank-45 or lower solution using Alternating Least Squares (ALS) 
       to fit rank-1 components to the residual tensor.
    5. Use random restarts to avoid local minima.
    """
    n, m, p = 2, 4, 7
    N = n * m  # 8
    M = m * p  # 28
    P = p * n  # 14
    
    # Construct the target tensor T
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Time management
    start_time = time.time()
    timeout = 110  # Leave 10s margin for 120s limit
    best_rank = 46
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with the known rank-46 solution
    def generate_rank46_solution():
        U = np.zeros((N, 46))
        V = np.zeros((M, 46))
        W = np.zeros((P, 46))
        
        def add_term(term_idx, u_entries, v_entries, w_entries):
            for idx, val in u_entries:
                U[idx, term_idx] = val
            for idx, val in v_entries:
                V[idx, term_idx] = val
            for idx, val in w_entries:
                W[idx, term_idx] = val

        def process_strassen_block(a_base, b_base, c_base, term_start):
            A11_idx = a_base + 0
            A12_idx = a_base + 1
            A21_idx = a_base + 4
            A22_idx = a_base + 5
            
            B11_idx = b_base + 0
            B12_idx = b_base + 1
            B21_idx = b_base + 7
            B22_idx = b_base + 8
            
            C11_idx = c_base + 0
            C12_idx = c_base + 2
            C21_idx = c_base + 1
            C22_idx = c_base + 3
            
            # M1 = (A11 + A22)(B11 + B22)
            add_term(term_start + 0, 
                     [(A11_idx, 1), (A22_idx, 1)], 
                     [(B11_idx, 1), (B22_idx, 1)], 
                     [(C11_idx, 1), (C22_idx, 1)])
                     
            # M2 = (A21 + A22) B11
            add_term(term_start + 1, 
                     [(A21_idx, 1), (A22_idx, 1)], 
                     [(B11_idx, 1)], 
                     [(C21_idx, 1), (C22_idx, -1)])
                     
            # M3 = A11 (B12 - B22)
            add_term(term_start + 2, 
                     [(A11_idx, 1)], 
                     [(B12_idx, 1), (B22_idx, -1)], 
                     [(C12_idx, 1), (C22_idx, 1)])
                     
            # M4 = A22 (B21 - B11)
            add_term(term_start + 3, 
                     [(A22_idx, 1)], 
                     [(B21_idx, 1), (B11_idx, -1)], 
                     [(C11_idx, 1), (C21_idx, 1)])
                     
            # M5 = (A11 + A12) B22
            add_term(term_start + 4, 
                     [(A11_idx, 1), (A12_idx, 1)], 
                     [(B22_idx, 1)], 
                     [(C11_idx, -1), (C12_idx, 1)])
                     
            # M6 = (A21 - A11) (B11 + B12)
            add_term(term_start + 5, 
                     [(A21_idx, 1), (A11_idx, -1)], 
                     [(B11_idx, 1), (B12_idx, 1)], 
                     [(C22_idx, 1)])
                     
            # M7 = (A12 - A22) (B21 + B22)
            add_term(term_start + 6, 
                     [(A12_idx, 1), (A22_idx, -1)], 
                     [(B21_idx, 1), (B22_idx, 1)], 
                     [(C11_idx, 1), (C12_idx, 1)])
                     
            return term_start + 7

        def process_rank2_block(a_base, b_base, c_base, term_start):
            A0_idx = a_base + 0
            A1_idx = a_base + 4
            
            B0_idx = b_base + 0
            B1_idx = b_base + 7
            
            C0_idx = c_base + 0
            C1_idx = c_base + 1
            
            add_term(term_start + 0, 
                     [(A0_idx, 1)], 
                     [(B0_idx, 1), (B1_idx, 1)], 
                     [(C0_idx, 1)])
                     
            add_term(term_start + 1, 
                     [(A1_idx, 1)], 
                     [(B0_idx, 1), (B1_idx, 1)], 
                     [(C1_idx, 1)])
                     
            return term_start + 2

        current_term = 0
        # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
        current_term = process_strassen_block(0, 0, 0, current_term)
        current_term = process_strassen_block(0, 2, 4, current_term)
        current_term = process_strassen_block(0, 4, 8, current_term)
        current_term = process_rank2_block(0, 6, 12, current_term)
        
        # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
        current_term = process_strassen_block(2, 14, 0, current_term)
        current_term = process_strassen_block(2, 16, 4, current_term)
        current_term = process_strassen_block(2, 18, 8, current_term)
        current_term = process_rank2_block(2, 20, 12, current_term)
        
        assert current_term == 46
        return U, V, W

    U46, V46, W46 = generate_rank46_solution()
    best_U, best_V, best_W = U46, V46, W46
    
    # Greedy Search Function
    def find_rank1_approx(R, max_iters=500):
        """
        Find a rank-1 approximation of tensor R using ALS.
        Returns u, v, w, error.
        """
        u = np.random.randn(N)
        v = np.random.randn(M)
        w = np.random.randn(P)
        
        # Normalize
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        w /= np.linalg.norm(w)
        
        prev_error = np.inf
        for _ in range(max_iters):
            # Solve for u
            # R = u (v x w)^T + ...
            # u = (R * (v x w)) / ||v x w||^2
            vw = np.outer(v, w) # Shape (M, P)
            # Contract R with vw over axes 1 and 2
            # R is (N, M, P). vw is (M, P).
            # u_i = sum_{j,k} R[i,j,k] * v[j] * w[k]
            u_new = np.einsum('ijk,jk->i', R, vw)
            norm_vw_sq = np.sum(vw**2)
            if norm_vw_sq > 1e-12:
                u_new /= norm_vw_sq
            else:
                u_new = u
            # Normalize u
            norm_u = np.linalg.norm(u_new)
            if norm_u > 1e-12:
                u = u_new / norm_u
            else:
                u = np.random.randn(N)
                u /= np.linalg.norm(u)
            
            # Solve for v
            # v_j = sum_{i,k} R[i,j,k] * u[i] * w[k]
            uw = np.outer(u, w) # Shape (N, P)
            v_new = np.einsum('ijk,ik->j', R, uw)
            norm_uw_sq = np.sum(uw**2)
            if norm_uw_sq > 1e-12:
                v_new /= norm_uw_sq
            else:
                v_new = v
            norm_v = np.linalg.norm(v_new)
            if norm_v > 1e-12:
                v = v_new / norm_v
            else:
                v = np.random.randn(M)
                v /= np.linalg.norm(v)
                
            # Solve for w
            # w_k = sum_{i,j} R[i,j,k] * u[i] * v[j]
            uv = np.outer(u, v) # Shape (N, M)
            w_new = np.einsum('ijk,ij->k', R, uv)
            norm_uv_sq = np.sum(uv**2)
            if norm_uv_sq > 1e-12:
                w_new /= norm_uv_sq
            else:
                w_new = w
            norm_w = np.linalg.norm(w_new)
            if norm_w > 1e-12:
                w = w_new / norm_w
            else:
                w = np.random.randn(P)
                w /= np.linalg.norm(w)
                
            # Calculate error
            # R_approx = u (v x w)^T
            R_approx = np.einsum('i,j,k->ijk', u, v, w)
            error = np.linalg.norm(R - R_approx)
            
            if abs(prev_error - error) < 1e-10:
                break
            prev_error = error
            
        # Final error and scaling
        R_approx = np.einsum('i,j,k->ijk', u, v, w)
        # Optimal scaling factor for u, v, w to best match R
        # We want to minimize || R - s * u (v x w)^T ||
        # s = <R, u (v x w)^T> / ||u (v x w)^T||^2
        num = np.sum(R * R_approx)
        den = np.sum(R_approx**2)
        if den > 1e-12:
            s = num / den
        else:
            s = 1.0
            
        u *= s
        error = np.linalg.norm(R - np.einsum('i,j,k->ijk', u, v, w))
        
        return u, v, w, error

    def greedy_decompose(target, max_rank, time_limit):
        """
        Try to decompose target tensor into sum of max_rank rank-1 tensors.
        Returns (U, V, W, success)
        """
        R = target.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(max_rank):
            if time.time() > time_limit:
                return None, None, None, False
                
            # Try a few random restarts for the rank-1 step
            best_u, best_v, best_w, best_err = None, None, None, np.inf
            
            # Number of restarts depends on remaining time
            restarts = 10
            for _ in range(restarts):
                u, v, w, err = find_rank1_approx(R, max_iters=200)
                if err < best_err:
                    best_err = err
                    best_u, best_v, best_w = u, v, w
                    
            if best_u is None:
                return None, None, None, False
                
            R = R - np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            if np.linalg.norm(R) < 1e-9:
                break
                
        if np.linalg.norm(R) < 1e-9:
            U = np.array(U_list).T
            V = np.array(V_list).T
            W = np.array(W_list).T
            return U, V, W, True
        else:
            return None, None, None, False

    # Search for lower rank
    # Try rank 45, 44, ...
    for rank in range(45, 20, -1):
        if time.time() > timeout:
            break
            
        # Try a few random initializations of the whole greedy process?
        # Greedy is deterministic given the restarts inside.
        # We can vary the random seed for the restarts.
        success = False
        for attempt in range(5):
            np.random.seed(attempt)
            U, V, W, success = greedy_decompose(T, rank, time_limit=timeout)
            if success:
                # Verify
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T, T_approx, atol=1e-9):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break # Found a valid decomposition for this rank, try lower
            if time.time() > timeout:
                break
                
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
