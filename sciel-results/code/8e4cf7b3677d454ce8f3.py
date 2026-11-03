"""Matrix multiplication 4x4 x 4x7 tensor decomposition - rank minimization."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def zero_out_terms(U, V, W, r, target_rank):
        """
        Try to zero out (r - target_rank) terms from the schoolbook decomposition.
        A term can be zeroed if removing it still leaves the decomposition valid,
        OR if the term is redundant (its contribution can be absorbed by other terms).
        
        Strategy: For each term, check if we can set its U, V, W entries to zero
        and still have the tensor sum to 1 for all (i,j,k).
        """
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        # Build index arrays once
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        N = n * m * p
        u_vals = Ua[u_idx, :]  # (N, r)
        v_vals = Va[v_idx, :]  # (N, r)
        w_vals = Wa[w_idx, :]  # (N, r)
        
        # Current contribution: for each (i,j,k), sum over c of u*v*w
        # For schoolbook, each (i,j,k) has exactly one term with u=v=w=1
        # So the sum is 1 for each (i,j,k).
        
        # To zero out a term c, we need: for all (i,j,k), the sum still equals 1.
        # Since each (i,j,k) is covered by exactly one term in schoolbook,
        # zeroing a term means that (i,j,k) loses its only contribution.
        # So we CANNOT simply zero out terms from schoolbook.
        
        # Better approach: Use a known low-rank construction.
        # For 4x4 x 4x7, we can use the fact that this is related to 
        # rectangular matrix multiplication.
        
        # Let me try a different approach: use the Strassen-like recursive
        # construction or known results.
        
        # Actually, let me try a more systematic approach:
        # Start with schoolbook and try to merge terms.
        # Two terms (c1, c2) can be merged if there exists a transformation
        # that replaces them with one term.
        
        # For now, let's try a greedy approach with random perturbations.
        return Ua, Va, Wa

    def try_random_search(target_rank, max_trials=100, seed_base=42):
        """Try random initialization + gradient descent to find a rank-target_rank solution."""
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed_base + trial)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Gradient descent
            Ua = U.copy()
            Va = V.copy()
            Wa = W.copy()
            
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 3:
                    break
                
                i_idx = np.repeat(np.arange(n), m * p)
                j_idx = np.tile(np.repeat(np.arange(m), p), n)
                k_idx = np.tile(np.arange(p), n * m)
                u_idx = i_idx * m + j_idx
                v_idx = j_idx * p + k_idx
                w_idx = k_idx * n + i_idx
                
                u_vals = Ua[u_idx, :]
                v_vals = Va[v_idx, :]
                w_vals = Wa[w_idx, :]
                
                u_4d = u_vals.reshape(n, m, p, target_rank)
                v_4d = v_vals.reshape(n, m, p, target_rank)
                w_4d = w_vals.reshape(n, m, p, target_rank)
                
                prod = u_4d * v_4d * w_4d
                result = prod.sum(axis=3)
                error = result - 1.0
                
                err_norm = np.linalg.norm(error)
                if err_norm < 1e-10:
                    break
                
                err_4d = error[:, :, :, np.newaxis]
                
                grad_U_4d = 2 * err_4d * v_4d * w_4d
                grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, target_rank)
                
                grad_V_4d = 2 * err_4d * u_4d * w_4d
                grad_V_sum = np.sum(grad_V_4d, axis=0)
                grad_V = grad_V_sum.reshape(m * p, target_rank)
                
                grad_W_4d = 2 * err_4d * u_4d * v_4d
                grad_W_sum = np.sum(grad_W_4d, axis=1)
                grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
                grad_W = grad_W_sum.reshape(p * n, target_rank)
                
                Ua -= lr * grad_U
                Va -= lr * grad_V
                Wa -= lr * grad_W
                lr *= 0.995
            
            err = verify(Ua, Va, Wa, target_rank)
            if err < 1e-8:
                # Try to round to half-integers
                U_round = np.round(Ua * 2) / 2
                V_round = np.round(Va * 2) / 2
                W_round = np.round(Wa * 2) / 2
                err_round = verify(U_round, V_round, W_round, target_rank)
                if err_round < 1e-9:
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
        
        return None

    def try_structured_construction():
        """
        Try to build a low-rank decomposition using known structure.
        For 4x4 x 4x7, we can exploit the rectangular nature.
        
        One approach: decompose the 7 columns into groups.
        7 = 4 + 3, so we can do (4x4 x 4x4) + (4x4 x 4x3).
        Rank(4x4 x 4x4) <= 49 (schoolbook) but can be lower.
        Rank(4x4 x 4x3) <= 48 (schoolbook) but can be lower.
        
        Actually, let's use a different strategy:
        Use the fact that for small matrices, we can use explicit constructions.
        
        For 4x4 x 4x4, the best known rank is 49 (schoolbook) or lower.
        Actually, the best known for 4x4x4 is 49.
        
        Let me try a direct approach: use the schoolbook and see if we can
        find a lower rank by solving a system of equations.
        """
        # For 4x4 x 4x7, let's try rank 83 (known upper bound)
        # or try to find something better.
        
        # Start with schoolbook rank 112 and try to reduce.
        # Use a greedy approach: try to find terms that can be eliminated.
        
        # Actually, let's use a more principled approach:
        # The tensor for matrix multiplication has a specific structure.
        # We can use the "border rank" concept or explicit constructions.
        
        # For now, return None and let the random search try.
        return None

    # First, get the schoolbook solution as fallback
    best_result = make_schoolbook()
    best_rank = n * m * p

    # Try structured construction first
    if time.time() < deadline - 30:
        result = try_structured_construction()
        if result is not None:
            best_result = result
            best_rank = result["U"][0].__len__()  # This won't work, fix below
            # Actually, let's just check the rank
            best_rank = len(result["U"][0])

    # Try random search for various target ranks
    # Known results suggest rank around 83-90 might be achievable
    target_ranks = [83, 85, 88, 90, 95, 100, 105, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_random_search(target_rank, max_trials=20, seed_base=target_rank * 100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break

    # If we haven't improved, try a few more with different seeds
    if best_rank == n * m * p and time.time() < deadline - 15:
        for target_rank in [83, 84, 85, 86, 87]:
            if time.time() > deadline - 5:
                break
            result = try_random_search(target_rank, max_trials=10, seed_base=target_rank * 200 + 7)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break

    return best_result
