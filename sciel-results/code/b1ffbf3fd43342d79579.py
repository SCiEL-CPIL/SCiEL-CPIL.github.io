import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W):
    T = _build_target_tensor()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_decomp, T, atol=1e-9)

def _to_half_int(x):
    return np.round(x * 2) / 2

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _try_rank_combinatorial(rank, time_limit=5.0):
    """
    Attempts to find a rank-`rank` decomposition by starting from the schoolbook
    rank-50 decomposition and greedily merging terms.
    
    The idea is that if two terms share the same U-row, V-row, or W-row,
    they might be combinable. Specifically, if we have terms t1 and t2,
    we can try to replace them with a single term t3 if there exist coefficients
    such that the tensor contribution is preserved.
    
    However, a simpler structural merge is if two terms have identical U, V, or W.
    In the schoolbook, no two terms are identical.
    
    A more powerful merge: 
    If we have terms (u1, v1, w1) and (u2, v2, w2).
    If u1 == u2, we can combine v1*w1 + v2*w2. If this product is a rank-1 tensor,
    we can merge. v1*w1 + v2*w2 is rank-1 if v1 and v2 are linearly dependent.
    
    Let's look for terms with linearly dependent V columns (or U or W).
    In schoolbook, V columns are basis vectors e_j*p+k. 
    v_{j1, k1} and v_{j2, k2} are linearly dependent iff (j1,k1)==(j2,k2).
    So no merges in schoolbook.
    
    This suggests simple merging of schoolbook terms won't work.
    We need a different initialization or a different merge rule.
    
    Alternative: Randomized search for half-integer solutions.
    Since the space is discrete (half-integers), we can use a local search
    on the discrete lattice.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Initialize with random half-integers
    # This is a heuristic: maybe the solution is "sparse"
    # Or we can start with schoolbook and randomly perturb/merge.
    
    # Let's try a discrete local search on half-integers.
    # Objective: minimize || U V^T W^T - T ||^2
    # But we must stay on the half-integer lattice.
    
    # Start with schoolbook
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We can't just use schoolbook if rank < 50.
    # We must construct a rank `rank` candidate.
    
    # Strategy: Start with random half-integers, then do coordinate descent.
    # Since rank is small, the space is huge.
    
    # Better Strategy: Use the known lower bounds.
    # For 2x5x5, the rank is likely 50 (schoolbook).
    # If we can't find a better one, return schoolbook.
    
    # Let's try to find a rank 49 solution using a specific structure.
    # Some matrices have rank < nmp.
    # For example, if one dimension is 1, rank = max(n,p).
    # Here n=2, m=5, p=5.
    
    # Let's try a few random restarts with discrete local search.
    # This is a "new direction" compared to continuous GD.
    
    def discrete_local_search(U, V, W, max_iters=2000):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Snap to half-integers
        U = _to_half_int(U)
        V = _to_half_int(V)
        W = _to_half_int(W)
        
        for _ in range(max_iters):
            if time.time() > start + time_limit:
                break
            
            # Calculate error
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_decomp - T
            loss = np.sum(err**2)
            
            if loss < 1e-9:
                return U, V, W, loss
            
            # Try to update one element at a time (coordinate descent on lattice)
            # To be fast, we pick a random element from U, V, or W
            # and try changing it by +/- 0.5
            
            # We can be smarter: update the element with the largest gradient
            # but constrained to the lattice.
            
            # Gradient for U[a, r] is sum_{b,c} err[a,b,c] V[b,r] W[c,r]
            # Let's compute gradients for all U, V, W
            grad_U = np.einsum('abc,br,cr->ar', err, V, W)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V)
            
            # Find the index with the largest gradient magnitude
            # We can't just step in the gradient direction because of the lattice.
            # We can check the two neighbors: current +/- 0.5.
            
            # To speed up, let's just pick a random few candidates.
            # Or pick the top K gradients.
            
            # Let's pick the single best move among a random sample of indices.
            # This is a stochastic local search.
            
            # Sample indices
            num_samples = 50
            best_loss = loss
            best_move = None
            
            for _ in range(num_samples):
                # Randomly choose which matrix to update
                mat_idx = np.random.choice(3)
                
                if mat_idx == 0:
                    a, r = np.random.choice(n*m), np.random.choice(rank)
                    # Try +/- 0.5
                    for delta in [-0.5, 0.5]:
                        U_new = U.copy()
                        U_new[a, r] += delta
                        # Check if it's a half-integer (it should be)
                        # Calculate new loss approximately? No, just full calc is fine for small dims.
                        # But 50 samples * full calc might be slow.
                        # Let's optimize the loss calc.
                        # Loss change is quadratic.
                        # For now, just do it.
                        
                        # Actually, let's just do one full update per iteration in the main loop
                        # but choose the move carefully.
                        pass # Skip inner loop for speed in this sketch, will implement below
                
                # For simplicity in this version, let's just do a random perturbation
                # and accept if it improves.
                pass
            
            # Revised: Single best move selection
            # We will evaluate a few random moves and pick the best.
            moves = []
            for _ in range(20):
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    a, r = np.random.choice(n*m), np.random.choice(rank)
                    for delta in [-0.5, 0.5]:
                        moves.append((0, a, r, delta))
                elif mat_idx == 1:
                    b, r = np.random.choice(m*p), np.random.choice(rank)
                    for delta in [-0.5, 0.5]:
                        moves.append((1, b, r, delta))
                else:
                    c, r = np.random.choice(p*n), np.random.choice(rank)
                    for delta in [-0.5, 0.5]:
                        moves.append((2, c, r, delta))
            
            best_move_loss = loss
            best_move_tuple = None
            
            for m_type, i, j, delta in moves:
                if m_type == 0:
                    U[i, j] += delta
                    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                    l = np.sum((T_decomp - T)**2)
                    U[i, j] -= delta
                elif m_type == 1:
                    V[i, j] += delta
                    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                    l = np.sum((T_decomp - T)**2)
                    V[i, j] -= delta
                else:
                    W[i, j] += delta
                    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                    l = np.sum((T_decomp - T)**2)
                    W[i, j] -= delta
                
                if l < best_move_loss:
                    best_move_loss = l
                    best_move_tuple = (m_type, i, j, delta)
            
            if best_move_tuple:
                m_type, i, j, delta = best_move_tuple
                if m_type == 0:
                    U[i, j] += delta
                elif m_type == 1:
                    V[i, j] += delta
                else:
                    W[i, j] += delta
            else:
                # Stuck? Try a larger perturbation or break
                # If stuck for a while, break
                break
                
        return U, V, W, loss

    # Run a few trials
    for trial in range(5):
        if time.time() > start + time_limit:
            break
        
        np.random.seed(42 + trial + rank)
        # Initialize with small random half-integers
        U = np.random.randint(0, 2, (n*m, rank)) * 0.5
        V = np.random.randint(0, 2, (m*p, rank)) * 0.5
        W = np.random.randint(0, 2, (p*n, rank)) * 0.5
        
        U, V, W, loss = discrete_local_search(U, V, W, max_iters=1000)
        
        if loss < 1e-9:
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            
    return None

def construct():
    start = time.time()
    deadline = start + 110
    best = _schoolbook()
    
    # The rank of 2x5x5 matrix multiplication is 50.
    # It is highly unlikely that a rank < 50 exists with half-integer entries.
    # The combinatorial search above is a "long shot" but structurally different.
    # If it fails (which is likely), we return schoolbook.
    
    # We will try ranks 49 and 48.
    for rank in [49, 48]:
        if time.time() > deadline - 10:
            break
        result = _try_rank_combinatorial(rank, time_limit=10.0)
        if result is not None:
            best = result
            break
            
    return best

# EVOLVE-BLOCK-END
