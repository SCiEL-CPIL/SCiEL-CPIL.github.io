import numpy as np
import time
import random
import itertools

# Dimensions
n, m, p = 2, 3, 6

# Precompute the target tensor in flattened form
# U index: a*m + b (a in 0..n-1, b in 0..m-1) -> size n*m = 6
# V index: c*p + d (c in 0..m-1, d in 0..p-1) -> size m*p = 18
# W index: k*n + i (k in 0..p-1, i in 0..n-1) -> size p*n = 12
#
# The tensor T is such that T[u, v, w] = 1 if and only if:
# u = a*m + b
# v = b*p + d  (Note: the middle index 'b' must match)
# w = d*n + a
# Otherwise 0.

def get_tensor_entries():
    """
    Returns a list of (u, v, w) tuples for all non-zero entries of the multiplication tensor.
    There are exactly n*m*p = 36 such entries.
    """
    entries = []
    for a in range(n):
        for b in range(m):
            for d in range(p):
                u = a * m + b
                v = b * p + d
                w = d * n + a
                entries.append((u, v, w))
    return entries

T_ENTRIES = get_tensor_entries()

def verify_decomposition(U, V, W, tol=1e-6):
    """
    Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    
    # Check all 36 target entries are 1
    for u, v, w in T_ENTRIES:
        s = np.dot(U[u], V[v] * W[w])
        if abs(s - 1.0) > tol:
            return False
            
    # Check all other entries are 0
    # To be efficient, we can sample or check all. 
    # Total combinations: 6 * 18 * 12 = 1296.
    # We need to ensure that for any (u,v,w) not in T_ENTRIES, the sum is 0.
    target_set = set(T_ENTRIES)
    
    # Optimization: Instead of checking all 1296, we can check the linear system.
    # But for correctness, checking all is safer and fast enough (1296 * rank ops).
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) in target_set:
                    continue
                s = np.dot(U[u], V[v] * W[w])
                if abs(s) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Start with a known valid high-rank decomposition (e.g., schoolbook rank 36).
    # 2. Use a greedy/sparsification approach to reduce rank.
    #    - Try to merge terms.
    #    - Or, more effectively: use a randomized search for a decomposition of rank R.
    
    # We know schoolbook is 36.
    # Block sum 2x3x3 (rank 11) * 2 = 22 is a strong baseline if we can construct it.
    # But let's try to find something even better or simply a very efficient 22 or 21.
    
    # Let's first try to construct the known rank-22 decomposition (2 * rank-11 2x3x3).
    # If that fails verification or is too complex to code correctly from scratch,
    # we fall back to a search.
    
    # Actually, a simpler robust strategy:
    # Use a local search to minimize rank.
    # Start with rank R = 22.
    # Initialize U, V, W with random sparse matrices.
    # Or better: Start with the schoolbook decomposition (rank 36) and try to compress.
    
    # Let's try a direct search for rank R.
    # The search space is huge. 
    # However, for small ranks, we can use the fact that the tensor is sparse.
    
    # Let's try to construct the rank 22 solution explicitly using the block method,
    # but implemented generically to avoid hardcoding errors.
    
    # Block 1: A(2x3) * B1(3x3) -> C1(2x3)
    # Block 2: A(2x3) * B2(3x3) -> C2(2x3)
    # Total rank = rank(2x3x3) + rank(2x3x3).
    
    # What is the minimal rank for 2x3x3?
    # Literature says 11.
    # Let's try to find a rank-11 decomposition for 2x3x3 using a local search.
    # If we find it, we can duplicate it for the 2x3x6 case (rank 22).
    
    def find_decomposition_233(target_rank, max_time=10.0):
        """
        Find a decomposition for 2x3x3 matrix multiplication.
        Returns U, V, W matrices of shape (6, r), (9, r), (6, r) or None.
        """
        start = time.time()
        
        # Target entries for 2x3x3
        # n=2, m=3, p=3
        entries = []
        for a in range(2):
            for b in range(3):
                for d in range(3):
                    u = a * 3 + b
                    v = b * 3 + d
                    w = d * 2 + a
                    entries.append((u, v, w))
        
        target_set = set(entries)
        
        # Initialize with random sparse matrices
        # We want integer/half-integer entries. 
        # Let's try values from {-1, -0.5, 0, 0.5, 1}
        
        best_U = None
        best_V = None
        best_W = None
        best_error = float('inf')
        
        # Use a simple hill climbing / simulated annealing
        # State: U, V, W
        # Cost: Sum of squared errors over all (u,v,w)
        
        def compute_error(U, V, W):
            err = 0.0
            # Check target entries
            for u, v, w in entries:
                val = np.dot(U[u], V[v] * W[w])
                err += (val - 1.0)**2
            # Check non-target entries (sample some if too slow, but 6*9*6=324 is small)
            for u in range(6):
                for v in range(9):
                    for w in range(6):
                        if (u, v, w) in target_set:
                            continue
                        val = np.dot(U[u], V[v] * W[w])
                        err += val**2
            return err
        
        # Initialize
        U = np.zeros((6, target_rank))
        V = np.zeros((9, target_rank))
        W = np.zeros((6, target_rank))
        
        # Random init with sparsity
        for k in range(target_rank):
            # Each term should have a few non-zeros
            # U has 6 elements, V has 9, W has 6
            # Randomly pick 2-3 non-zeros for each
            u_idx = np.random.choice(6, size=2, replace=False)
            v_idx = np.random.choice(9, size=2, replace=False)
            w_idx = np.random.choice(6, size=2, replace=False)
            
            U[u_idx, k] = np.random.choice([-1, 1, 0.5, -0.5], size=2)
            V[v_idx, k] = np.random.choice([-1, 1, 0.5, -0.5], size=2)
            W[w_idx, k] = np.random.choice([-1, 1, 0.5, -0.5], size=2)
            
        current_error = compute_error(U, V, W)
        
        # Local search
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        while time.time() - start < max_time and T > min_T:
            # Pick a random element to perturb
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                i, j = np.random.randint(0, 6), np.random.randint(0, target_rank)
                old_val = U[i, j]
                new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                if new_val == old_val:
                    continue
                U[i, j] = new_val
                new_error = compute_error(U, V, W)
            elif mat == 'V':
                i, j = np.random.randint(0, 9), np.random.randint(0, target_rank)
                old_val = V[i, j]
                new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                if new_val == old_val:
                    continue
                V[i, j] = new_val
                new_error = compute_error(U, V, W)
            else:
                i, j = np.random.randint(0, 6), np.random.randint(0, target_rank)
                old_val = W[i, j]
                new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                if new_val == old_val:
                    continue
                W[i, j] = new_val
                new_error = compute_error(U, V, W)
            
            delta = new_error - current_error
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_error < 1e-8:
                        break
            else:
                # Revert
                if mat == 'U': U[i, j] = old_val
                elif mat == 'V': V[i, j] = old_val
                else: W[i, j] = old_val
                
            T *= cooling
        
        if best_error < 1e-6:
            return best_U, best_V, best_W
        return None

    # Try to find rank 11 for 2x3x3
    # If found, duplicate for 2x3x6 (rank 22)
    
    # We have ~120s. Let's spend 20s trying to find rank 11 for 2x3x3.
    # If successful, we return rank 22.
    # If not, we fall back to a known valid rank (e.g. 33 or 34 or 36).
    
    # Actually, let's try to find rank 10 or 11.
    # The lower bound for 2x3x3 is 11? Or 10?
    # Let's try rank 11 first.
    
    U233 = None
    V233 = None
    W233 = None
    
    # Try rank 11
    if time.time() < deadline - 30:
        res = find_decomposition_233(11, max_time=20.0)
        if res:
            U233, V233, W233 = res
    
    # If we found a 2x3x3 rank 11 decomposition, we can build 2x3x6 rank 22.
    if U233 is not None:
        U = np.zeros((6, 22))
        V = np.zeros((18, 22))
        W = np.zeros((12, 22))
        
        # Block 1: B columns 0-2
        # V indices for block 1: b*6 + d where d in 0..2
        # For 2x3x3, V index is b*3 + d.
        # Mapping: V233[b*3+d, k] -> V[b*6+d, k]
        for b in range(3):
            for d in range(3):
                V[b*6+d, :11] = V233[b*3+d, :11]
        
        U[:, :11] = U233
        # W indices for block 1: k*n + i where k in 0..2
        # For 2x3x3, W index is d*2 + i (using d as p-index).
        # Mapping: W233[d*2+i, k] -> W[k*2+i, k] where k is 0..2
        for d in range(3):
            for i in range(2):
                W[d*2+i, :11] = W233[d*2+i, :11]
                
        # Block 2: B columns 3-5
        # V indices for block 2: b*6 + (3+d) where d in 0..2
        for b in range(3):
            for d in range(3):
                V[b*6+3+d, 11:22] = V233[b*3+d, :11]
                
        U[:, 11:22] = U233
        # W indices for block 2: (3+d)*n + i
        for d in range(3):
            for i in range(2):
                W[(3+d)*2+i, 11:22] = W233[d*2+i, :11]
                
        if verify_decomposition(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback: Construct a known valid rank 33 or 34.
    # Let's try to construct rank 33 using a mix.
    # Or just return schoolbook rank 36 if we don't have time.
    # But we can easily construct rank 34 (2 * 17).
    # And rank 33 is possible if we find one 2x3x3 rank 16 and one rank 17?
    
    # Let's try to find rank 16 for 2x3x3 quickly.
    # If we find rank 16, we can use rank 16 + rank 16 = 32? Or 16+17=33?
    # If we find rank 16, total rank 32 is possible.
    
    U233_16 = None
    V233_16 = None
    W233_16 = None
    
    if time.time() < deadline - 20:
        res = find_decomposition_233(16, max_time=10.0)
        if res:
            U233_16, V233_16, W233_16 = res
            if verify_decomposition_233_local(U233_16, V233_16, W233_16):
                # Build rank 32
                U = np.zeros((6, 32))
                V = np.zeros((18, 32))
                W = np.zeros((12, 32))
                
                for b in range(3):
                    for d in range(3):
                        V[b*6+d, :16] = V233_16[b*3+d, :16]
                        V[b*6+3+d, 16:32] = V233_16[b*3+d, :16]
                        
                U[:, :16] = U233_16
                U[:, 16:32] = U233_16
                
                for d in range(3):
                    for i in range(2):
                        W[d*2+i, :16] = W233_16[d*2+i, :16]
                        W[(3+d)*2+i, 16:32] = W233_16[d*2+i, :16]
                        
                if verify_decomposition(U, V, W):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }

    # Final Fallback: Schoolbook Rank 36
    # This is always valid.
    U = np.zeros((6, 36))
    V = np.zeros((18, 36))
    W = np.zeros((12, 36))
    
    idx = 0
    for a in range(2):
        for b in range(3):
            for d in range(6):
                u = a * 3 + b
                v = b * 6 + d
                w = d * 2 + a
                U[u, idx] = 1
                V[v, idx] = 1
                W[w, idx] = 1
                idx += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def verify_decomposition_233_local(U, V, W, tol=1e-6):
    """Helper to verify 2x3x3 decomposition locally."""
    entries = []
    for a in range(2):
        for b in range(3):
            for d in range(3):
                u = a * 3 + b
                v = b * 3 + d
                w = d * 2 + a
                entries.append((u, v, w))
    target_set = set(entries)
    
    for u, v, w in entries:
        s = np.dot(U[u], V[v] * W[w])
        if abs(s - 1.0) > tol:
            return False
            
    for u in range(6):
        for v in range(9):
            for w in range(6):
                if (u, v, w) in target_set:
                    continue
                s = np.dot(U[u], V[v] * W[w])
                if abs(s) > tol:
                    return False
    return True

if __name__ == "__main__":
    construct()
