"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Integer Linear Programming to find minimum rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    required_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                required_triples.append((a, b, c))
    
    n_required = len(required_triples)  # 36

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c_idx in range(rank):
            uc = U[:, c_idx]
            vc = V[:, c_idx]
            wc = W[:, c_idx]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook as baseline
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 36

    # Try to find better decomposition using ILP
    # 
    # Key insight: A rank-1 tensor is determined by:
    # - Which a-indices it has non-zero U values (subset of {0,...,5})
    # - Which b-indices it has non-zero V values (subset of {0,...,17})
    # - Which c-indices it has non-zero W values (subset of {0,...,11})
    # 
    # For the decomposition to be exact, each required triple (a,b,c) must be "covered"
    # by exactly one rank-1 tensor that has non-zero entries at all three positions.
    #
    # However, we also need to ensure NO false positives: no (a,b,c) with T[a,b,c]=0
    # should have non-zero contribution.
    #
    # This is complex. Let's use a different approach:
    # Enumerate all "valid" rank-1 patterns and use set cover.
    #
    # A rank-1 pattern is (S_a, S_b, S_c) where:
    # - S_a ⊆ {0,...,5}, S_b ⊆ {0,...,17}, S_c ⊆ {0,...,11}
    # - For all (a,b,c) in S_a × S_b × S_c, T[a,b,c] must be 1
    #   (otherwise we'd create false positives)
    #
    # Then we need to cover all required triples with minimum number of such patterns.
    #
    # But the number of possible patterns is huge. Let's restrict to "structured" patterns.
    
    # Alternative approach: Use the structure of matrix multiplication.
    # Each rank-1 tensor corresponds to a "combinatorial rectangle" in the multiplication.
    
    # Let's try a greedy + local search approach instead, which is more tractable.
    
    def greedy_decomposition():
        """
        Greedy approach: iteratively pick the best rank-1 tensor to add.
        A rank-1 tensor is defined by which (a,b,c) triples it covers.
        """
        # Start with empty decomposition
        covered = set()
        factors = []  # List of (U_col, V_col, W_col)
        
        # Precompute all valid rank-1 patterns
        # A pattern is valid if for all (a,b,c) in the pattern, T[a,b,c] = 1
        # and the pattern covers at least one uncovered required triple.
        
        # To keep this tractable, let's consider patterns based on the structure:
        # For each choice of which "block" of the multiplication we're handling
        
        # Actually, let's use a simpler heuristic:
        # Try all possible single-triple patterns first (schoolbook), then try to merge.
        
        # Better: use the fact that in matrix multiplication, the natural structure
        # is that A[i,j] * B[j,k] contributes to C[i,k].
        # So a rank-1 tensor naturally covers triples where i is fixed or j is fixed or k is fixed.
        
        # Let's try patterns of the form:
        # Type 1: Fix i, vary j and k -> covers {(i*m+j, j*p+k, k*n+i) for all j,k}
        # Type 2: Fix j, vary i and k -> covers {(i*m+j, j*p+k, k*n+i) for all i,k}  
        # Type 3: Fix k, vary i and j -> covers {(i*m+j, j*p+k, k*n+i) for all i,j}
        # Type 4: Fix i and j, vary k -> covers {(i*m+j, j*p+k, k*n+i) for all k}
        # Type 5: Fix i and k, vary j -> covers {(i*m+j, j*p+k, k*n+i) for all j}
        # Type 6: Fix j and k, vary i -> covers {(i*m+j, j*p+k, k*n+i) for all i}
        
        all_patterns = []
        
        # Type 1: Fix i
        for i in range(n):
            triples = []
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
            all_patterns.append(('fix_i', i, triples))
        
        # Type 2: Fix j
        for j in range(m):
            triples = []
            for i in range(n):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
            all_patterns.append(('fix_j', j, triples))
        
        # Type 3: Fix k
        for k in range(p):
            triples = []
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
            all_patterns.append(('fix_k', k, triples))
        
        # Type 4: Fix i and j
        for i in range(n):
            for j in range(m):
                triples = []
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
                all_patterns.append(('fix_ij', (i, j), triples))
        
        # Type 5: Fix i and k
        for i in range(n):
            for k in range(p):
                triples = []
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
                all_patterns.append(('fix_ik', (i, k), triples))
        
        # Type 6: Fix j and k
        for j in range(m):
            for k in range(p):
                triples = []
                for i in range(n):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
                all_patterns.append(('fix_jk', (j, k), triples))
        
        # Now greedily pick patterns
        remaining = set(required_triples)
        
        while remaining:
            if time.time() > deadline - 10:
                break
            
            best_pattern = None
            best_new_cover = 0
            
            for ptype, pkey, triples in all_patterns:
                new_cover = len(set(triples) & remaining)
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_pattern = (ptype, pkey, triples)
            
            if best_pattern is None or best_new_cover == 0:
                break
            
            ptype, pkey, triples = best_pattern
            remaining -= set(triples)
            factors.append(best_pattern)
        
        # Build the decomposition from the selected patterns
        rank = len(factors)
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        for col, (ptype, pkey, triples) in enumerate(factors):
            for (a, b, c) in triples:
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
        
        return U, V, W, rank

    # Try greedy
    if time.time() < deadline - 20:
        U_g, V_g, W_g, rank_g = greedy_decomposition()
        if verify_decomposition(U_g, V_g, W_g):
            if rank_g < best_rank:
                best_U, best_V, best_W = U_g, V_g, W_g
                best_rank = rank_g

    # Try local search to improve
    def local_search(U, V, W, max_iters=1000):
        """
        Try to improve the decomposition by replacing subsets of rank-1 tensors
        with smaller sets.
        """
        rank = U.shape[1]
        current_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try removing one tensor and see if we can replace it
            # with fewer tensors
            improved = False
            
            for col_to_remove in range(current_rank):
                if time.time() > deadline - 5:
                    break
                
                # What triples does this tensor cover?
                a_indices = np.nonzero(U[:, col_to_remove])[0]
                b_indices = np.nonzero(V[:, col_to_remove])[0]
                c_indices = np.nonzero(W[:, col_to_remove])[0]
                
                covered_triples = set()
                for a in a_indices:
                    for b in b_indices:
                        for c in c_indices:
                            if T[a, b, c] == 1:
                                covered_triples.add((a, b, c))
                
                if not covered_triples:
                    # This tensor doesn't cover any required triples, remove it
                    best_U = np.delete(best_U, col_to_remove, axis=1)
                    best_V = np.delete(best_V, col_to_remove, axis=1)
                    best_W = np.delete(best_W, col_to_remove, axis=1)
                    current_rank -= 1
                    improved = True
                    break
                
                # Check if these triples are covered by other tensors
                other_U = np.delete(U, col_to_remove, axis=1)
                other_V = np.delete(V, col_to_remove, axis=1)
                other_W = np.delete(W, col_to_remove, axis=1)
                
                all_covered = True
                for (a, b, c) in covered_triples:
                    # Check if any other tensor covers this triple
                    covered_by_other = False
                    for col in range(other_U.shape[1]):
                        if other_U[a, col] != 0 and other_V[b, col] != 0 and other_W[c, col] != 0:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        all_covered = False
                        break
                
                if all_covered:
                    # We can remove this tensor
                    best_U = np.delete(best_U, col_to_remove, axis=1)
                    best_V = np.delete(best_V, col_to_remove, axis=1)
                    best_W = np.delete(best_W, col_to_remove, axis=1)
                    current_rank -= 1
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_U, best_V, best_W, current_rank

    # Apply local search to current best
    if time.time() < deadline - 10:
        U_ls, V_ls, W_ls, rank_ls = local_search(best_U, best_V, best_W)
        if verify_decomposition(U_ls, V_ls, W_ls):
            if rank_ls < best_rank:
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                best_rank = rank_ls

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
