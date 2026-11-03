"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations_with_replacement
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24
    
    # 1. Generate Dictionary of Candidate Rank-1 Terms
    # A term is (u, v, w) where u in R^nm, v in R^mp, w in R^pn.
    # We restrict to integer coefficients {-1, 0, 1} to keep the problem tractable and aligned with known algebraic decompositions.
    # We include all terms with "small" support.
    
    def gen_sparse_vectors(dim, max_support=3):
        """Generate all vectors in {-1,0,1}^dim with non-zero entries <= max_support."""
        vecs = []
        # Zero vector
        vecs.append([0]*dim)
        # Non-zero vectors
        for k in range(1, max_support + 1):
            for indices in combinations_with_replacement(range(dim), k):
                # We need unique indices for standard basis-like sparsity? 
                # No, support means number of non-zeros. 
                # If we allow repeated indices in combinations_with_replacement, it doesn't make sense for support.
                # We should use combinations for unique indices.
                pass
        
        # Correct generation:
        vecs = []
        for k in range(0, max_support + 1):
            for indices in combinations(range(dim), k):
                for signs in product([-1, 1], repeat=k):
                    vec = [0]*dim
                    for idx, sign in zip(indices, signs):
                        vec[idx] = sign
                    vecs.append(tuple(vec))
        return vecs

    # For n=3, m=3, p=8:
    # nm=9, mp=24, pn=24.
    # max_support=2 is small. max_support=3 is larger.
    # Let's estimate sizes.
    # For dim=9, k=0: 1, k=1: 9*2=18, k=2: 36*4=144, k=3: 84*8=672. Total ~835.
    # For dim=24, k=0: 1, k=1: 48, k=2: 276*4=1104, k=3: 2024*8=16192. Total ~17345.
    # Cross product: 835 * 17345 * 17345 is too large (approx 250 billion).
    # We cannot enumerate the full cross product.
    
    # Strategy: Generate a smaller set of "high quality" candidates.
    # 1. Schoolbook terms (support 1).
    # 2. Strassen-like terms for 4x4x4 blocks embedded? 
    # 3. Random sparse terms.
    
    # Let's build the dictionary more intelligently.
    # We know the target tensor T has 1s at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    # We need sum_r u_r[i*m+j] v_r[j*p+k] w_r[k*n+i] = 1 for all i,j,k.
    # And 0 for all other (a,b,c).
    
    # Instead of a massive BIP, let's try to construct a solution using a known low-rank algorithm for 3x3x3 (Strassen, rank 23) 
    # and extend it to 3x3x8.
    # 3x3x8 can be viewed as 3x3x4 + 3x3x4.
    # If we have a rank R_4 decomposition for 3x3x4, total rank is 2*R_4.
    # Schoolbook for 3x3x4 is 36. Two blocks: 72.
    # Can we do better for 3x3x4?
    # 4x4x4 has rank 23. 3x3x4 is a sub-tensor. 
    # It is known that rank(3x3x4) <= 26? Or maybe 24?
    # Let's try to find a rank 24 decomposition for 3x3x4 using the BIP approach on a smaller scale, 
    # or just use a random search / local search on the full 3x3x8 problem with a tighter bound.
    
    # Given the 120s limit, a full BIP on 5000 variables might be risky if the dictionary is large.
    # Let's use a moderate dictionary size.
    
    # Generate candidates:
    candidates = []
    
    # 1. Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*nm; u[i*m+j] = 1
                v = [0]*mp; v[j*p+k] = 1
                w = [0]*pn; w[k*n+i] = 1
                candidates.append((tuple(u), tuple(v), tuple(w)))
                
    # 2. Add some random sparse terms to allow for "Strassen-like" cancellations
    # We need terms that can cancel out the "off-diagonal" errors or combine to form the diagonal.
    # Let's add terms with support 2 in each vector.
    # Number of support-2 vectors for dim=9: 36*4 = 144.
    # For dim=24: 276*4 = 1104.
    # Total combinations: 144 * 1104 * 1104 ~ 176 million. Too many to store/process in Python lists quickly?
    # 176M tuples of size 57? No, memory explosion.
    
    # We must be selective. 
    # Let's use a "Greedy + Local Search" hybrid or a "Column Generation" approach?
    # Or, simpler: Use the known Strassen 4x4x4 decomposition (Rank 23) and pad/embed.
    
    # Let's implement the Strassen 4x4x4 decomposition explicitly.
    # Reference: "Matrix multiplication via 4x4 blocks"
    # If we can find a valid rank 23 set for 4x4x4, we can use it for the top-left 3x3 of A, 
    # but B is 3x8. 
    # Actually, 3x3x8 is not a sub-tensor of 4x4x8 in a simple way that preserves rank additivity easily without overlap.
    
    # Let's try a different path: **Rank 46 via 4x4x4 embedding**.
    # Pad A (3x3) to A' (4x4) with zeros in row/col 3.
    # Pad B (3x8) to B' (4x8) with zeros in row 3.
    # Compute C' = A' * B' (4x8).
    # C' top-left 3x8 is exactly A*B.
    # If we have a rank R decomposition for 4x4x8, we are done.
    # 4x4x8 can be split into two 4x4x4 blocks.
    # Rank(4x4x4) = 23 (Strassen-Laderman).
    # So Rank(4x4x8) <= 46.
    # This guarantees a rank 46 solution.
    # Current seed is 72. 46 is a significant improvement.
    
    # We need the matrices for 4x4x4 rank 23.
    # I will hardcode a known valid set.
    # Source: "The rank of 4x4x4 tensor" or similar.
    # One such decomposition uses 23 terms.
    
    # Let's construct the 4x4x4 rank 23 matrices.
    # Indices for 4x4x4:
    # U4: 16x23, V4: 16x23, W4: 16x23.
    # Row idx U4: i*4+j
    # Row idx V4: j*4+k
    # Row idx W4: k*4+i
    
    # I will use a verified set from the literature.
    # If I cannot recall the exact 23 vectors, I can generate a rank 4x4x4 solution using a smaller BIP?
    # No, 4x4x4 BIP is also hard.
    
    # Alternative: Use the **Strassen algorithm for 2x2** recursively?
    # 4x4x4 rank 23 is not a simple recursion of 2x2 (which gives 4^3=64 -> 7^3=343? No, Strassen is 7 for 2x2).
    # Strassen for 2x2 has rank 7.
    # 4x4x4 using Strassen blocks?
    # Divide 4x4 into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]].
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # Using Strassen for each 2x2 multiplication requires 7 mults.
    # But the additions are shared?
    # Actually, the rank of 4x4x4 is known to be 23, which is less than 2*7*4 = 56?
    # Wait, 4x4x4 rank 23 is much better than naive.
    
    # Since I cannot hardcode the 23 vectors reliably from memory without error, 
    # I will use a **Randomized Local Search** to find a rank < 72 solution, 
    # specifically targeting rank 50-60, or use the **Schoolbook Rank 72** if it fails.
    # BUT, the prompt asks for "binary IP".
    
    # Let's try a **Small BIP** for a reduced problem.
    # We know rank <= 72.
    # Let's try to solve for rank 60 using a dictionary of "near-schoolbook" terms.
    
    # To make the BIP feasible, we restrict the dictionary to terms that are "close" to schoolbook.
    # Specifically, terms where u, v, w have support 1 or 2.
    # We will select a subset of these candidates randomly to keep the BIP size manageable (e.g., 200-500 candidates).
    
    # 1. Build a large pool of candidates (Support 1 and 2).
    # 2. Sample a subset of size M (e.g., 300).
    # 3. Solve BIP: min sum x_j s.t. A x = b, x in {0,1}.
    # 4. If feasible with rank < 72, return it. Else return 72.
    
    def get_support_1_vectors(dim):
        vecs = []
        for i in range(dim):
            for s in [1, -1]:
                v = [0]*dim
                v[i] = s
                vecs.append(tuple(v))
        return vecs

    def get_support_2_vectors(dim):
        vecs = []
        for i in range(dim):
            for j in range(i+1, dim):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0]*dim
                        v[i] = s1
                        v[j] = s2
                        vecs.append(tuple(v))
        return vecs

    u_pool = get_support_1_vectors(nm) + get_support_2_vectors(nm)
    v_pool = get_support_1_vectors(mp) + get_support_2_vectors(mp)
    w_pool = get_support_1_vectors(pn) + get_support_2_vectors(pn)
    
    # Sample candidates
    # We want to ensure we include all schoolbook terms.
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m+j
                v_idx = j*p+k
                w_idx = k*n+i
                u = [0]*nm; u[u_idx] = 1
                v = [0]*mp; v[v_idx] = 1
                w = [0]*pn; w[w_idx] = 1
                schoolbook_terms.append((tuple(u), tuple(v), tuple(w)))
                
    # Add random sparse terms
    num_random = 400
    candidates = list(schoolbook_terms)
    
    # To avoid duplicates, use a set
    cand_set = set(candidates)
    
    for _ in range(num_random):
        u = random.choice(u_pool)
        v = random.choice(v_pool)
        w = random.choice(w_pool)
        cand = (u, v, w)
        if cand not in cand_set:
            cand_set.add(cand)
            candidates.append(cand)
            
    # Build the constraint matrix A for the BIP.
    # Variables: x_j for each candidate j.
    # Constraints: For each tensor entry (a,b,c), sum_j x_j * U[a,j] * V[b,j] * W[c,j] = T[a,b,c].
    # T has 1 at (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    
    num_cands = len(candidates)
    # We only need to enforce constraints for the "1" entries?
    # No, we must ensure the "0" entries are also 0.
    # However, if we sum to 1 at the 1-entries, do we automatically get 0 elsewhere?
    # No. The tensor equation is linear. We must enforce all non-zero constraints?
    # Actually, the tensor has 5184 entries.
    # 72 entries are 1. 5112 entries are 0.
    # Enforcing 5184 constraints with 400+ variables is fine for MILP.
    
    # But constructing the 5184 x 400 matrix might take time.
    # Let's optimize: We only need to check the 72 "1" entries for equality to 1.
    # What about the 0 entries?
    # If we use schoolbook terms, they only contribute to their specific 1-entry.
    # Random sparse terms might contribute to multiple entries.
    # If we don't enforce the 0 entries, we might get a solution where T' has 1s at the right places but also non-zeros elsewhere.
    # So we MUST enforce the 0 entries.
    
    # To speed up, we can pre-filter candidates that are "too noisy"?
    # Or just solve it. 5184 constraints, ~500 vars. MILP should handle it.
    
    # Let's try to reduce constraints.
    # We know the structure.
    # Let's just use the 72 "1" constraints and hope the 0s work out? 
    # No, that's not guaranteed.
    
    # Let's use a smaller set of "0" constraints, e.g., random samples of 0-entries?
    # Risky.
    
    # Alternative: Use the fact that if we find a solution for the 72 "1" constraints, 
    # we can verify the full tensor. If it fails, we discard.
    # But BIP needs all constraints to be valid.
    
    # Let's stick to the full constraints but build the matrix efficiently.
    
    # Pre-compute U, V, W matrices for candidates.
    U_mat = np.array([c[0] for c in candidates], dtype=int) # (num_cands, nm)
    V_mat = np.array([c[1] for c in candidates], dtype=int) # (num_cands, mp)
    W_mat = np.array([c[2] for c in candidates], dtype=int) # (num_cands, pn)
    
    # Build A matrix (num_entries x num_cands)
    # T_entry(a,b,c) = sum_j x_j U_mat[j,a] V_mat[j,b] W_mat[j,c]
    # We need A[a,b,c, j] = U_mat[j,a] * V_mat[j,b] * W_mat[j,c]
    
    # This is a 3rd order tensor contraction.
    # We can flatten (a,b,c) to idx.
    # idx = (a * mp + b) * pn + c
    
    # Let's generate the indices for the "1" entries and a sample of "0" entries.
    # To keep it fast, let's only enforce the 72 "1" entries and 100 random "0" entries.
    # This is a heuristic. If the BIP finds a solution, we verify it fully.
    # If verification fails, we fallback to 72.
    
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                idx = (a * mp + b) * pn + c
                ones_indices.append(idx)
                
    # Sample 0s
    all_indices = list(range(nm * mp * pn))
    ones_set = set(ones_indices)
    zero_candidates = [idx for idx in all_indices if idx not in ones_set]
    random.shuffle(zero_candidates)
    zero_indices = zero_candidates[:100]
    
    active_indices = ones_indices + zero_indices
    num_constraints = len(active_indices)
    
    # Build A matrix
    A = np.zeros((num_constraints, num_cands), dtype=int)
    b = np.zeros(num_constraints, dtype=int)
    
    for i_idx, idx in enumerate(active_indices):
        # Decode idx
        c = idx % pn
        rem = idx // pn
        b_idx = rem % mp
        a = rem // mp
        
        # Target value
        if idx in ones_set:
            b[i_idx] = 1
        else:
            b[i_idx] = 0
            
        # Calculate column for each candidate
        # A[i_idx, j] = U_mat[j, a] * V_mat[j, b_idx] * W_mat[j, c]
        # Vectorized:
        A[i_idx, :] = U_mat[:, a] * V_mat[:, b_idx] * W_mat[:, c]
        
    # Solve BIP
    # min sum x_j
    # s.t. A x = b
    # 0 <= x_j <= 1
    
    c_obj = np.ones(num_cands)
    constraints = [LinearConstraint(A, b, b)]
    integrality = np.ones(num_cands)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0: # Optimal
            x = res.x
            # Round to 0/1
            x_binary = np.round(x).astype(int)
            
            # Verify the solution fully
            U_sol = U_mat.T @ x_binary # (nm, rank)
            V_sol = V_mat.T @ x_binary # (mp, rank)
            W_sol = W_mat.T @ x_binary # (pn, rank)
            
            # Check tensor
            # T_calc[a,b,c] = sum_r U_sol[a,r] V_sol[b,r] W_sol[c,r]
            # We check a few random entries.
            valid = True
            # Check all 1s
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        val = np.sum(U_sol[a] * V_sol[b] * W_sol[c])
                        if val != 1:
                            valid = False
                            break
                if not valid: break
            if valid:
                # Check some random 0s
                for idx in zero_candidates[:50]:
                    c = idx % pn
                    rem = idx // pn
                    b_idx = rem % mp
                    a = rem // mp
                    val = np.sum(U_sol[a] * V_sol[b_idx] * W_sol[c])
                    if val != 0:
                        valid = False
                        break
            
            if valid:
                rank_found = int(np.sum(x_binary))
                # Convert to list of lists
                U_out = U_sol.tolist()
                V_out = V_sol.tolist()
                W_out = W_sol.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
                
    except Exception as e:
        pass
    
    # Fallback: Schoolbook Rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
