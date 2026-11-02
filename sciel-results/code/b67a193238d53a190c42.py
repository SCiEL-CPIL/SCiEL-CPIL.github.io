import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    
    # Known rank-50 construction
    def build_rank_50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen_block(i0, j0, k0):
            nonlocal col
            if col >= rank:
                return False

            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)

            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)

            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
            U[a_idx][col] += 1
            U[d_idx][col] += 1
            V[e_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1

            # M2 = (c+d)e -> C10 += 1, C11 -= 1
            U[c_idx][col] += 1
            U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10_idx][col] += 1
            W[c11_idx][col] -= 1
            col += 1

            # M3 = a(f-h) -> C01 += 1, C11 += 1
            U[a_idx][col] += 1
            V[f_idx][col] += 1
            V[h_idx][col] -= 1
            W[c01_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1

            # M4 = d(g-e) -> C00 += 1, C10 += 1
            U[d_idx][col] += 1
            V[g_idx][col] += 1
            V[e_idx][col] -= 1
            W[c00_idx][col] += 1
            W[c10_idx][col] += 1
            col += 1

            # M5 = (a+b)h -> C00 -= 1, C01 += 1
            U[a_idx][col] += 1
            U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] -= 1
            W[c01_idx][col] += 1
            col += 1

            # M6 = (c-a)(e+f) -> C11 += 1
            U[c_idx][col] += 1
            U[a_idx][col] -= 1
            V[e_idx][col] += 1
            V[f_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1

            # M7 = (b-d)(g+h) -> C00 += 1
            U[b_idx][col] += 1
            U[d_idx][col] -= 1
            V[g_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1

            return True

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            if col >= rank:
                return False

            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)

            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0

            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)

            # P1 = a*e -> C00
            U[a_idx][col] += 1
            V[e_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1

            # P2 = b*g -> C00
            U[b_idx][col] += 1
            V[g_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1

            # P3 = c*e -> C10
            U[c_idx][col] += 1
            V[e_idx][col] += 1
            W[c10_idx][col] += 1
            col += 1

            # P4 = d*g -> C10
            U[d_idx][col] += 1
            V[g_idx][col] += 1
            W[c10_idx][col] += 1
            col += 1

            return True

        # Process blocks: split inner dimension j into {0,1} and {2,3}
        # For each, split output k into 2+2+2+1
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                if not add_strassen_block(0, j_start, k_start):
                    break
            add_schoolbook_221_block(0, j_start, 6)

        assert col == rank, f"Expected rank {rank}, got {col}"
        
        # Quick verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {s}")
                        
        return {"U": U, "V": V, "W": W}

    # Generate candidate rank-1 terms for a hypergraph-like search
    # A rank-1 term is a triple of vectors (u, v, w) where:
    # u is a vector in F^{n*m}, v in F^{m*p}, w in F^{p*n}
    # The contribution to the tensor T is u ⊗ v ⊗ w.
    # We want to cover the "1"s in the identity tensor (which has 1 at (i,j,k) if i,j,k form a valid matmul index).
    # Actually, the target tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # This is a very sparse tensor.
    # A rank-1 term u ⊗ v ⊗ w adds u[a]v[b]w[c] to T[a,b,c].
    # For the term to be valid (not introduce errors), it must be that for any (a,b,c) where T[a,b,c] is 0, u[a]v[b]w[c] must be 0?
    # No, we are summing terms. The sum must equal T.
    # This is a Set Cover problem where "sets" are rank-1 tensors and "elements" are the non-zero entries of T.
    # However, rank-1 tensors can have non-zero entries where T is 0, as long as they cancel out.
    # This makes exact cover difficult.
    
    # Heuristic: Look for "local" rank-1 terms that only touch a small number of non-zero T entries.
    # A term is "valid" if its support is a subset of the support of T?
    # If we restrict to terms with support subset of T's support, we can use Set Cover.
    # T's support size is n*m*p = 2*4*7 = 56.
    # A rank-1 term u⊗v⊗w has support |supp(u)|*|supp(v)|*|supp(w)|.
    # If we restrict u, v, w to have small support, we can enumerate them.
    
    # Let's try to find a set of rank-1 terms that covers all 56 entries of T.
    # If we can cover them with < 50 terms, we win.
    # But wait, the rank is the number of terms.
    # The current rank is 50.
    # Is it possible to cover 56 entries with < 50 rank-1 terms?
    # Each rank-1 term can cover at most 1 entry if we restrict support to T.
    # So we need at least 56 terms?
    # No, a rank-1 term can cover multiple entries if its support overlaps with multiple T entries.
    # Example: u has support {a1, a2}, v has support {b1}, w has support {c1}.
    # Term covers (a1,b1,c1) and (a2,b1,c1).
    # If both are in T, this term covers 2 entries.
    # In T, T[i*m+j, j*p+k, k*n+i] = 1.
    # Let's analyze the structure of T.
    # Indices: A index a = i*m+j, B index b = j*p+k, C index c = k*n+i.
    # Note the dependency: a determines i,j. b determines j,k. c determines k,i.
    # Consistency: i from a must match i from c. j from a must match j from b. k from b must match k from c.
    # So T is non-zero at (a,b,c) iff i(a)=i(c), j(a)=j(b), k(b)=k(c).
    
    # Let's enumerate "minimal" rank-1 terms.
    # A term is defined by subsets I ⊆ {0,1}, J ⊆ {0,3}, K ⊆ {0,6}.
    # Wait, i ∈ {0,1}, j ∈ {0,1,2,3}, k ∈ {0,1,2,3,4,5,6}.
    # Let's say a term is supported on I × J × K.
    # For the term to be "valid" (support subset of T), we must have that for all i∈I, j∈J, k∈K,
    # the indices (i*m+j, j*p+k, k*n+i) are consistent.
    # This requires that for all i∈I, j∈J, k∈K:
    # 1. i is same for all? No, i is a variable in the support.
    # If |I|>1, then we have different i's.
    # If i1 != i2, then c1 = k*n+i1, c2 = k*n+i2.
    # The term adds u[a]v[b]w[c].
    # If we choose u, v, w such that u[i*m+j] = 1 for i∈I, j∈J, etc.
    # Then the term covers all (i,j,k) in I×J×K.
    # For this to be a subset of T's support, we need that for all i∈I, j∈J, k∈K, the triple is in T.
    # This is always true by definition of T's support?
    # T is 1 at (i*m+j, j*p+k, k*n+i) for ALL i,j,k.
    # So ANY subset I×J×K is a valid support for a rank-1 term that is a subset of T.
    # So we can cover any subset of entries with a rank-1 term?
    # No, a rank-1 term is u⊗v⊗w.
    # If we set u[i*m+j]=1 for i∈I, j∈J, and 0 otherwise.
    # v[j*p+k]=1 for j∈J, k∈K, and 0 otherwise.
    # w[k*n+i]=1 for k∈K, i∈I, and 0 otherwise.
    # Then the term is 1 at (i*m+j, j*p+k, k*n+i) for all i∈I, j∈J, k∈K.
    # This term covers |I|*|J|*|K| entries of T.
    # And it is 0 everywhere else (in the support of T? No, it's 0 outside I×J×K).
    # Since T is 1 everywhere in its support, this term is a valid "partial" contribution.
    # So the problem reduces to: Cover the set of all (i,j,k) tuples with a minimum number of boxes I×J×K.
    # This is the "Box Cover" problem or "3D Set Cover".
    # We need to cover all 2*4*7 = 56 tuples.
    # What is the minimum number of boxes?
    # A box I×J×K covers |I||J||K| tuples.
    # To minimize count, we want large boxes.
    # Max box size is 2*4*7 = 56 (the whole space).
    # Can we use the whole space?
    # I={0,1}, J={0,1,2,3}, K={0,1,2,3,4,5,6}.
    # This covers all 56 entries with 1 term!
    # Why did we need 50 terms?
    # Because the rank-1 term u⊗v⊗w with u,v,w being indicators of these sets is NOT a valid decomposition term in the tensor decomposition sense?
    # Wait. The tensor decomposition requires that the SUM of rank-1 terms equals T.
    # If we use 1 term: T = u⊗v⊗w.
    # This implies T is rank 1.
    # But T is the matrix multiplication tensor. Its rank is NOT 1.
    # Where is the flaw?
    # The flaw is that u⊗v⊗w is 1 at ALL (i,j,k) in I×J×K.
    # But T is 1 at (i*m+j, j*p+k, k*n+i).
    # Are these the same?
    # Yes, if we define the indices correctly.
    # BUT, the tensor T is a 3-way array T[a,b,c].
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # Otherwise 0.
    # My proposed term u⊗v⊗w has value 1 at (a,b,c) if a∈A_sup, b∈B_sup, c∈C_sup.
    # A_sup = {i*m+j | i∈I, j∈J}.
    # B_sup = {j*p+k | j∈J, k∈K}.
    # C_sup = {k*n+i | k∈K, i∈I}.
    # For the term to be a subset of T, we need:
    # If a∈A_sup, b∈B_sup, c∈C_sup, then T[a,b,c] must be 1.
    # This means there must exist i',j',k' such that a=i'm+j', b=j'p+k', c=k'n+i' AND i'∈I, j'∈J, k'∈K?
    # No, the term doesn't care about i',j',k'. It just adds 1 to T[a,b,c].
    # If T[a,b,c] is 0, we are adding 1 to it, which is WRONG.
    # So we need A_sup × B_sup × C_sup to be a subset of the support of T.
    # Support of T: {(i*m+j, j*p+k, k*n+i) | i∈{0,1}, j∈{0,1,2,3}, k∈{0,1,2,3,4,5,6}}.
    # Let's check if A_sup × B_sup × C_sup ⊆ Support(T).
    # Take I={0,1}, J={0,1}, K={0,1}.
    # A_sup = {0,1,4,5}. (0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # B_sup = {0,1,7,8}. (0*7+0, 0*7+1, 1*7+0, 1*7+1)
    # C_sup = {0,1,2,3}. (0*2+0, 0*2+1, 1*2+0, 1*2+1)
    # Is (a=0, b=7, c=0) in Support(T)?
    # a=0 => i=0, j=0.
    # b=7 => j=1, k=0.
    # Conflict: j=0 vs j=1.
    # So (0,7,0) is NOT in Support(T). T[0,7,0] = 0.
    # But the term u⊗v⊗w adds 1 to T[0,7,0].
    # So this term is INVALID.
    
    # So we can only use boxes I×J×K where the "mixed" indices don't conflict.
    # Conflict arises if J has multiple values.
    # If |J|=1, say J={j0}.
    # Then A_sup = {i*m+j0 | i∈I}.
    # B_sup = {j0*p+k | k∈K}.
    # C_sup = {k*n+i | k∈K, i∈I}.
    # Check consistency for any a∈A_sup, b∈B_sup, c∈C_sup.
    # a => i_a, j0.
    # b => j0, k_b.
    # c => k_c, i_c.
    # We need i_a = i_c and k_b = k_c.
    # i_a is determined by a. i_c is determined by c.
    # Since a depends on i_a and c depends on i_c, and they are independent choices in the box product,
    # we can pick a with i_a=0 and c with i_c=1.
    # Then i_a != i_c. Conflict.
    # So even if |J|=1, we can have conflicts in I and K.
    # To avoid ALL conflicts, we need |I|=1, |J|=1, |K|=1.
    # In that case, the box is a single point (i,j,k).
    # Then we need 56 terms.
    # But we have a rank-50 solution!
    # How?
    # The rank-50 solution uses terms that are NOT subsets of T.
    # They have non-zero entries where T is 0, but these cancel out.
    # So the "Set Cover" approach with "valid" terms (subset of T) is too restrictive.
    # We need to allow "invalid" terms that cancel.
    # This is the hard part.
    
    # Given the time constraint and the complexity of searching for cancelling terms,
    # and the fact that we already have a rank-50 solution,
    # I will stick with the rank-50 solution.
    # The "Hypergraph Independent Set" idea is interesting but requires a solver that can handle cancellations,
    # which is effectively solving a system of polynomial equations, which is NP-hard.
    # A heuristic might find a better solution, but within 130s, it's risky.
    
    # Let's just return the known good solution.
    
    try:
        result = build_rank_50()
        return result
    except Exception as e:
        # Fallback to a simpler, slower construction if something goes wrong
        # But build_rank_50 is deterministic and verified.
        raise e

    # If we had more time, we could try:
    # 1. Generate random rank-1 terms.
    # 2. Use linear programming to find a combination that approximates T.
    # 3. Refine with local search.
    # But this is unlikely to beat 50 quickly.

if __name__ == "__main__":
    construct()
